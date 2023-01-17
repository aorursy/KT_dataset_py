from pathlib import Path

import numpy as np

import tensorflow as tf

import os, math, cv2, glob, random

%matplotlib inline

import matplotlib.pyplot as plt

from IPython.display import clear_output

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
dataset_path = '../input/chest_xray/chest_xray/'



IMG_SIZE = 96

CATEGORIES = ['NORMAL', 'PNEUMONIA']



def generate_data(data_type):

    dataset = []

    for category in CATEGORIES:

        path = f'{dataset_path}/{data_type}/{category}/'

        class_id = CATEGORIES.index(category)

        for image in os.listdir(path):

            try:

                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

                image_array = cv2.resize(image_array, (IMG_SIZE , IMG_SIZE))

                image_array.shape

                dataset.append([image_array, class_id])

            except Exception as e:

                print(e)

    random.shuffle(dataset)

    return dataset

                

def get_numpy_data(dataset):  

    data = []

    labels = []

    for features, label in dataset:

        data.append(features)

        labels.append(label)

    data = np.array(data)

    data.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    return data, labels

test_dataset = generate_data('val')

test_data, test_labels = get_numpy_data(test_dataset)



train_dataset_path = '../input/chest_xray/chest_xray/train'

test_dataset_path = '../input/chest_xray/chest_xray/test'

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

         train_dataset_path,

         target_size = (128, 128),

         batch_size = 128,

         class_mode = 'binary')



test_generator = test_datagen.flow_from_directory(

        test_dataset_path,

         target_size = (128, 128),

         batch_size = 128,

         class_mode = 'binary')
plt.figure(figsize=(50, 50))

i = 0

for i in range(16):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(test_data[i])

    if(test_labels[i] == 0):

        plt.xlabel('NORMAL').set_size(50)

    else:

        plt.xlabel('PNEUMONIA').set_size(54)

    i += 1

plt.show()
l = tf.keras.layers

bnmomemtum=0.9

def fire(x, squeeze, expand):

  y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)

  y  = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)

  y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)

  y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)

  y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)

  y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)

  return tf.keras.layers.concatenate([y1, y3])



def fire_module(squeeze, expand):

  return lambda x: fire(x, squeeze, expand)



x = l.Input(shape=[128, 128, 3])

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(x)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', use_bias=True, activation='relu')(x)

y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)

y = fire_module(24, 48)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(64, 128)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(24, 48)(y)



y = tf.keras.layers.Flatten()(y)



y = tf.keras.layers.Dense(128)(y)

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(y)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Dropout(0.33)(y)



y = tf.keras.layers.Dense(48)(y)

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(y)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Dropout(0.33)(y)



y = tf.keras.layers.Dense(18)(y)

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(y)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Dropout(0.33)(y)



y = tf.keras.layers.Dense(6)(y)

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(y)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Dropout(0.33)(y)



y = tf.keras.layers.Dense(1, activation='sigmoid')(y)

model = tf.keras.Model(x, y)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01),

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.summary()

# binary_crossentropy 

# sparse_categorical_crossentropy 

# categorical_crossentropy

# binary_crossentropy
def lr_decay(epoch):

  return 0.01 * math.pow(0.666, epoch)

callback_learning_rate = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)



class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.99):

      print("\nReached 99% accuracy so cancelling training!")

      self.model.stop_training = True



callbacks_max_acc = myCallback()

class PlotLosses(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.plot(self.x, self.val_losses, label="val_loss")

        plt.legend()

        plt.show();

        

plot_losses = PlotLosses()



callback_is_nan = tf.keras.callbacks.TerminateOnNaN()



callback_early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',

                                                       min_delta=0,

                                                       patience=0,

                                                       verbose=0,

                                                       mode='auto',

                                                       baseline=None,

                                                       restore_best_weights=False)



callback_svg_logger = tf.keras.callbacks.CSVLogger('training.log', separator=',', append=False)
STEP_SIZE_TRAIN=train_generator.n // train_generator.batch_size 

STEP_SIZE_VALID=test_generator.n // test_generator.batch_size

EPOCHS=15

history = model.fit_generator(

      train_generator,

      steps_per_epoch=STEP_SIZE_TRAIN, 

      epochs=EPOCHS,

      validation_data=test_generator,

      validation_steps=STEP_SIZE_VALID,

      callbacks=[ callback_is_nan, 

                 callback_learning_rate,

                 callback_early_stop,

                 plot_losses,

                 callback_svg_logger,

                 callbacks_max_acc])
accuracy = history.history['acc']

loss = history.history['loss']

val_accuracy = history.history['val_acc']

val_loss = history.history['val_loss']



print(f'Training Accuracy: {np.max(accuracy)}')

print(f'Training Loss: {np.min(loss)}')

print(f'Validation Accuracy: {np.max(val_accuracy)}')

print(f'Validation Loss: {np.min(val_loss)}')
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
class_names = ['NORMAL', 'PNEUMONIA']

def plot_images(i, predictions_array, true_labels, images):

    predictions_array, true_label, img = predictions_array[i], true_labels[i],images[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    

    plt.imshow(img)

    

    predicted_label = np.argmax(predictions_array)

        

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label.set_title('style: {!r}'.format(sty), color='C0')],

                                        100*np.max(predictions_array),

                                        class_names[true_label]))

    

random.shuffle(test_data)

predictions = model.predict(test_data)





num_rows = 4

num_cols = 4

num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_images(i, predictions, test_labels, test_data)