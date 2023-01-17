import numpy as np

import tensorflow as tf

import os, math, cv2, glob, random

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)


IMG_SIZE = 50

CATEGORIES = ['Parasitized', 'Uninfected']

dataset = []



def generate_data():

    for category in CATEGORIES:

        path = f'../input/cell_images/cell_images/{category}'

        class_id = CATEGORIES.index(category)

        for image in os.listdir(path):

            try:

                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

                image_array = cv2.resize(image_array, (IMG_SIZE , IMG_SIZE))

                dataset.append([image_array, class_id])

            except Exception as e:

                print(e)

    random.shuffle(dataset)

                

generate_data()



data = []

labels = []

for features, label in dataset:

    data.append(features)

    labels.append(label)

    

data = np.array(data)

data.reshape(-1, 50, 50, 3)



train_data, data, train_labels, labels = train_test_split(data,labels,test_size=0.15)

test_data, validation_data, test_labels, validation_labels = train_test_split(data,labels,test_size=0.7)



datagen_train = ImageDataGenerator(rescale=1.0/255.,

                            rotation_range=45,

                            width_shift_range=0.2,

                            height_shift_range=0.2,

                            shear_range=0.2,

                            zoom_range=0.2,

                            horizontal_flip=True)



datagen_test = ImageDataGenerator(rescale=1.0/255.)

datagen_validation = ImageDataGenerator(rescale=1.0/255.)



datagen_train.fit(train_data)

datagen_test.fit(test_data)

datagen_test.fit(validation_data)
plt.figure(figsize=(10, 10))

i = 0

for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(test_data[i])

    if(test_labels[i] == 0):

        plt.xlabel('Infected')

    else:

        plt.xlabel('Uninfected')

    i += 1

plt.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(50, 50, 3)), 

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.Activation('relu'),



    tf.keras.layers.Conv2D(kernel_size=3, filters=12,use_bias=False, padding='same'),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.Activation('relu'),



    

    tf.keras.layers.Conv2D(kernel_size=3, filters=12, use_bias=False, padding='same', strides=2),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.Activation('relu'),



    

    tf.keras.layers.Flatten(),



    tf.keras.layers.Dense(80, use_bias=False),

    tf.keras.layers.BatchNormalization(scale=False),

    tf.keras.layers.Activation('relu'),



    tf.keras.layers.Dropout(0.4),

#     tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()

# sparse_categorical_crossentropy

# binary_crossentropy 

# kullback_leibler_divergence

# categorical_crossentropy
def lr_decay(epoch):

  return 0.01 * math.pow(0.666, epoch)

lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)



class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.97):

      print("\nReached 97% accuracy so cancelling training!")

      self.model.stop_training = True



callbacks_max_acc = myCallback()
BATCH_SIZE=120

epochs=10

history = model.fit_generator(

    datagen_train.flow(train_data, train_labels, batch_size=BATCH_SIZE),

    steps_per_epoch=len(train_data) / BATCH_SIZE,

    epochs=epochs,

    validation_data=datagen_validation.flow(validation_data, 

    validation_labels, batch_size=BATCH_SIZE),

    verbose=1,

    callbacks=[lr_decay_callback, callbacks_max_acc])
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
class_names = ['Infected', 'Uninfected']

def plot_images(i, predictions_array, true_labels, images):

    predictions_array, true_label, img = predictions_array[i], true_labels[i],images[i]

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    

    plt.imshow(img)

    

    predicted_label = np.argmax(predictions_array)

        

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                        100*np.max(predictions_array),

                                        class_names[true_label]))

    

random.shuffle(test_data)

predictions = model.predict(test_data)
num_rows = 5

num_cols = 4

num_images = num_rows * num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    plot_images(i, predictions, test_labels, test_data)