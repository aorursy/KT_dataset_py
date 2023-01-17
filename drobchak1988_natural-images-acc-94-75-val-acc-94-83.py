import numpy as np

import tensorflow as tf

import os, math, cv2, glob, random

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from random import randint

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

print(tf.__version__)
IMG_WIDTH     = 108

IMG_HEIGTH    = 72

DATASET_PATH  = '../input/data/natural_images/'

data_list     = os.listdir(DATASET_PATH)

IMAGE_SIZE    = (IMG_WIDTH ,IMG_HEIGTH)

NUM_CLASSES   = len(data_list)

BATCH_SIZE    = 64  

EPOCHS        = 15

CATEGORIES    = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
def generate_data():

    dataset = []

    for category in CATEGORIES:

        path = f'../input/data/natural_images/{category}/'

        class_id = CATEGORIES.index(category)

        for image in os.listdir(path):

            try:

                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

                image_array = cv2.resize(image_array, (IMG_WIDTH ,IMG_HEIGTH))

                dataset.append([image_array, class_id])

            except Exception as e:

                print(e)

    random.shuffle(dataset)

    data = []

    labels = []

    for features, label in dataset:

        data.append(features)

        labels.append(label)

    data = np.array(data)

    data.reshape(-1, IMG_WIDTH ,IMG_HEIGTH,  3)

    train_data, data, train_labels, labels = train_test_split(data,labels,test_size=0.15)

    pred_data, validation_data, pred_labels, validation_labels = train_test_split(data,labels,test_size=0.95)

    return train_data, train_labels, pred_data, validation_data, pred_labels, validation_labels



train_data, train_labels, pred_data, validation_data, pred_labels, validation_labels = generate_data()

# labels_one_hot = tf.one_hot(train_labels, NUM_CLASSES + 1)



datagen_train = ImageDataGenerator(rescale=1.0/255., dtype='float32')

datagen_validation = ImageDataGenerator(rescale=1.0/255., dtype='float32')

datagen_pred = ImageDataGenerator(rescale=1.0/255., dtype='float32')



datagen_train.fit(train_data) # len 5864

datagen_validation.fit(validation_data) # len 981

datagen_pred.fit(pred_data) # len 51
plt.figure(figsize=(18, 12))

i = 0

for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(pred_data[i])

    for j in range(len(CATEGORIES)):

        if pred_labels[i] == j:

            plt.xlabel(CATEGORIES[j] , color='tomato').set_size(15)

        j += 1

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



x = l.Input(shape=[IMG_HEIGTH, IMG_WIDTH, 3])

y = tf.keras.layers.BatchNormalization(center=True, scale=False)(x)

y = tf.keras.layers.Activation('relu')(y)

y = tf.keras.layers.Conv2D(kernel_size=3, filters=12, padding='same', use_bias=True, activation='relu')(y)

y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)

y = fire_module(24, 48)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(64, 128)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(64, 128)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)

y = fire_module(24, 48)(y)



y = tf.keras.layers.GlobalAveragePooling2D()(y)

y = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(y)

model = tf.keras.Model(x, y)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.01),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
def lr_decay(epoch):

  return 0.01 * math.pow(0.666, epoch)

callback_learning_rate = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)



class myCallback(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):

    if(logs.get('acc')>0.99):

      print("\nReached 99% accuracy so cancelling training!")

      self.model.stop_training = True

callbacks_max_acc = myCallback()



callback_is_nan = tf.keras.callbacks.TerminateOnNaN()



callback_early_stop = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, mode='auto', baseline=None, restore_best_weights=False)



callback_svg_logger = tf.keras.callbacks.CSVLogger('training.log', separator=',', append=False)
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)

validation_labels = to_categorical(validation_labels)



history = model.fit_generator(

    datagen_train.flow(train_data, train_labels, batch_size=BATCH_SIZE),

    steps_per_epoch=len(train_data) / BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=datagen_validation.flow(validation_data,     

    validation_labels, batch_size=BATCH_SIZE),

    callbacks=[callback_is_nan, 

              callback_learning_rate,

              callback_early_stop,

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