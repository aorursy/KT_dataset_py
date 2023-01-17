import numpy as np

import tensorflow as tf

import os, math, cv2, glob, random

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input



train_dataset_path = '../input/10-monkey-species/training/training'

validation_dataset_path  = '../input/10-monkey-species/validation/validation'



# width 550 height 367

IMAGE_HEIGHT  = 112

IMAGE_WIDTH   = 112

IMAGE_SIZE    = (IMAGE_HEIGHT, IMAGE_WIDTH)



CATEGORIES    = os.listdir(train_dataset_path)

NUM_CLASSES   = len(CATEGORIES)

BATCH_SIZE    = 8 

EPOCHS        = 60
def generate_data(path_data):

    dataset = []

    for category in CATEGORIES:

        path = f'{path_data}/{category}/'

        class_id = CATEGORIES.index(category)

        for image in os.listdir(path):

            try:

                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR) 

                image_array = cv2.resize(image_array, (IMAGE_HEIGHT, IMAGE_WIDTH))  

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

    data = data / 255.0

    return data, labels



train_data, train_label = generate_data(train_dataset_path)

data, labels = generate_data(validation_dataset_path)

validation_data, test_data, validation_label, _ = train_test_split(data, labels, test_size=0.15)



datagen_train = ImageDataGenerator(rescale=1.0/255.,

                                  rotation_range=35,

                                  width_shift_range=0.2,

                                  height_shift_range=0.2,

                                  shear_range=0.2,

                                  zoom_range=0.2,

                                  horizontal_flip=True)



datagen_test = ImageDataGenerator(rescale=1.0/255.,)

datagen_validation = ImageDataGenerator(rescale=1.0/255.,)



datagen_train.fit(train_data)

datagen_validation.fit(validation_data) 

datagen_test.fit(test_data)
plt.figure(figsize=(IMAGE_HEIGHT/4, IMAGE_WIDTH/4))

i = 0

for i in range(15):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(test_data[i])

    plt.xlabel(_[i], color='tomato')

    i += 1

plt.show()
def lr_decay(epoch):

    lr = 0.01 * math.pow(0.77, epoch);

    if lr <= 0.000001:

        return 0.000001

    else:

        return lr

lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)



callback_early_stop = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, mode='auto', baseline=None, restore_best_weights=False)



tensorboard = tf.keras.callbacks.TensorBoard("logs")



reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(

                              monitor='loss',

                              mode='min',

                              factor=0.7,

                              patience=1,

                              min_lr=0.000001,

                              verbose=1)
l = tf.keras.layers

bnmomemtum=0.9

def fire(x, squeeze, expand):

  y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)

  y  = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)

  y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)

  y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)

  y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)

  y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)

  return tf.keras.layers.concatenate([y, y3])



def fire_module(squeeze, expand):

  return lambda x: fire(x, squeeze, expand)



x = l.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3])

y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum, axis=1)(x)

y = tf.keras.layers.Activation('relu')(y)



y = tf.keras.layers.Conv2D(kernel_size=3, filters=12, padding='same', activation='relu')(y)

y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)



y = fire_module(12, 24)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)



y = fire_module(24, 48)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)





y = fire_module(48, 96)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)



y = fire_module(24, 48)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)



y = fire_module(12, 24)(y)

y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)



y = tf.keras.layers.GlobalAveragePooling2D()(y)



y = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(y)

model = tf.keras.Model(x, y)



model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
%load_ext tensorboard.notebook

%tensorboard --logdir logs
from keras.utils import to_categorical

train_labels = to_categorical(train_label)

validation_labels = to_categorical(validation_label)



history = model.fit_generator(

    datagen_train.flow(train_data, train_labels, batch_size=BATCH_SIZE),

    steps_per_epoch=len(train_data) / BATCH_SIZE,

    epochs=EPOCHS,

    verbose=1,

    validation_data=datagen_validation.flow(validation_data,     

    validation_labels, batch_size=BATCH_SIZE),

    callbacks=[tensorboard, reduce_learning_rate]) # callback_early_stop, lr_decay_callback  reduce_learning_rate,

acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,max(plt.ylim())])

plt.title('Training and Validation Loss')

plt.show()