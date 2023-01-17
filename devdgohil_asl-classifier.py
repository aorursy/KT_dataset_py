import math

import os

from glob import glob

import cv2

from random import randint, sample

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras import Model, layers



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Dropout, Flatten

from keras.regularizers import l2
TARGET_SIZE = (64, 64)

VALIDATION_SPLIT = 0.3



TRAIN_DIR = '../input/asl_alphabet_train/asl_alphabet_train'

TEST_DIR = '../input/asl_alphabet_test/asl_alphabet_test'

CLASSES = [folder[len(TRAIN_DIR) + 1:] for folder in glob(TRAIN_DIR + '/*')]

CLASSES.sort()
def plot_each_sign(base_path):

    cols = 5

    rows = int(np.ceil(len(CLASSES) / cols))

    fig = plt.figure(figsize=(16, 16))

    

    for i in range(len(CLASSES)):

        cls = CLASSES[i]

        img_path = base_path + '/' + cls + '/**'

        path_contents = glob(img_path)

    

        imgs = sample(path_contents, 1)



        sp = plt.subplot(rows, cols, i + 1)

        plt.imshow(cv2.imread(imgs[0]))

        plt.title(cls)

        sp.axis('off')



    plt.show()

    return
plot_each_sign(TRAIN_DIR)
def get_sample_img(letter):

    img_path = TRAIN_DIR + '/' + letter + '/**'

    path_contents = glob(img_path)

    imgs = sample(path_contents, 1)

    img = cv2.resize(cv2.imread(imgs[0]), TARGET_SIZE)

    

    plt.figure(figsize=(32, 32))

    plt.subplot(121)

    plt.imshow(cv2.imread(imgs[0]))

    plt.subplot(122)

    plt.imshow(img)

    return img
sample_img = get_sample_img('A')
data_augmentor = ImageDataGenerator(

    samplewise_center=True, 

    samplewise_std_normalization=True,

    horizontal_flip=True,

    rescale=1.0/255,

    validation_split=VALIDATION_SPLIT)



train_generator = data_augmentor.flow_from_directory(

    '../input/asl_alphabet_train/asl_alphabet_train',

    batch_size=50,

    class_mode='sparse',

    target_size=TARGET_SIZE,

    subset='training')



validation_generator = data_augmentor.flow_from_directory(

    '../input/asl_alphabet_train/asl_alphabet_train',

    batch_size=50,

    class_mode='sparse',

    target_size=TARGET_SIZE,

    subset='validation')
def findKey(indices, search_value):

    for key, value in indices.items():

        if(value == search_value):

            return key

    return -1
train_generator.class_indices
model = Sequential()

model.add(Conv2D(64, kernel_size=(4, 4), activation='relu', input_shape=(64, 64, 3), name="conv1"))

model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', name="conv2"))

model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', name="conv3"))

model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu', name="conv4"))

model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=(4, 4), activation='relu', name="conv5"))

model.add(Conv2D(256, kernel_size=(4, 4), strides=(2, 2), activation='relu', name="conv6"))

model.add(Flatten())

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dense(29, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=0.0001),

              metrics=['accuracy'])
history = model.fit_generator(generator=train_generator,steps_per_epoch=1218, epochs=10, validation_data=validation_generator, validation_steps=522, use_multiprocessing=True) 
# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
def predict_test_imgs():

    cols = 5

    rows = int(np.ceil(len(CLASSES) / cols))

    fig = plt.figure(figsize=(16, 16))

    i = 0

    print(train_generator.class_indices)

    for imgfile in os.listdir(TEST_DIR):

        orgimg = cv2.imread(TEST_DIR + '/' + imgfile)

        img = cv2.resize(orgimg, TARGET_SIZE)

        sp = plt.subplot(rows, cols, i + 1)

        plt.imshow(img)

        pred = model.predict(img.reshape(1, 64, 64, 3))

        pred_class = pred.argmax(axis=-1)

        title = imgfile+' '+str(pred_class)

        plt.title(title)

        sp.axis('off')

        i += 1

    plt.show()

    return
predict_test_imgs()