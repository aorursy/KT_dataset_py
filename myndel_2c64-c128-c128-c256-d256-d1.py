import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

import os, cv2, random

import pandas as pd

import numpy as np

tf.random.set_seed(1)

np.random.seed(1)



TRAIN_DIR = 'train/'

TEST_DIR = 'test/'

IMG_SIZE = 64

CHANNELS = 3  # 1 for cv2.IMREAD_GRAYSCALE

BATCH_SIZE = 32



class_names = ('cat', 'dog')
# Unzip data

os.system('unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip')

os.system('unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip')
# For full dataset uncomment

train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]

test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)]



#train_images = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)[:100]]

#test_images = [TEST_DIR + i for i in os.listdir(TEST_DIR)[:100]]



random.shuffle(train_images)



print(f'Loaded {len(train_images)} train images')

print(f'Loaded {len(test_images)} test images')
def read_img(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)  # cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)





def prep_data(images, test_data=False):

    count = len(images)

    if test_data:

        data = np.ndarray((count, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.float16)

    else:

        data = np.ndarray((count, IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        img = read_img(image_file)

        data[i] = img



    return data
x_train = prep_data(train_images)

y_train = [1 if 'dog' in image_path else 0 for image_path in train_images]



del train_images



y_train = np.array(y_train, dtype=np.uint8)



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
model = Sequential()



model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(1, activation='sigmoid'))





model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=40, validation_data=(x_test, y_test), verbose=1)
# Save model

#model.save('my_model_image_generated')
# Load model

#model = keras.models.load_model('my_model_image_generated')
# Plot val & acc

fig, ax = plt.subplots(2,1)



ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image

        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)



history = model.fit_generator(

    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),

    epochs=40,

    validation_data=(x_test, y_test),

    verbose=1,

    steps_per_epoch=x_train.shape[0] // BATCH_SIZE)
x_test = prep_data(test_images)

results = model.predict(x_test)



results = np.round(results)

series = pd.Series(results[i][0] for i in range(len(results)))

series = pd.Series(series, name='label')



submission = pd.concat([pd.Series(range(1,12501), name = "id"), series], axis = 1)



submission.to_csv("submission.csv", index=False)