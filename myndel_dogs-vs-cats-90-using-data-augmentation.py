import numpy as np

import pandas as pd

import os

from zipfile import ZipFile

import matplotlib.pyplot as plt

from matplotlib.image import imread

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from sklearn.model_selection import train_test_split



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
TRAIN_ZIP = '../input/dogs-vs-cats/train.zip'

TEST_ZIP = '../input/dogs-vs-cats/test1.zip'

TRAIN_DIR = 'train/'

TEST_DIR = 'test1/'

IMG_SIZE = (100, 100)

EPOCHS = 50

BATCH_SIZE = 32
for zipdir in (TRAIN_ZIP, TEST_ZIP):

    with ZipFile(zipdir, 'r') as zipObj:

       zipObj.extractall()
print(f'Train examples: {len(os.listdir(TRAIN_DIR))}')

print(f'Test examples: {len(os.listdir(TEST_DIR))}')



print(os.listdir(TRAIN_DIR)[:5])
# Plot first few train images

for i in range(9):

    plt.subplot(330 + 1 + i)

    filename = TRAIN_DIR + f'dog.{i}.jpg'

    image = imread(filename)

    plt.imshow(image)

plt.show()
# x - photos

# y - labels; 1 stands for dogs, 0 for cats

photos, labels = list(), list()

for filename in os.listdir(TRAIN_DIR):

    # Determine class

    label = 1.0

    if 'cat' in filename:

        label = 0.0

    # Load image

    image = load_img(TRAIN_DIR + filename, target_size=IMG_SIZE)

    # Convert to numpy array

    image = img_to_array(image) / 255.0

    # Store

    photos.append(image)

    labels.append(label)



photos = np.asarray(photos)

labels = np.asarray(labels)



print(photos.shape, labels.shape)
x_train, x_val, y_train, y_val = train_test_split(photos, labels, test_size=0.2)
model = Sequential()



# 100x100 input

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

# 100x100

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

# 50x50

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

# 50x50

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

# 50x50

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform', padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

# 25x25

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(1024, kernel_initializer='he_uniform'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(1024, kernel_initializer='he_uniform'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



model.add(Dense(1, activation='sigmoid'))



sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
history = model.fit(

    x_train, 

    y_train, 

    batch_size=BATCH_SIZE, 

    epochs=EPOCHS, 

    validation_data=[x_val, y_val])
# Vizualize history

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

    epochs=EPOCHS,

    validation_data=(x_val, y_val),

    verbose=1,

    steps_per_epoch=x_train.shape[0] // BATCH_SIZE)
# Vizualize history with Data Augmentation

fig, ax = plt.subplots(2,1)



ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Clear memory

del x_train, x_val, y_train, y_val, photos, labels, 
test_data = list()

for i in range(7500):

    # Load image

    image = load_img(TEST_DIR + f'{i+1}.jpg', target_size=IMG_SIZE)

    # Convert to numpy array

    image = img_to_array(image) / 255.0

    # Store

    test_data.append(image)



test_data = np.asarray(test_data)



print(test_data.shape)
results = model.predict(test_data)

results = np.round(results)



series = pd.Series(results[i][0] for i in range(len(results)))

series = pd.Series(series, name='label')



submission = pd.concat([pd.Series(range(1, len(results) + 1), name = "id"), series], axis = 1)



submission.to_csv("submission.csv", index=False)