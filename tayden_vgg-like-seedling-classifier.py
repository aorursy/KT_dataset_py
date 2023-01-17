import tensorflow as tf

import numpy as np

import matplotlib.pylab as plt

import keras

import os

from PIL import Image, ImageOps



%matplotlib inline
SPECIES = [

    "Black-grass",

    "Charlock",

    "Cleavers",

    "Common Chickweed",

    "Common wheat",

    "Fat Hen",

    "Loose Silky-bent",

    "Maize",

    "Scentless Mayweed",

    "Shepherds Purse",

    "Small-flowered Cranesbill",

    "Sugar beet"

]

TRAIN_PATH = './train'

TEST_PATH = './test'

RANDOM_SEED=42

IMG_DIMS = (100, 100)

IMG_BANDS = 3
# Load the data

train_X = []

train_y = []

for sp_id, sp in enumerate(SPECIES):

    for img in os.listdir(os.path.join(TRAIN_PATH, sp)):

        path = os.path.join(TRAIN_PATH, sp, img)

        img = Image.open(path)

        train_X.append(ImageOps.fit(img, IMG_DIMS, Image.ANTIALIAS).convert('RGB'))

        train_y.append(sp_id)

        

test_X = []

for img in os.listdir(TEST_PATH):

    path = os.path.join(TEST_PATH, img)

    img = Image.open(path)

    test_X.append(ImageOps.fit(img, IMG_DIMS, Image.ANTIALIAS).convert('RGB'))
from sklearn.model_selection import train_test_split



# Create the validation set

train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2, random_state=RANDOM_SEED)
# Convert all images to ndarray

train_X = np.array([np.array(im) for im in train_X])

train_X = train_X.reshape(train_X.shape[0], IMG_DIMS[0], IMG_DIMS[1], IMG_BANDS) / 255



val_X = np.array([np.array(im) for im in val_X])

val_X = val_X.reshape(val_X.shape[0], IMG_DIMS[0], IMG_DIMS[1], IMG_BANDS) / 255



test_X = np.array([np.array(im) for im in test_X])

test_X = test_X.reshape(test_X.shape[0], IMG_DIMS[0], IMG_DIMS[1], IMG_BANDS) / 255



# Convert to one-hot labels

train_y = np.asarray(keras.utils.to_categorical(train_y, num_classes=len(SPECIES)))

val_y = np.asarray(keras.utils.to_categorical(val_y, num_classes=len(SPECIES)))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint



model = Sequential()



model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_DIMS[0], IMG_DIMS[1], IMG_BANDS)))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(len(SPECIES), activation='softmax'))



adam = Adam()

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])



checkpoint = ModelCheckpoint('./model.hd5', monitor='val_acc', save_best_only=True, mode='auto', period=1)

history = model.fit(train_X, train_y, batch_size=32, epochs=100, 

                    validation_data=(val_X, val_y),

                    callbacks=[checkpoint])
from keras.models import load_model



model = load_model('./model.hd5')



score = model.evaluate(val_X, val_y, batch_size=128)

print("Validation score: %f" % score[1])
# Generate a submission

predictions = model.predict(test_X, batch_size=128, verbose=0)

predictions = np.argmax(predictions, axis=1)
import csv



output = [[path, SPECIES[predictions[i]]] for i, path in enumerate(os.listdir(TEST_PATH))]



with open('submission.csv', 'w') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')

    writer.writerow(['file', 'species'])

    for r in output:

        writer.writerow(r)    