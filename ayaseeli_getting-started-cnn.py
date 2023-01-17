import numpy as np

import pandas as pd

import os

import cv2
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 128

COLS = 128

CHANNELS = 1



train = os.listdir(TRAIN_DIR)

train = sorted(train,key=lambda x: int(os.path.splitext(x)[0]))

train = [TRAIN_DIR + x for x in train]



test  = os.listdir(TEST_DIR)

test  = sorted(test,key=lambda x: int(os.path.splitext(x)[0]))

test  = [TEST_DIR + x for x in test]
def read_image(file_path):

    img = cv2.imread(file_path, 0)

    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    return img





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train)

test = prep_data(test)



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
label = pd.read_csv("../input/train_labels.csv")

y = label["invasive"]

y.shape
from matplotlib import pyplot as plt

from matplotlib import cm

for i in range(9):

    plt.subplot(331+i)

    plt.imshow(train.reshape(-1,1,128,128)[i][0], cmap=cm.binary)

plt.show()

print(label[0:9])
y_train = y[500:]

y_valid = y[:500]



X_train = train[500:]

X_valid = train[:500]

y_train[0:9]
from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1).astype('float32')

X_valid = X_valid.reshape(X_valid.shape[0], 128, 128, 1).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_valid = X_valid / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_valid = np_utils.to_categorical(y_valid)

num_classes = y_valid.shape[1]

num_classes
def larger_model():

    # create model

    model = Sequential()

    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(128, 128, 1), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# build the model

model = larger_model()

# Fit the model

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=10, batch_size=250, verbose=2)