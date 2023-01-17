import numpy as np

import pandas as pd

import matplotlib.pylab as plt

from random import shuffle



import os

import cv2
train_path = "../input/train"

test_path = "../input/test"



ROWS = 64

COLS = 64

CHANNELS = 3



images      = [img for img in os.listdir(train_path)]

images_dog  = [img for img in os.listdir(train_path) if "dog" in img]

images_cat  = [img for img in os.listdir(train_path) if "cat" in img]



images_test = [img for img in os.listdir(test_path)]
#only taking a subset (less accuracy but faster training)

train_dog = images_dog[:1000]

train_cat = images_cat[:1000]

valid_dog = images_dog[1000:1100]

valid_cat = images_cat[1000:1100]



train_list = train_dog + train_cat

valid_list = valid_dog + valid_cat

test_list  = images_test[0:]



shuffle(train_list)



train = np.ndarray(shape=(len(train_list),ROWS, COLS))

train_color = np.ndarray(shape=(len(train_list), ROWS, COLS, CHANNELS), dtype=np.uint8)

test = np.ndarray(shape=(len(test_list),ROWS, COLS))

test_color = np.ndarray(shape=(len(images_test), ROWS, COLS, CHANNELS), dtype=np.uint8)

valid = np.ndarray(shape=(len(valid_list), ROWS, COLS))

valid_color = np.ndarray(shape=(len(valid_list), ROWS, COLS, CHANNELS), dtype=np.uint8)
labels = np.ndarray(len(train_list))



for i, img_path in enumerate(train_list):

    img_color = cv2.imread(os.path.join(train_path, img_path), 1)

    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    

    train[i] = img

    train_color[i] = img_color

   

    if "dog" in img_path:

        labels[i] = 0

    else:

        labels[i] = 1
valid_labels = np.ndarray(len(valid_list))



for i, img_path in enumerate(valid_list):

    img_color = cv2.imread(os.path.join(train_path, img_path), 1)

    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    

    valid[i] = img

    valid_color[i] = img_color

    

    if "dog" in img_path:

        valid_labels[i] = 0

    else:

        valid_labels[i] = 1
for i, img_path in enumerate(test_list):

    img_color = cv2.imread(os.path.join(test_path, img_path), 1)

    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    

    test[i] = img

    test_color[i] = img_color
for i, img_path in enumerate(test_list):

    img_color = cv2.imread(os.path.join(test_path, img_path), 1)

    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    

    test[i] = img

    test_color[i] = img_color
from keras.utils import np_utils



X_train = train_color / 255

X_valid = valid_color / 255

X_test  = test_color  / 255

# one hot encode outputs

y_train = np_utils.to_categorical(labels)

y_valid = np_utils.to_categorical(valid_labels)

num_classes = y_valid.shape[1]
def larger_model():

	# create model

	model = Sequential()

	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(64, 64, 3), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(15, 3, 3, activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

# build the model

model = larger_model()

# Fit the model

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=10, batch_size=200, verbose=2)

# Final evaluation of the model

scores = model.evaluate(X_valid, y_valid, verbose=0)

print("Classification Error: %.2f%%" % (100-scores[1]*100))
submission = model.predict_classes(X_test, verbose=2)
pd.DataFrame({"id": list(range(1,len(test_color)+1)), 

              "label": submission}).to_csv('submission.csv', index=False,header=True)