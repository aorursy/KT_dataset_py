import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D
from keras.utils.np_utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout, Reshape
EPOCHS = 20
NUM_CLASS = 10
BATCH_SIZE = 32
TRAIN_VAL_FRAC = 0.7
IH, IW, IC = 28, 28 , 1
DATASET_PATH = "../input"
model_in = Input([IH * IW * IC])
model_in_reshape = Reshape(target_shape = [IH, IW, IC])(model_in)

conv1 = Conv2D(filters = 8, kernel_size = (7, 7), strides = (2, 2), padding = 'same', 
               activation='relu')(model_in_reshape)
conv2 = Conv2D(filters = 16, kernel_size = (5, 5), strides = (2, 2), padding = 'same', 
               activation='relu')(conv1)
conv3 = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'same', 
               activation='relu')(conv2)
flatten4 = Flatten()(conv3)
flatten4 = Dropout(rate=0.5)(flatten4)
dense5 = Dense(units=256, activation='relu')(flatten4)
dense5 = Dropout(rate=0.5)(dense5)
dense6 = Dense(units=10, activation='softmax')(dense5)

model = Model(model_in, dense6)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
df_train = pd.read_csv(filepath_or_buffer=os.path.join(DATASET_PATH, "train.csv"))
df_test = pd.read_csv(filepath_or_buffer=os.path.join(DATASET_PATH, "test.csv"))
X_train = df_train.values[:, 1:]
X_train = X_train / 255.0
Y_train = to_categorical(df_train.values[:, [0]], num_classes=NUM_CLASS)

X_test = df_test.values
X_test = X_test / 255.0
idx = np.random.choice(np.arange(X_train.shape[0]), X_train.shape[0], replace = False)

X_train = X_train[idx]
Y_train = Y_train[idx]
num_train = int(TRAIN_VAL_FRAC * X_train.shape[0])
X_val = X_train[num_train:]
Y_val = Y_train[num_train:]

X_train = X_train[:num_train]
Y_train = Y_train[:num_train]
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
probs = model.predict(X_test, batch_size = BATCH_SIZE, verbose = 1)
predictions = np.argmax(probs, axis = 1)

submission = pd.DataFrame({
                "ImageId" : np.arange(1, X_test.shape[0] + 1),
                "Label" : predictions
            })
submission.to_csv("kaggle.csv", index = False)