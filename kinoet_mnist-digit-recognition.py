import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import os
print(os.listdir("../input"))

# load data and preprocessing
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
train_Y = to_categorical(train_data.label, 10)
train_X = train_data.drop(labels = ["label"], axis = 1)
train_X = train_X.values.reshape(-1,28,28,1) / 255
test_data = test_data.values.reshape(-1,28,28,1) / 255

train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, random_state=0)
# specify model and compile
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(28,28,1)))
#model.add(Dropout(0.5))
for i in range(4, 9):
    model.add(Conv2D(16 * i, kernel_size=(i, i), activation='relu'))
    model.add(Dropout(0.5))
    #model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer = Adam(), loss = "categorical_crossentropy", metrics=["accuracy"])
# fit model with data augmentation
batch_Size = 1000
data_generator = ImageDataGenerator(
    rotation_range=15,
    zoom_range = 0.15,
    width_shift_range=0.15,
    height_shift_range=0.15)
data_generator.fit(train_X)
model.fit_generator(data_generator.flow(train_X, train_Y, batch_size = batch_Size),
                    epochs = 50, validation_data = (valid_X, valid_Y),
                    steps_per_epoch = train_X.shape[0] / batch_Size)
# predict results
results = model.predict(test_data)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submit.csv", index=False)
