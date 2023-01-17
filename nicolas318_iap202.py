import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob
name_train = sorted(glob("/kaggle/input/enseeiht/cerfacs/TRAIN/*"))

name_test = sorted(glob("/kaggle/input/enseeiht/cerfacs/TEST/*"))



y_train = np.load("/kaggle/input/enseeiht/cerfacs/y_train.npy")



print (len(name_train), len(name_test))
import matplotlib.pyplot as plt

from PIL import Image



num = np.random.randint(len(name_train))

plt.figure(figsize=(6, 6))

plt.title("Image {} : {}".format(num, y_train[num]))

plt.imshow(Image.open(name_train[num]));
figure = plt.figure(figsize=(12, 12))

size = 5

grid = plt.GridSpec(size, size, hspace=0.05, wspace=0.0)



for line in range(size):

    for col in range(size):

        figure.add_subplot(grid[line, col])

        num = np.random.randint(len(name_train))

        plt.imshow(Image.open(name_train[num]))

        plt.axis('off')  
X_train = np.array([np.array(Image.open(jpg)) for jpg in name_train])

X_test = np.array([np.array(Image.open(jpg)) for jpg in name_test])

y_train = np.load("/kaggle/input/enseeiht/cerfacs/y_train.npy")



print (X_train.shape, X_test.shape)

print (y_train.shape)
print (y_train[0])
from sklearn.preprocessing import OneHotEncoder



print (f"Shape label raw : {y_train.shape}")



encoder = OneHotEncoder()

y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()



print (f"Shape label One Hot Encoded : {y_train.shape}")

print (f"Label for y_train[0] : {y_train[0]}")
X_train, X_valid = X_train[:15000], X_train[15000:]

y_train, y_valid = y_train[:15000], y_train[15000:]
X_train, X_valid, X_test = X_train/255, X_valid/255, X_test/255
import keras 

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D



model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(64, 64, 3)))

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation = 'softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.SGD(),

              metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = 32, 

                   validation_data=(X_valid, y_valid), epochs=30)
loss, metrics = model.evaluate(X_valid, y_valid)



print (metrics)