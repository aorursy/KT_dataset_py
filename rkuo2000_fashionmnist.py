import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow  as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt
labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
# Import Dataset

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

print(trainX.shape)

print(testX.shape)
# Show image of training data

plt.figure(figsize = (5, 5)) # set size of figure 10x10

rand_indexes = np.random.randint(0, trainX.shape[0], 8) # select 8 digits(0~9) randomly 

print(rand_indexes)

for index,im_index in enumerate(rand_indexes):

    plt.subplot(4, 4, index+1)

    plt.imshow(trainX[im_index], cmap = 'gray', interpolation = 'none')

    plt.title(labelNames[trainY[im_index]])

plt.tight_layout()
# reshape for model's input_shape

X_train = trainX.reshape((trainX.shape[0], 28,28,1))

X_test  = testX.reshape((testX.shape[0], 28,28,1))
# normalization

X_train = X_train.astype("float32") /255.0

X_test  = X_test.astype("float32")  /255.0



# one-hot encoding

Y_train = to_categorical(trainY, 10)

Y_test  = to_categorical(testY, 10)
num_classes = 10

input_shape = (28, 28, 1)
# Build Model

model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()
batch_size = 32

num_epochs = 25
# Train Model

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs)
# Evaluate Model

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
# Model Predict

preds = model.predict(X_test)
# Show image of test data

plt.figure(figsize = (5, 5)) # set size of figure 10x10

rand_indexes = np.random.randint(0, testX.shape[0], 8) # select 8 digits(0~9) randomly 

print(rand_indexes)

for index,im_index in enumerate(rand_indexes):

    plt.subplot(4, 4, index+1)

    plt.imshow(testX[im_index], cmap = 'gray', interpolation = 'none')

    maxindex = int(np.argmax(preds[im_index])) # maxindex = index of max. probility 

    if maxindex==testY[im_index]: 

        plt.title(labelNames[maxindex], color='green') # title in green if match

    else:

        plt.title(labelNames[maxindex], color='red')   # title in red if mismatch 

plt.tight_layout()