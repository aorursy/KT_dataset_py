import warnings; warnings.simplefilter('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

import seaborn as sns

from sklearn import metrics
train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
print(train.shape)

print(test.shape)
train.head()
values = {0: "T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle Boot"}

def lab_to_desc(label):

    return values[label]
train_arr = np.array(train)

test_arr = np.array(test)

train_arr
fig = plt.figure()



# From training dataset

ax1 = fig.add_subplot(1,2,1)

ax1.imshow(train_arr[1, 1:].reshape(28, 28), cmap="Greys")

print("Image 1 label: ",train_arr[1, 0])



# From testing dataset

ax2 = fig.add_subplot(1,2,2)

ax2.imshow(test_arr[1, 1:].reshape(28, 28), cmap="Greys")

print("Image 2 label: ",test_arr[1, 0])
train_X = train_arr[:, 1:]/255

test_X = test_arr[:, 1:]/255

print(train_X.shape)

print(test_X.shape)
train_y = train_arr[:, 0]

test_y = test_arr[:, 0]
train_X = train_X.reshape([train_X.shape[0], 28, 28, 1])

print(train_X.shape)

test_X = test_X.reshape([test_X.shape[0], 28, 28, 1])

print(test_X.shape)
from keras.models import Sequential

from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, Dropout

from keras.optimizers import Adam
classifier1 = Sequential()

classifier1.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu'))

classifier1.add(MaxPooling2D(pool_size=(2, 2)))

classifier1.add(Flatten())

classifier1.add(Dense(units = 16, activation = 'relu'))

# 10 units in the last Dense layer as there are 10 classes to be classified into

classifier1.add(Dense(units = 10, activation = 'sigmoid'))
classifier1.summary()
classifier1.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
%%time

history = classifier1.fit(train_X, train_y, epochs=10, batch_size=256)
print(history.history.keys())



plt.plot(history.history['acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
result = classifier1.evaluate(x=test_X, y=test_y)

print("Accuracy of the model is: %.2f percent"%(result[1]*100))

predictions = classifier1.predict_classes(test_X)
import random

fig, axes = plt.subplots(5, 5, figsize=(14, 14))

# axes is currently in multiple lists, ravel reshapes it to 1D

axes = axes.ravel()



randnum = random.randint(0, 9975)



for i in range(25):

    axes[i].imshow(test_X[randnum + i].reshape(28, 28), cmap="Greys")

    axes[i].set_title('Prediction: %s\n True: %s' %

                      (lab_to_desc(predictions[randnum + i]), lab_to_desc(test_y[randnum + i])))

    axes[i].axis('off')



plt.subplots_adjust(wspace=0.5)