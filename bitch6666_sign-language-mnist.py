# check directory of input

import os

print(os.listdir("../input/"))
# check directory of sign-language-mnist

print(os.listdir("../input/sign-language-mnist"))
from IPython.display import Image

Image("../input/sign-language-mnist/amer_sign2.png")
Image("../input/sign-language-mnist/american_sign_language.PNG")
# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

import tensorflow.keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout



from tensorflow.keras.utils import to_categorical
# Read Dataset

train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')

test = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')
# show data format

train.head()
# get labels 

labels = train['label']
# droping the label coloumn from the training set

train.drop('label', axis = 1, inplace = True)

print(train.shape)
# Reshape

x_train = train.values.reshape(train.shape[0],28,28,1)
# Show a reshape image

plt.imshow(x_train[0].reshape(28,28))
# explore the label distribution of training images

plt.figure(figsize = (18,8))

sns.countplot(x =labels)
# one-hot encoding

labels = to_categorical(labels)

print(labels.shape)
num_classes = 25
# Split Dataset into Train and Test

x_train, x_test, y_train, y_test = train_test_split(x_train, labels, test_size = 0.2, random_state = 2)
# Normalize Dataset

x_train = x_train / 255.0

x_test = x_test / 255.0
print(x_train.shape)

print(x_test.shape)
# Build Model

model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation = 'relu', input_shape=(28, 28 ,1) ))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Conv2D(64, kernel_size = (3, 3), padding='same', activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.20))

model.add(Dense(num_classes, activation = 'softmax'))



model.summary()
# Compile Model

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])
# Train Model

batch_size = 128

epochs = 20



history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
def show_history(t1, t2):

    plt.plot(history.history[t1])

    plt.plot(history.history[t2])

    plt.title("History of "+t1)

    plt.xlabel('epoch')

    plt.ylabel('value')

    plt.legend([t1,t2])

    plt.show()
# Show Training History

show_history('accuracy', 'val_accuracy')

show_history('loss', 'val_loss')
# Test Accuracy

score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss: ', score[0])

print('Test accuracy: ', score[1])