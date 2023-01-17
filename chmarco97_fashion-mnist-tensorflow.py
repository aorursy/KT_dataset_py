# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



train_data = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv", sep=",")

train_data.head()
# pop out target column

y_train = train_data.pop("label").values

print(y_train.shape)

# x train data

x_train = train_data.values

print(x_train.shape)
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from keras.models import Sequential

from keras.layers import Dense, Flatten, Activation, Dropout



model = Sequential()

model.add(Flatten(input_shape=(28, 28)))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax'))



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.summary()
x_train = [i.reshape(28,28) for i in x_train]



with tf.device("/GPU:0"):

    model_history = model.fit([x_train], y_train, epochs=20, shuffle=True, batch_size=500,validation_split=0.2)
import matplotlib.pyplot as plt

%matplotlib inline





train_acccuracy = model_history.history["accuracy"]

train_loss = model_history.history["loss"]

val_loss = model_history.history["val_loss"]



f = plt.figure()

f.set_size_inches(5, 15, forward=True)



plt.subplot(3,1,1)

plt.title("Training accuracy")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.plot(range(len(train_acccuracy)), train_acccuracy, "red", label="Train accuracy")

plt.legend()

plt.subplot(3,1,2)

plt.title("Training Loss")

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.plot(range(len(train_loss)), train_loss, "blue", label="Train loss")

plt.legend()

plt.subplot(3,1,3)

plt.title("Training Val_Loss")

plt.xlabel("Epochs")

plt.ylabel("Val_loss")

plt.plot(range(len(val_loss)), val_loss, "green", label="Val_loss")

plt.legend()
test_data = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv", sep=",")
# pop out target column

y_test = test_data.pop("label").values

print(y_test.shape)



# x train data

x_test = test_data.values

print(x_test.shape)
x_test = [i.reshape(28,28) for i in x_test]



test_loss, test_acc = model.evaluate([x_test],  y_test, verbose=1)



print('Test accuracy:', test_acc, "\nTest loss:", test_loss)



predictions = model.predict([x_test], batch_size=len(x_test))
classes = {0: "T-shirt/top", 1: "Trouser", 2:"Pullover", 3:"Dress", 

           4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 

           8:"Bag", 9:"Ankle boot"}



f = plt.figure()

f.set_size_inches(18.5, 35, forward=True)

print()

for i in range(20):

    plt.subplot(10, 5, i+1)

    plt.imshow(x_test[i].reshape(28,28), interpolation='nearest')

    plt.title("Prediction: "+classes[np.unravel_index(np.argmax(predictions[i], axis=None), predictions[i].shape)[0]])



plt.show()