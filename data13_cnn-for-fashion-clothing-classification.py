import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import BatchNormalization
%matplotlib inline
# load training and test data into dataframes

train_set = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

test_set = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# show the first few records of the train_set datafreame

train_set.head()
# show the first few records of the test_set datafreame

test_set.head()
# check out the dataset format

train_set.shape
test_set.shape
# convert integers to floats

train_arr = np.array(train_set, dtype = 'float')

test_arr = np.array(test_set, dtype = 'float')
# show some images from the train dataset

plt.figure(figsize = (10, 10))

for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_arr[i,1:].reshape((28,28)), cmap = plt.cm.binary)

    train_labels = train_set['label'][i]

    plt.xlabel(classes[train_labels])

plt.show()
# normalise to range (0:1)

# exclude label column

X_train = train_arr[:, 1:] / 255

# include label column

y_train = train_arr[:, 0]
# exclude label column

X_test = test_arr[:, 1:] / 255

# include label column

y_test = test_arr[:,0]
# X_train represents the images, y_train represents the labels

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2)
# check for sizes

print (X_test.shape)

print (X_train.shape)

print (X_validate.shape)
# reshape dataset to have 28×28 pixels size

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))

X_validate = X_validate.reshape((X_validate.shape[0], 28, 28, 1))
# check again

print (X_test.shape)

print (X_train.shape)

print (X_validate.shape)
model = Sequential()
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
# first CONV => RELU => CONV => RELU => POOL layer set

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = input_shape))

model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3), padding = 'same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



# second CONV => RELU => CONV => RELU => POOL layer set

model.add(Conv2D(64, (3, 3), padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding = 'same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



# first (and only) set of FC => RELU layers

model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))



# softmax classifier

model.add(Dense(10))

model.add(Activation('softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = 1024, epochs = 50, verbose = 1, validation_data = (X_validate, y_validate))
# evaluate the model

evaluate = model.evaluate(X_test, y_test, verbose = 0)

print('Test Loss: {}'.format(round(evaluate[0], 3)))

print('Test Accuracy: {}'.format(round(evaluate[1], 3)))
plt.figure(figsize = (12, 10))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='val_Loss')

plt.legend()

plt.title('Loss evolution')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label='val_accuracy')

plt.legend()

plt.title('Accuracy evolution')
plt.figure(figsize = (12, 10))



accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(accuracy))



plt.subplot(2, 2, 1)

plt.plot(epochs, accuracy, '-g', label = 'Training Accuracy')

plt.plot(epochs, val_accuracy, 'b', label = 'Validation Accuracy')

plt.title('Training vs Validation Accuracy')

plt.legend()





plt.subplot(2, 2, 2)

plt.plot(epochs, loss, '-g', label = 'Training Loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title('Training vs validation Loss')

plt.legend()
total_classes = 10

predicted_classes = model.predict_classes(X_test)

target_classes = ['Class {}'.format(i) for i in range(total_classes)]

print(classification_report(y_test, predicted_classes, target_names = target_classes))
fig, axes = plt.subplots(5, 5, figsize = (15, 15))

axes = axes.ravel()



for i in np.arange(0, 5*5):  

    axes[i].axis('off')

    axes[i].imshow(X_test[i].reshape(28,28), cmap = plt.cm.binary)

    axes[i].set_title('Predicted Class = {}\n Actual Class = {}'.format((classes[predicted_classes[i]]), classes[int(y_test[i])]))

#     axes[i].set_title('Predicted Class = {}\n Actual Class = {}'.format(round(predicted_classes[i], 2), round(y_test[i], 2)))



# cmap=plt.cm.binary

plt.subplots_adjust(wspace = 0.5)