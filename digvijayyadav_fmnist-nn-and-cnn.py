# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import tensorflow as tf

from tensorflow import keras

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test_df = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')

print('The shape of training dataset : ', train_df.shape)

print('The shape of testing dataset : ', test_df.shape)
train_df.head(5)
train = np.array(train_df, dtype = 'float32')

test = np.array(test_df, dtype = 'float32')
x_train = train[:,1:]/255



y_train = train[:,0]



x_test= test[:,1:]/255



y_test=test[:,0]
X_train, X_validate,y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.2, random_state = 5000)

print('The size of training data after model selection : ', X_train.shape, y_train.shape)

print('The size of Validation data after model selection : ', X_validate.shape, y_validate.shape)
class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(17, 17))



for i in range(25):

    plt.subplot(5, 5, i + 1)

    plt.grid(False)

    plt.imshow(x_train[i].reshape((28,28)))

    plt.colorbar()

    label_index = int(y_train[i])

    plt.title(class_names[label_index])

plt.show()
#Assigning image dimensions for model

img_rows = 28

img_cols = 28

img_shape = (img_rows, img_cols,1)



X_train = X_train.reshape(X_train.shape[0],*img_shape)

x_test = x_test.reshape(x_test.shape[0],*img_shape)

X_validate = X_validate.reshape(X_validate.shape[0],*img_shape)

model = tf.keras.Sequential([

        tf.keras.layers.Flatten(input_shape = img_shape),

        tf.keras.layers.Dense(512, activation = 'relu'),

        tf.keras.layers.Dense(10, activation = 'softmax') #since we want a probability based output

])

model.compile(optimizer = 'adam',

             loss = tf.keras.losses.SparseCategoricalCrossentropy(),

             metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 20, verbose=2, validation_data=(X_validate, y_validate))
plt.figure(figsize=(17,17))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Train - Accuracy without Dropout')
print('Training accuracy without dropout included : ', history.history['accuracy'][-1])

print('Validation accuracy without dropout included : ', history.history['val_accuracy'][-1])
model = tf.keras.Sequential([

        tf.keras.layers.Flatten(input_shape = img_shape),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(512, activation = 'relu'),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(10, activation = 'softmax') #since we want a probability based output

])

model.summary()
model.compile(optimizer = 'adam',

             loss = tf.keras.losses.SparseCategoricalCrossentropy(),

             metrics = ['accuracy'])
history = model.fit(X_train, y_train, epochs = 20, verbose=2, validation_data=(X_validate, y_validate))
plt.figure(figsize=(17,17))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Train - Accuracy using Dropout')
print('Training accuracy after dropout included : ', history.history['accuracy'][-1])

print('Validation accuracy after dropout included : ', history.history['val_accuracy'][-1])
test_loss, test_acc = model.evaluate(X_validate, y_validate)
predictions = model.predict(X_validate)
np.argmax(predictions[0])
print(predictions[11])
np.sum(predictions[11])
cnn_model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), padding='same', activation=tf.nn.relu,

                           input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.6),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])
cnn_model.summary()

cnn_model.compile(optimizer = 'adam',

             loss = tf.keras.losses.SparseCategoricalCrossentropy(),

             metrics = ['accuracy'])
history = cnn_model.fit(X_train, y_train, epochs = 15, verbose=2, validation_data=(X_validate, y_validate))
plt.figure(figsize=(17,17))



plt.subplot(2, 2, 1)

plt.plot(history.history['loss'], label='Loss')

plt.plot(history.history['val_loss'], label='Validation Loss')

plt.legend()

plt.title('Training - Loss Function')



plt.subplot(2, 2, 2)

plt.plot(history.history['accuracy'], label='Accuracy')

plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.legend()

plt.title('Train - Accuracy')
print('Training accuracy on CNN model : ', history.history['accuracy'][-1])

print('Validation accuracy on CNN model : ', history.history['val_accuracy'][-1])
test_loss, test_acc = cnn_model.evaluate(X_validate, y_validate, batch_size=32)
predictions = cnn_model.predict(X_validate)

print('Display the indices of the maximum values along an axis : {}'.format(np.argmax(predictions[0])))

print('Displaying the predictions : \n{}'.format(predictions[11]))

print('Total sum of these indices : {}'.format(np.sum(predictions[11])))