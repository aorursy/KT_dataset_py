import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow import keras



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv", sep=",")

test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv", sep=",")

train.head()
print("train data shape is: ",train.shape)

print("test data shape is: ", test.shape)
test.head()
org_X = train.drop(['label'], axis = 1).values

Y = train['label'].values

X_test = test.drop(['label'], axis = 1).values

Y_test = test['label'].values
# reshape to 28x28 matrix

X = org_X.reshape(org_X.shape[0],28,28,1)

X_test = X_test.reshape(X_test.shape[0],28,28,1)



#Y = tf.keras.utils.to_categorical(Y)



print("new X shape is: ", X.shape)

print("new Y shape is: ", Y.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X[i][:,:,0], cmap=plt.cm.binary)

    plt.xlabel(class_names[train.label[i]])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28,1)),

    keras.layers.BatchNormalization(momentum=0.99),

    keras.layers.Dropout(0.25),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.BatchNormalization(momentum=0.99),

    keras.layers.Dense(10, activation='softmax')

])

model.summary()
# compile the model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
from sklearn.model_selection import train_test_split

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size = 0.2)
# train model

model.fit(X, Y, epochs=30)
# evaluate model performance

test_loss, test_acc = model.evaluate(X_test, Y_test,verbose=2)



print('\nTest accuracy: ', test_acc)

print('\nTest loss: ', test_loss)
# reshape to 28x28 matrix

#X = np.reshape(org_X,(org_X.shape[0],28,28,1))

#X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size = 0.8)



# model2 with Convolution layers

model2 = keras.Sequential([

    keras.layers.Conv2D(32,kernel_size = (3,3),activation='relu',kernel_initializer='he_uniform', padding = 'same',input_shape=(28,28,1)),

    keras.layers.Conv2D(32, (3,3), activation = 'relu',padding='same'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.Dropout(0.4),

    

    keras.layers.Conv2D(64,kernel_size = (3,3),activation='relu',padding = 'same'),

    keras.layers.Conv2D(64, (3,3), activation = 'relu',padding='same'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPooling2D(2,2),

    keras.layers.Dropout(0.2),

    

    #keras.layers.Conv2D(128,(3,3),padding='same'),

    #keras.layers.Conv2D(128,(3,3),activation='relu'),

    #keras.layers.BatchNormalization(momentum=0.99),

    #keras.layers.MaxPooling2D(2,2),

    

    keras.layers.Flatten(),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(128, activation ='relu'),

    keras.layers.BatchNormalization(momentum=0.99),

    keras.layers.Dense(10, activation='softmax')

])

model2.summary()
# compile the model

model2.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



# add early stopping

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)



# train model

model2.fit(X, Y, epochs=30,

          validation_data=[X_dev,Y_dev],

          #callbacks = [callback]

          )
# evaluate model performance

test_loss, test_acc = model2.evaluate(X_test, Y_test,verbose=2)



print('\nTest accuracy: ', test_acc)

print('\nTest loss: ', test_loss)