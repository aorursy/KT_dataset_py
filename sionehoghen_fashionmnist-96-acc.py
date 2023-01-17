import numpy as np

import pandas as pd

import os

import tensorflow as tf

import matplotlib.pyplot as mat

%matplotlib inline

import random

import time
# Read the csv train and test files.

training = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
print(f'training_set_shape: {training.shape}')

print(f'test_set_shape: {test.shape}')
# Splitting the label from pixels in training set

label = training.pop('label')
sample= random.randint(0, training.shape[0])

sample_image = training.iloc[sample, :]

sample_image = np.array(sample_image)

sample_image = sample_image.reshape(28, 28)

mat.imshow(sample_image, cmap = 'gray')
items = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

r = 3

mat.figure(figsize=(10,10))

for i in range(9):

    mat.subplot(r, r, i+1)

    ran = random.randint(0, training.shape[0])

    img = training.loc[ran]

    img = np.array(img).reshape(28,28)

    mat.imshow(img)

    ran = label[ran]

    mat.xlabel(items[ran])

mat.show()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (28,28,1)),

    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),

    

    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),

    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),

    

    tf.keras.layers.Conv2D(256, (3,3), activation = 'relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D((2,2)),

    

    tf.keras.layers.Flatten(),

    

    tf.keras.layers.Dense(128, activation = 'relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(64, activation = 'relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(10, activation = 'softmax'),

])
model.summary()
# Compile the model

model.compile(loss='sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
train = np.array(training)

train_final = train.reshape(train.shape[0], 28,28,1)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_final,label, test_size = 0.2, random_state = 42)
x_train = x_train/255

x_val = x_val/255

y_train = np.array(y_train).reshape(48000, 1)
check = tf.keras.callbacks.ModelCheckpoint('./')

hist = model.fit(x_train, y_train, epochs= 20,batch_size = 32, validation_data=(x_val, y_val), callbacks = [check] )
loss = hist.history['loss']

accuracy = hist.history['accuracy']

val_loss = hist.history['val_loss']

val_accuracy = hist.history['val_accuracy']

epochs = hist.epoch

mat.figure(figsize=(10,10))

mat.subplot(2,1, 1)

mat.plot(epochs, loss, label= 'Loss')

mat.plot(epochs, accuracy, label = 'Accuracy')

mat.xlabel('Epochs')

mat.legend()

mat.title('LOSS, ACC vs EPOCHS')



mat.subplot(2,1, 2)

mat.plot(epochs, val_loss, label = 'Val_loss')

mat.plot(epochs, val_accuracy, label = 'Val_acc')

mat.xlabel('Epochs')

mat.legend()

mat.title('VAL_LOSS, VAL_ACC vs EPOCHS')
y_test = test.pop('label')

y_test = np.array(y_test).reshape(y_test.shape[0], 1)
test = np.array(test).reshape(test.shape[0],28,28,1)

test = test/255.

model.evaluate(test, y_test)
items = ['T-shirt/Top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

mat.figure(figsize = (15,10))

for i in range(15):

    mat.subplot(3, 5, i+1)

    sample_int = random.randint(0, test.shape[0])

    sample_actual = y_test[sample_int]

    sample = test[sample_int]

    sample_actual = int(sample_actual[0])

    name = items[sample_actual]

    pred = model.predict(sample.reshape(-1,28,28,1))

    act = np.argmax(pred)

    if act == sample_actual:

        color= 'green'

    else:

        color = 'red'

    mat.imshow(sample.reshape(28,28))

    mat.xlabel(name, color= color)

mat.show()