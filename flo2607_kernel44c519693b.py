import tensorflow as tf

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

%matplotlib inline
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(f"Train size : {train.shape}")

print(f"Test size : {test.shape}")
test
train_images = train.drop(['label'], axis=1)

train_labels = train['label']

sns.countplot(train_labels)
train_images = train_images / 255.0

test = test / 255.0

train_images = train_images.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

random_seed = 5

train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size = 0.1, random_state=random_seed)

sns.countplot(train_labels)

sns.countplot(validation_labels)
# show some numbers

plt.figure(figsize=(15,15))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i][:,:,0], cmap=plt.cm.binary)

    plt.xlabel(train_labels.values[i])

plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)),

    tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',  activation ='relu', input_shape = (28,28,1)),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),

    tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
epochs = 100

batch_size = 10000



history = model.fit(train_images, train_labels.values, batch_size = batch_size, epochs = epochs, validation_data = (validation_images, validation_labels.values), verbose = 1)
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
#just to show a fancy result

plt.figure(figsize=(10,15))

for i in range(390):#

    plt.subplot(20,39,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test[i][:,:,0], cmap=plt.cm.binary)

    plt.xlabel(results.values[i])

plt.show()
results=results.to_frame()

results.index += 1

results["ImageId"]=results.index

results.to_csv("results.csv",encoding='utf-8',index=False)
