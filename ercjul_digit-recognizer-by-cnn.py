# 安装tf2.0环境

!pip install tensorflow-gpu==2.0.0b1
import os

print(os.listdir("../input"))



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Any results you write to the current directory are saved as output.

print(tf.__version__)
# 读取数据

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
Y_train = train["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



_ = sns.countplot(Y_train)
# 数据归一化

X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28, 1)

test = test.values.reshape(-1,28,28, 1)
# 数据展示

plt.figure()

plt.imshow(X_train[0][:,:,0], cmap=plt.cm.binary)

plt.colorbar()

plt.show()
# 数据增强

datagen = keras.preprocessing.image.ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
# 模型

model = keras.Sequential([

    keras.layers.Conv2D(32, (3,3), activation ='relu', input_shape=(28, 28, 1)),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(32, (3,3), activation ='relu'),

    keras.layers.MaxPool2D((2,2)),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, (3,3), activation ='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(64, (3,3), activation ='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.MaxPool2D((2,2)),

    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),

    keras.layers.Dense(1024, activation='relu'),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(10, activation='softmax')

])
model.summary()
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=1000), steps_per_epoch=300, epochs=50-5)
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=1000), steps_per_epoch=300, epochs=5)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='r', label="Training accuracy")

legend = ax[1].legend(loc='best', shadow=True)
predictions = model.predict(test)
results = np.argmax(predictions, axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)