import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(train.shape)

print(test.shape)
train.head()
test.head()
X_train = train.drop(labels=['label'], axis=1)

Y_train = train["label"]
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255.0

test = test/255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
import tensorflow as tf

Y_train = tf.keras.utils.to_categorical(Y_train, num_classes = 10)

Y_train
plt.plot(Y_train[41999])

plt.xticks(range(10));
from sklearn.model_selection import train_test_split



X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=2)
steps_train = int(len(X_Train)/64)

steps_valid = int(len(X_Val)/64)
from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator



training_datagen = ImageDataGenerator(

    rescale=1/255,

#     rotation_range=40,

#     zoom_range=0.2,

#     horizontal_flip=True

)



validation_datagen = ImageDataGenerator(rescale=1/255)



train_generator = training_datagen.flow(X_Train, Y_Train, batch_size=64)

valid_generator = validation_datagen.flow(X_Val, Y_Val, batch_size=64)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28,28,1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation = "relu"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation = "softmax")

])



model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(loss= 'categorical_crossentropy', optimizer=RMSprop(lr = 0.001), metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs = 10)