# Import packages and data

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Conv2D

from keras.layers import Flatten

from keras.layers import MaxPooling2D

from sklearn.model_selection import train_test_split



train = pd.read_csv("../input/train.csv")

true_test = pd.read_csv("../input/test.csv")
np.shape(train)
print(train.head())
x_train, x_test, y_train, y_test = train_test_split(train.iloc[:,1:], train.iloc[:,0], test_size=0.3, random_state=10)
# Supposed to used .to_numpy() but doesn't work, pandas isn't version 0.24 here. Use .values instead.

sampleimage = x_train.iloc[50].values

sampleimage = sampleimage.reshape(28,28)



# Show an example of an image, take index 50

plt.imshow(sampleimage, cmap='Greys')

print("The image should be a '{}'".format(y_train.iloc[50]))
sns.set(style="darkgrid")

fig, axe = plt.subplots(1, figsize=(10, 5))

p = sns.countplot(x='label',data = train, ax=axe).set_title("Counts of different digits")
x_train = x_train.values.astype("float")

x_test = x_test.values.astype("float")

x_train = np.reshape(x_train,(x_train.shape[0], 28, 28, 1))

x_test = np.reshape(x_test,(x_test.shape[0], 28, 28, 1))
x_train /= 255

x_test /= 255
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1)))  

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25)) 

model.add(Flatten())  



model.add(Dense(128, activation=tf.nn.relu))

model.add(Dropout(0.25)) 

model.add(Dense(10, activation=tf.nn.softmax))
model.summary()
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)
results = model.evaluate(x_test, y_test)

print("Accuracy is", results[1])