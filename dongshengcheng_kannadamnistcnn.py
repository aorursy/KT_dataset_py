# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

dig = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
train_copy = train.copy()

test_copy = test.copy()

dig_copy = dig.copy()
train_label = train["label"].copy()

train_img = train_copy.drop("label",axis=1)
train_img.head()
images = train_img/255.0
images = np.array(images)
labels = np.array(train_label)
onehot = OneHotEncoder()
labels = onehot.fit_transform(labels.reshape([-1,1]))
labels = labels.A
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
x_train = tf.reshape(x_train,[-1,28,28,1])

x_test = tf.reshape(x_test, [-1,28,28,1])
x_train.shape
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([

                                    tf.keras.layers.Conv2D(32,(5,5),padding="same",activation="relu"),

                                    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"),

                                    tf.keras.layers.Conv2D(32,(5,5),padding="same",activation="relu"),

                                    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(1024,activation="relu"),

                                    tf.keras.layers.Dropout(0.5),

                                    tf.keras.layers.Dense(10,activation="softmax")

])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-5 * 10**(epoch / 100))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer,metrics=["accuracy"])

history = model.fit(x_train,y_train,epochs=300,callbacks=[lr_schedule])
plt.figure(figsize=(16,12))

plt.semilogx(history.history["lr"],history.history["loss"])

plt.grid(True)

plt.axis([1e-5,1e-2,0,0.025])
tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([

                                    tf.keras.layers.Conv2D(32,(5,5),padding="same",activation="relu"),

                                    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"),

                                    tf.keras.layers.Conv2D(32,(5,5),padding="same",activation="relu"),

                                    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same"),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(1024,activation="relu"),

                                    tf.keras.layers.Dropout(0.5),

                                    tf.keras.layers.Dense(10,activation="softmax")

])
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=optimizer,metrics=["accuracy"])

history = model.fit(x_train,y_train,epochs=300)
model.summary()
valid_lose = model.evaluate(x_test, y_test)
test_images= test_copy.drop("id",axis=1)
test_images = test_images/255.0
test_images = tf.reshape(test_images,[-1,28,28,1])
test_ret = model.predict(test_images)
predict = tf.argmax(test_ret,1)
predict = predict.numpy()
predict.shape
sunmission = pd.DataFrame({'id': test.id,'label': predict})
sunmission.to_csv("submission.csv",index=False)