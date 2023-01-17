# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
import tensorflow as tf

tr=train.drop(["label"], axis =1)

X_orig = tr.to_numpy()

Y_orig = train.label.to_numpy().reshape(42000,1)

X_test = test.to_numpy() /255

X_train = X_orig.reshape(42000,28,28,1) /255

Y_train = tf.keras.utils.to_categorical(Y_orig, num_classes=10, dtype="float32")
print("Training ", X_orig.shape)

print("Lavels ", Y_orig.shape) 

print("Training 2 ", X_train.shape)

print("Lavels 2 ", Y_train.shape) 
X_train[75]
import tensorflow as tf
input_shape=(28,28,1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation="relu"))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation="relu"))

model.add(tf.keras.layers.Dense(256, activation="relu"))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
hist=model.fit(X_train, Y_train, batch_size=512, epochs=10, validation_split=0.2, verbose=1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

Pred = model.predict(X_test)

Pred.shape
y_pred = Pred.argmax(axis=1)

ImageID = np.arange(len(y_pred))+1

Out = pd.DataFrame([ImageID,y_pred]).T

Out.rename(columns = {0:'ImageId', 1:'Label'})

#Out

Out.to_csv('MNIST_Predictionv2.csv', header =  ['ImageId', 'Label' ], index = None)
hist.history
import matplotlib.pyplot as plt

val = hist.history["val_loss"]



tra = hist.history["loss"]

i = range(1, 1+len(tra))

plt.plot(i, tra, "bo", label="train")

plt.plot(i, val, "b", label ="val")

plt.legend()