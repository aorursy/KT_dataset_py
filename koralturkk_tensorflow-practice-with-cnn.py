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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print("Number of missing values in training set: {}".format(train_data.isnull().any().sum()))

print("Number of missing values in test set: {}".format(test_data.isnull().any().sum()))
print(train_data.shape)

print(test_data.shape)
Y_train = train_data["label"]

train_data = train_data.drop(columns=["label"])

print(train_data.head())
train_data = train_data.to_numpy().reshape(train_data.shape[0],28,28,1)

test_data = test_data.to_numpy().reshape(test_data.shape[0],28,28,1)



Y_train = Y_train.to_numpy().T

X_train = train_data

X_test = test_data





print("\n---Y_train---")

print(Y_train.shape)

print("\n---X_train---")

print(train_data.shape)

print("\n---X_test---")

print(test_data.shape)
X_train = X_train / 255

X_test = X_test / 255
np.random.seed(4)



from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.1, random_state=4, shuffle = True)



import matplotlib.pyplot as plt



plt.figure(1)

plt.subplot(221)

image = plt.imshow(X_train[3][:,:,0])

plt.subplot(222)

image = plt.imshow(X_train[10][:,:,0])

plt.subplot(223)

image = plt.imshow(X_train[1001][:,:,0])

plt.subplot(224)

image = plt.imshow(X_train[3671][:,:,0])



plt.show()
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch, logs={}):

        if logs.get("acc" > .994):

            print("Accuracy is high enough now to stop the training")

            self.model.stop_training = True



callback = myCallback()
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dense(256, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

              #callbacks= [callback])

model.summary()

model.fit(X_train, Y_train, epochs=5)
f, axarr = plt.subplots(3,4)

FIRST_IMAGE=0

SECOND_IMAGE=7

THIRD_IMAGE=26

CONVOLUTION_NUMBER = 10

layer_outputs = [layer.output for layer in model.layers]



activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)



for x in range(0,4):

  f1 = activation_model.predict(X_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')

  axarr[0,x].grid(False)

  f2 = activation_model.predict(X_test[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')

  axarr[1,x].grid(False)

  f3 = activation_model.predict(X_test[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')

  axarr[2,x].grid(False)
test_loss, test_acc = model.evaluate(X_val, Y_val)

print(test_acc)
results = model.predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("tensor_flow_cnn.csv",index=False)



print(submission)