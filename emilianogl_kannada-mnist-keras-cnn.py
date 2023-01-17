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
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import random

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

Dig_MNIST = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
print(f"Train shape: {train.shape}")

print(f"Test shape: {test.shape}")

print(f"Dif shape: {Dig_MNIST.shape}")
train.head()
X = train.iloc[:,1:].values

Y = train.iloc[:,0].values



Y[:10]
X = X.reshape(X.shape[0], 28, 28 ,1)

print(f"X data shape: {X.shape}")
Y = tf.keras.utils.to_categorical(Y, num_classes=10)

print(f"Y data shape: {Y.shape}")
test.head()
x_test=test.drop('id', axis=1).iloc[:,:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28,1)



print(f"x_test shape: {x_test.shape}")
Dig_MNIST.head()
x_dig=Dig_MNIST.drop('label', axis=1).iloc[:,:].values

x_dig = x_dig.reshape(x_dig.shape[0], 28, 28,1)



print(f"x_dig shape: {x_dig.shape}")
y_dig=Dig_MNIST.label



print(f"y_dig shape: {y_dig.shape}")
X = X / 255

x_test = x_test / 255

x_dig = x_dig / 255
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X, Y, test_size = 0.20)
for i in range(3):

    image = Xtrain[random.randint(0,1024)].reshape(28,28)

    plt.imshow(image, cmap="gray")

    plt.axis("off")

    plt.show()
inputs = Input(shape=(28, 28, 1))

out = Conv2D(64, kernel_size=(3, 3), activation="relu")(inputs)

out = Conv2D(128, kernel_size=(3, 3), activation="relu")(out)

out = MaxPooling2D(pool_size=(2, 2))(out)

out = Flatten()(out)

out = Dense(60, activation="relu")(out)

out = Dropout(0.25)(out)

out = Dense(20, activation="relu")(out)

out = Dropout(0.25)(out)

out = Dense(10, activation="softmax")(out)



model = Model(inputs=inputs, outputs=out)

model.summary()
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=["categorical_accuracy"])
model.fit(Xtrain, Ytrain, batch_size=1024, 

          epochs=25, validation_data=[Xvalid, Yvalid], verbose=1)
score = model.evaluate(Xvalid, Yvalid, batch_size=1024, verbose=0)



print(f"\nLoss: {score[0]}")

print(f"Accuracy: {score[1]}")
predictions = model.predict(x_test).argmax(axis=-1)



predictions[:10]
submission['label'] = predictions



submission.head()
submission.to_csv("submission.csv",index=False)