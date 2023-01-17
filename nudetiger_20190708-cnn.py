# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

x_train = train.drop('label', axis=1)

y_train = train['label']

test = pd.read_csv('../input/test.csv')
x_train = x_train.values

y_train = y_train.values

test = test.values
import matplotlib.pyplot as plt



labels_text = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]



fig, ax = plt.subplots(1, 10, figsize=(20,20))



idxs = [np.where(y_train == i)[0] for i in range(10)]



for i in range(10):

    k = np.random.choice(idxs[i])

    ax[i].imshow(x_train[k].reshape(28, 28), cmap="gray")

    ax[i].set_title("{}".format(labels_text[i]))
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=1004)
from keras import models

from keras import layers



model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))

model.add(layers.Dense(10, activation='softmax'))



model.compile(optimizer='adam',

             loss='categorical_crossentropy',

             metrics=['acc'])
X_train = X_train.reshape((33600, 28*28))

X_train = X_train.astype('float32') / 255

X_val = X_val.reshape((8400, 28*28))

X_val = X_val.astype('float32') / 255



test = test.reshape((28000, 28* 28))

test = test.astype('float32') / 255
model.summary()
from keras.utils import to_categorical



y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data =(X_val, y_val))
predict = model.predict(test)
predict = np.argmax(predict,axis = 1)



predict = pd.Series(predict,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predict],axis = 1)



submission.to_csv("predict_20190706.csv",index=False)