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
import tensorflow

print(tensorflow.__version__)

from tensorflow import keras
data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")
train_labels = data_train.label.values

data_train = data_train.drop("label",axis =1)
train_images = data_train.values

test_images = data_test.values

print(train_images.shape,'\n',test_images.shape,'\n',train_labels.shape)
plt.figure(figsize = (10,10))

for i in range(25):

    plt.subplot(5,5,1+i)

    plt.xticks([])

    plt.imshow(train_images[i].reshape(28,28))

    plt.xlabel(prediction[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape = (28*28,)),

    keras.layers.Dense(512,activation = 'relu'),

    keras.layers.Dense(10,activation = 'softmax')

])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics= ['accuracy'])
model.fit(train_images,train_labels,epochs = 10)
pred = model.predict(test_images)

prediction = np.argmax(pred,axis=1)

print(pred.shape,'\n',prediction.shape)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(test_images[i].reshape(28,28))

    plt.xlabel("pred:{}  ".format(prediction[i]))

    plt.xticks([])
!cd ../input

!ls -a
data = {"ImageId" : np.arange(1,prediction.shape[0]+1), "Label" : prediction}

submission = pd.DataFrame(data)