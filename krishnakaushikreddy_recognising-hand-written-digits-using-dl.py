# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install tensorflow keras matplotlib

import tensorflow as tf

from tensorflow import keras



train=pd.read_csv("../input/digit-recognizer/train.csv")

test=pd.read_csv("../input/digit-recognizer/test.csv")

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import to_categorical

import matplotlib.pyplot as plt


train_images=[]

for i in range(0,len(train)):

    x=np.array(train.iloc[i][1:].values)

    x=x.reshape(28,28)

    train_images.append(x)

train_images = np.asarray(train_images) 





test_images = []

for i in range(0,len(test)):

    x = np.array(test.iloc[i].values)

    x = x.reshape(28,28)

    test_images.append(x)

test_images = np.asarray(test_images)
plt.imshow(train_images[0],cmap='gray')

plt.show()
train_labels = train['label']
train_images= (train_images/255.0)

test_images=(test_images/255.0)


model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

     keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])


model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=["accuracy"]

    

)
train_images.shape, train_labels.shape
model.fit(train_images, train_labels, epochs=10)
y_pred = model.predict_classes(test)

results = pd.Series(y_pred,name="Label")

submission123 = pd.concat([pd.Series(range(1,len(y_pred)+1),name = "ImageId"),results],axis = 1)
print(submission123.head(10))
submission123.to_csv("final1.csv",index=False)