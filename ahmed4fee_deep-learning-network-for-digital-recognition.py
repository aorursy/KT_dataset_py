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


import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



test.head()
train.head()
train.shape
train.info()
#checks for null values

train.isnull().any().describe()

test.isnull().any().describe()
X_train=(train.iloc[:,1:].values.astype('float32'))

y_train=(train.iloc[:,0].values.astype('float32'))

X_test=test.values   #.astype('float32')
X_train.shape
y_train.shape
X_test.shape
#normalize the data

X_train=X_train/255.0

test=test/255.0
#Convert train datset to (num_images, img_rows, img_cols) format 

X_train = X_train.reshape(X_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])
X_train=X_train.reshape(X_train.shape[0],28,28,1)

X_test=X_test.reshape(X_test.shape[0],28,28,1)

X_train.shape
sns.countplot(y_train)
from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)

X_train,X_val,y_train,y_val=train_test_split(X_train ,y_train ,test_size=.2 , random_state=2)
plt.imshow(X_train[1][:,:,0])
from keras import layers

from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))



model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))





model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])



from keras.preprocessing import image

gen = image.ImageDataGenerator()
batches = gen.flow(X_train, y_train, batch_size=64)

val_batches=gen.flow(X_val, y_val, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, 

                    validation_data=val_batches, validation_steps=val_batches.n)
test_loss, test_acc = model.evaluate(X_val, y_val)
test_loss
test_acc
y_test=model.predict(X_test)

#

y_pred = np.argmax(y_test,axis=1)
results = pd.Series(y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("Sub.csv",index=False)
submission
"Sub.csv"