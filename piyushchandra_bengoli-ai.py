# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt

import matplotlib as mlt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##import pyarrow as pa

##import pyarrow.parquet as pq 
import pandas as pd

class_map = pd.read_csv("../input/bengaliai-cv19/class_map.csv")

sample_submission = pd.read_csv("../input/bengaliai-cv19/sample_submission.csv")

test = pd.read_csv("../input/bengaliai-cv19/test.csv")

train = pd.read_csv("../input/bengaliai-cv19/train.csv")
y_train_grapheme_root=train["grapheme_root"]
y_train_vowel_diacritic=train["vowel_diacritic"]
y_train_consonant_diacritic=train["consonant_diacritic"]
del class_map

del sample_submission

del test

del train
train0=pd.read_parquet("../input/bengaliai-cv19/train_image_data_0.parquet")

train0=train0.drop(["image_id"],axis=1)
train0.shape
y_train_consonant_diacritic0=y_train_consonant_diacritic[:50210]
y_train_consonant_diacritic0.shape
train0=train0.values.reshape(-1,236,137,1)

g=plt.imshow(train0[1000][:,:,0])
from tensorflow import keras

from tensorflow.keras import models,layers

model=models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(137,59 , 4)))

model.add(layers.MaxPooling2D((2, 2)))

##model.add(layers.Conv2D(64, (3, 3), activation='relu'))

##model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dropout(0.1))

##model.add(layers.Dense(128,activation='relu'))

##model.add(layers.Dropout(0.1))

model.add(layers.Dense(7, activation='softmax'))
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
#y_train_consonant_diacritic0=y_train_consonant_diacritic0.to_numpy()
model.fit(train0, y_train_consonant_diacritic0, epochs=5)
loss,acc=model.evaluate(train0, y_train_consonant_diacritic0)
del train0

del y_train_consonant_diacritic0
train1=pd.read_parquet("../input/bengaliai-cv19/train_image_data_1.parquet")

train1=train1.drop(["image_id"],axis=1)
train1.shape
y_train_consonant_diacritic1=y_train_consonant_diacritic[50210:100420]
y_train_consonant_diacritic1.shape
train1=train1.values.reshape(-1,236,137,1)

g=plt.imshow(train1[1000][:,:,0])
#y_train_consonant_diacritic1=y_train_consonant_diacritic1.to_numpy()
model.fit(train1, y_train_consonant_diacritic1, epochs=5)
loss1,acc1=model.evaluate(train1, y_train_consonant_diacritic1)
del train1

del y_train_consonant_diacritic1
train2=pd.read_parquet("../input/bengaliai-cv19/train_image_data_2.parquet")

train2=train2.drop(["image_id"],axis=1)
y_train_consonant_diacritic2=y_train_consonant_diacritic[100420:150630]
train2=train2.values.reshape(-1,236,137,1)

g=plt.imshow(train2[1000][:,:,0])
#y_train_consonant_diacritic2=y_train_consonant_diacritic2.to_numpy()
model.fit(train2, y_train_consonant_diacritic2, epochs=5)
loss2,acc2=model.evaluate(train2, y_train_consonant_diacritic2)
del train2

del y_train_consonant_diacritic2

train3=pd.read_parquet("../input/bengaliai-cv19/train_image_data_3.parquet")

train3=train3.drop(["image_id"],axis=1)

train3.shape
y_train_consonant_diacritic3=y_train_consonant_diacritic[150630:200840]
train3=train3.values.reshape(-1,236,137,1)

g=plt.imshow(train3[1000][:,:,0])
#y_train_consonant_diacritic3=y_train_consonant_diacritic3.to_numpy()
model.fit(train3, y_train_consonant_diacritic3, epochs=5)
loss3,acc3=model.evaluate(train3, y_train_consonant_diacritic3)
del train3

del y_train_consonant_diacritic3
test0 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_0.parquet")

test0=test0.drop(["image_id"],axis=1)
test1 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_1.parquet")

test1=test1.drop(["image_id"],axis=1)
test2 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_2.parquet")

test2=test2.drop(["image_id"],axis=1)
test3 = pd.read_parquet("../input/bengaliai-cv19/test_image_data_3.parquet")

test3=test3.drop(["image_id"],axis=1)
x_test=pd.concat([test0,test1,test2,test3],ignore_index=True)
del test0

del test1

del test2

del test3
x_test=x_test.values.reshape(-1,236,137,1)

g=plt.imshow(x_test[0][:,:,0])
pred_consonant_diacritic=model.predict(x_test)

pred_consonant_diacritic
pred_consonant_diacritic=np.argmax(pred_consonant_diacritic,axis=1)

dict={'consonant_diacritic':pred_consonant_diacritic}

submission=pd.DataFrame(dict)

submission.to_csv("pred_consonant_diacritic.csv")