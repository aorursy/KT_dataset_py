# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

file_names=[]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_names.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import cv2
train=pd.read_csv(file_names[1])

# file_names
train

labels=train[["label"]]
labels=labels.to_numpy()

labels
train=train.iloc[:,1:]

train.head()
labels
train=train.to_numpy()

train.shape
lis=[]

for i in range(0,42000):

    lis.append(train[i].reshape(28,28,1))

lis=np.array(lis)

lis.shape

lis_inverse=[]

for i in lis:

    lis_inverse.append((i.T).reshape(28,28,1))

final_set=[]

for i in lis:

    final_set.append(i)

for i in lis_inverse:

    final_set.append(i)

final_set=np.array(final_set)

final_set.shape

# plt.imshow(final_set[12].reshape(28,28))
final_label=[]

for i in labels:

    final_label.append(i)

for i in labels:

    final_label.append(i)

final_label=np.array(final_label)

final_label.shape
labels.shape

labels[34]

# labels are the correct present here yes

p.shape
# trying the custom model

model=tf.keras.models.Sequential(

[

    tf.keras.layers.Conv2D(filters=128,kernel_size=3,input_shape=(28,28,1),activation="relu"),

    tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=3,padding="same"),

    tf.keras.layers.Dropout(0.3),

#     tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu"),

#     tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=3,padding="same"),

#     tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation="relu"),

    tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=3,padding="same"),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"),

    tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=3,padding="same"),

    tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"),

#     tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=3,padding="same"),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dense(128,activation="relu"),

#     tf.keras.layers.Dropout(0.3),

#     tf.keras.layers.Dense(128,activation="relu"),

#     tf.keras.layers.Dense(128,activation="relu"),

#     tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dense(64,activation="relu"),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32,activation="relu"),

    tf.keras.layers.Dense(10,activation="softmax")

])

# # coustom model does not do well here

# model=tf.keras.applications.InceptionV3(weights="imagenet")

# model.trainable=False

# model=tf.keras.layers.Sequential([

#     model,

#     tf.keras.layers.Dense(128,activation="relu",input_shape=(10,)),

#     tf.keras.layers.Dense(10,activation="softmax")

# ])
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dropout,Flatten,Dense

model = tf.keras.Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
# labels.shape

# lis.shape

model.fit(final_set,final_label,epochs=45,batch_size=200)

# model2=model

# model2.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

file_names

test=pd.read_csv(file_names[2])



test=test.to_numpy()
lis=[]

for i in range(0,28000):

    lis.append(test[i].reshape(28,28,1))

lis=np.array(lis)
model.evaluate(final_set[32:6787],final_label[32:6787])
answer=model.predict(lis)
answers=[]

for i in answer:

    answers.append(np.argmax(i))
answers=np.array(answers)

answers=pd.Series(answers)

ImageId=[]

for i in range(1,28001):

    ImageId.append(i)

ImageId=pd.Series(ImageId)
Submission=pd.DataFrame(data={"ImageId":ImageId,"Label":answers})

Submission.index+=1

Submission
from IPython.display import HTML

import pandas as pd

import numpy as np

df = Submission



df.to_csv('this_submission.csv',index=False)



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='this_submission.csv')
print("hello")