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
train=pd.read_csv("../input/train.csv")
len(train)
train_data=[]

valid_data=[]

for index, row in train.iterrows():

    a=row['id']

    filepath='../input/train/train/'+a

    if index<15000:

        train_data.append((filepath,row['has_cactus']))

    else:

        valid_data.append((filepath,row['has_cactus']))
from random import shuffle

from PIL import Image

from skimage import io

from skimage.transform import resize

from numpy import newaxis

image_size=32

no_classes=2
def load_image(file_path):

    img=Image.open(file_path)

    img=img.resize((image_size,image_size))

    X=np.array(img)/255

    return X
def train_generator(data, batch_size):

    while True:

        shuffle(data)

        batch_images =[]

        batch_labels=[]

        for batch_files in data[:batch_size]:

            image=load_image(batch_files[0])

            label=batch_files[1]

            batch_images.append(image)

            batch_labels.append(label)

        batch_images=np.array(batch_images)

        batch_labels=np.array(batch_labels)

        yield batch_images, batch_labels
import tensorflow as tf

model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(image_size,image_size,3)),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(2,2),activation='relu'),

    tf.keras.layers.Conv2D(128,(2,2),activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(no_classes,activation='softmax')

])
model.summary()
model.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
cbl = [

    EarlyStopping(

        monitor='val_loss',

        patience=20,

    ),

    ModelCheckpoint(

        filepath='best_model.h5',

        monitor='val_loss',

        save_best_only=True,

    ),

    ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.1,

        patience=10

    )

]
model.fit_generator(train_generator(train_data,256), steps_per_epoch=len(train_data)//256,validation_data=train_generator(valid_data,256),

                    validation_steps=len(valid_data)//256,epochs=100,callbacks=cbl)
model1=tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(image_size,image_size,3)),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(256,activation='relu'),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(no_classes,activation='softmax')

])
model1.summary()
model1.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=['acc'])
cbl = [

    EarlyStopping(

        monitor='val_loss',

        patience=20,

    ),

    ModelCheckpoint(

        filepath='best_model1.h5',

        monitor='val_loss',

        save_best_only=True,

    ),

    ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.1,

        patience=10

    )

]
model1.fit_generator(train_generator(train_data,256), steps_per_epoch=len(train_data)//256,validation_data=train_generator(valid_data,256),

                    validation_steps=len(valid_data)//256,epochs=200,callbacks=cbl)
model2=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(image_size,image_size,3)),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(no_classes,activation='softmax')

])
model2.summary()
model2.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=['acc'])
cbl = [

    EarlyStopping(

        monitor='val_loss',

        patience=20,

    ),

    ModelCheckpoint(

        filepath='best_mode2.h5',

        monitor='val_loss',

        save_best_only=True,

    ),

    ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.1,

        patience=10

    )

]
model2.fit_generator(train_generator(train_data,256), steps_per_epoch=len(train_data)//256,validation_data=train_generator(valid_data,256),

                    validation_steps=len(valid_data)//256,epochs=100,callbacks=cbl)
len(os.listdir("../input/test/test"))
submission=pd.read_csv("../input/sample_submission.csv")
def most_common(lst):

    return max(set(lst), key=lst.count)
for index, row in submission.iterrows():

    image=row['id']

    filepath="../input/test/test/"+image

    img=load_image(filepath)

    img=img[newaxis,:, :, :]

    result=np.argmax(model.predict(img))

    result1=np.argmax(model1.predict(img))

    result2=np.argmax(model2.predict(img))

    a=[result,result1,result2]

    submission.at[index,'has_cactus'] = most_common(a)

    
submission.to_csv('submission.csv',index=False, columns=['id','has_cactus'])