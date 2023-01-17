import tensorflow as tf

import numpy as np

import random

import os

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras import models

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow.keras import models, layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

train.head()
train.head()
ttrain = train.copy()
train1 = train[train['target']==1]

train0 = train[train['target']==0]
train = pd.concat([train0.iloc[0:2836,:],train1,train1,train1,train1,train1])
labels=[]

data=[]

for i in range(train.shape[0]):

    data.append(train_dir + train['image_name'].iloc[i]+'.jpg')

    labels.append(train['target'].iloc[i])

df=pd.DataFrame(data)

df.columns=['images']

df['target']=labels



test_data=[]

for i in range(test.shape[0]):

    test_data.append(test_dir + test['image_name'].iloc[i]+'.jpg')

df_test=pd.DataFrame(test_data)

df_test.columns=['images']



X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], test_size=0.2, random_state=1234)



train=pd.DataFrame(X_train)

train.columns=['images']

train['target']=y_train



validation=pd.DataFrame(X_val)

validation.columns=['images']

validation['target']=y_val



train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(

    train,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    shuffle=True,

    class_mode='raw')



validation_generator = val_datagen.flow_from_dataframe(

    validation,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode='raw')
df_test.head()
model = models.Sequential()

eff = tf.keras.applications.EfficientNetB5(

    include_top=False,

    weights="imagenet",

    input_shape=(224,224,3) )

model.add(eff)

model.add(layers.Flatten())

model.add(layers.Dense(units=1000, activation='relu'))

model.add(layers.Dense(units=1000, activation='relu'))

model.add(layers.Dense(units=500, activation='relu'))

model.add(layers.Dense(units=1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

fit = model.fit_generator(train_generator, steps_per_epoch=10, epochs=40,

                         validation_data=validation_generator, validation_steps=10)
test_datagen = ImageDataGenerator(rescale=1./255)
start = 0

end=500
pred2 = np.array([])
# 1

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 2

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500



# 3

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 4

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 5

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 6

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 7

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 8

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 9

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 10

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 11

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 12

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 13

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 14

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 15

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 16

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 17

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 18

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 19

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 20

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
df_test.shape
# 21

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:end],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
# 22

test_generator = test_datagen.flow_from_dataframe(

    df_test[start:],

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode=None)

pred1= np.ravel(model.predict_generator(test_generator))

pred2 = np.append(pred2,pred1)

start= end

end = start+500
df_test.head()
submission['target']= pred2
submission.to_csv('submission.csv',index=False)