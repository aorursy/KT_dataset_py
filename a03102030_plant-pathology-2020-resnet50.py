from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten

from keras.layers import GlobalAveragePooling2D

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.layers.normalization import BatchNormalization

import os 

import pandas as pd

import plotly.graph_objs as go

import matplotlib.ticker as ticker

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import cv2

import numpy as np

from sklearn.model_selection import train_test_split
os.listdir('../input/plant-pathology-2020-fgvc7')
train=pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")

test=pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train['image_id']=train['image_id']+'.jpg'

test['image_id']=test['image_id']+'.jpg'

train.head()
fig,ax=plt.subplots(2,2,figsize=(20,20))

sns.barplot(y=train.healthy.value_counts(),x=train.healthy.value_counts().index,ax=ax[0,0])

ax[0,0].set_title("Value count for healthy",size=20)

ax[0,0].set_xlabel('healthy',size=18)

ax[0,0].set_ylabel('',size=18)



sns.barplot(y=train.multiple_diseases.value_counts(),x=train.multiple_diseases.value_counts().index,ax=ax[0,1])

ax[0,1].set_title("Value count for multiple_diseases",size=20)

ax[0,1].set_xlabel('multiple_diseases',size=18)

ax[0,1].set_ylabel('',size=18)



sns.barplot(y=train.rust.value_counts(),x=train.rust.value_counts().index,ax=ax[1,0])

ax[1,0].set_title("Value count for rust",size=20)

ax[1,0].set_xlabel('rust',size=18)

ax[1,0].set_ylabel('',size=18)



sns.barplot(y=train.scab.value_counts(),x=train.scab.value_counts().index,ax=ax[1,1])

ax[1,1].set_title("Value count for scab",size=20)

ax[1,1].set_xlabel('healthy',size=18)

ax[1,1].set_ylabel('',size=18)

img=[]

filename=train.image_id

for file in filename:

    image=cv2.imread("../input/plant-pathology-2020-fgvc7/images/"+file)

    res=cv2.resize(image,(256,256))

    img.append(res)

img=np.array(img)

print(img.shape)
plt.figure(figsize=(15,15))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(img[i])
train_labels = np.float32(train.loc[:, 'healthy':'scab'].values)



train, val = train_test_split(train, test_size = 0.15)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( horizontal_flip=True,

    vertical_flip=True,

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range=.1,

    fill_mode='nearest',

    shear_range=0.1,

    rescale=1/255,

    brightness_range=[0.5, 1.5])

train_generator=train_datagen.flow_from_dataframe(train,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',

                                                      target_size=(384,384),

                                                      x_col="image_id",

                                                      y_col=['healthy','multiple_diseases','rust','scab'],

                                                      class_mode='raw',

                                                      shuffle=False,

                                                       subset='training',

                                                      batch_size=32)
val_generator=train_datagen.flow_from_dataframe(val,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',

                                                      target_size=(384,384),

                                                      x_col="image_id",

                                                      y_col=['healthy','multiple_diseases','rust','scab'],

                                                      class_mode='raw',

                                                      shuffle=False,

                                                      batch_size=32,

                                                  )
test_generator=train_datagen.flow_from_dataframe(test,directory='/kaggle/input/plant-pathology-2020-fgvc7/images/',

                                                      target_size=(384,384),

                                                      x_col="image_id",

                                                      y_col=None,

                                                      class_mode=None,

                                                      shuffle=False,

                                                      batch_size=32)
from keras.applications.resnet50 import ResNet50

from keras.models import Model

import keras

from keras import optimizers

model_finetuned = ResNet50(include_top=False, weights='imagenet', input_shape=(384,384,3))

x = model_finetuned.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

x = Dense(64, activation="relu")(x)

predictions = Dense(4, activation="softmax")(x)

model_finetuned = Model(inputs=model_finetuned.input, outputs=predictions)

model_finetuned.compile(optimizer='adam',

                  loss = 'categorical_crossentropy',

                  metrics=['accuracy'])

model_finetuned.summary()
from keras.callbacks import ReduceLROnPlateau
history_1 = model_finetuned.fit_generator(train_generator,                                    

                                  steps_per_epoch=100, 

                                  epochs=25,validation_data=val_generator,validation_steps=100

                                  ,verbose=1,callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=3, min_lr=0.000001)],use_multiprocessing=False,

               shuffle=True)
fig = go.Figure(data=[

    go.Line(name='train_acc', x=history_1.epoch, y=history_1.history['accuracy']),

    go.Line(name='Val_acc', x=history_1.epoch, y=history_1.history['val_accuracy'])])



fig.update_layout(

    title="Accuracy",

    xaxis_title="epoch",

    yaxis_title="accuracy",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    ))

fig
SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"



sub = pd.read_csv(SUB_PATH)

probs_RESNET = model_finetuned.predict(test_generator, verbose=1)

sub.loc[:, 'healthy':] = probs_RESNET

sub.to_csv('submission_RESNET.csv', index=False)

sub.head()