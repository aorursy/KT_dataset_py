# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input/he_challenge_data/data'))

os.chdir('/kaggle/input/he_challenge_data/data')



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv('train.csv')

test_df=pd.read_csv('test.csv')

train_df.head()
from keras.applications.vgg16 import VGG16

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import Dense

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
train_path='train/'

test_path='test/'



train_images=[]

train_labels=[]



for i in range(len(train_df.image_id)):

    img=cv2.imread(train_path+str(train_df.image_id[i])+'.jpg')

    img=cv2.resize(img,(224,224))

    train_images.append(img)

    train_labels.append(train_df.category[i])
train_images=np.array(train_images)

test_images=np.array(test_images)



print('Train shape: {}'.format(train_images.shape))

print('Test shape: {}'.format(test_images.shape))
train_images=train_images.reshape(train_images.shape[0],224,224,3)

test_images=test_images.reshape(test_images.shape[0],224,224,3)



print('Train shape: {}'.format(train_images.shape))

print('Test shape: {}'.format(test_images.shape))
train_gen = ImageDataGenerator(featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_images,to_categorical(train_labels),random_state=42)



print('X-Train shape: {}'.format(X_test.shape))

print('X-Test shape: {}'.format(X_test.shape))

print('Y-Train shape: {}'.format(y_train.shape))

print('Y-Test shape: {}'.format(y_test.shape))
vgg=VGG16(weights='imagenet',include_top=True)

vgg.summary()
model=Sequential()

for layer in vgg.layers[:-1]:

    model.add(layer)



for layer in model.layers[:]:

    layer.trainable=False

model.add(Dense(103,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
history=model.fit(X_train,y_train,epochs=100,batch_size=32,validation_data=(X_test,y_test),callbacks=[EarlyStopping(min_delta=0.01)])
submission=pd.read_csv('sample_submission.csv')

submission.head()
test_id=[]

test_pred=[]



for i in submission.image_id:

    img=cv2.resize(cv2.imread('test/'+str(i)+'.jpg'),(224,224))

    img=np.expand_dims(img,axis=0)

    test_id.append(i)

    test_pred.append(int(model.predict_classes(img)))
final_submission=pd.DataFrame({'image_id':test_id,'category':test_pred})

final_submission.head()
final_submission.to_csv('/kaggle/final_submission.csv',index=False)