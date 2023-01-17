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
import cv2,numpy as np,pandas
from matplotlib import pyplot as pt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
import random
from sklearn import metrics
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model
dt_path_train='/kaggle/input/intel-image-classification/seg_train/seg_train'
dt_path_test='/kaggle/input/intel-image-classification/seg_test/seg_test'
dt_path_pred='/kaggle/input/intel-image-classification/seg_pred/seg_pred'
lst_train=os.listdir(dt_path_train)
lst_test=os.listdir(dt_path_test)
lst_pred=os.listdir(dt_path_pred)
label_dict={'sea':0,'forest':1,'mountain':2,'glacier':3,'buildings':4,'street':5}
num_keys={j:i for i ,j in label_dict.items()}
images_train=[]
number_train=[]
types_train=[]
for i in range(0,len(lst_train)):
    images_path_train=dt_path_train+'/'+lst_train[i]
    imges_train=os.listdir(images_path_train)
    number_train.append(len(imges_train))
    print(lst_train[i])
    for m in imges_train:
        img=cv2.imread(images_path_train+'/'+m)
        resized=cv2.resize(img,(224,224))
        images_train.append(resized)
        types_train.append(label_dict[lst_train[i]])
types_arr_train=np.array(types_train)
images_test=[]
number_test=[]
types_test=[]
for i in range(0,len(lst_test)):
    images_path_test=dt_path_test+'/'+lst_test[i]
    imges_test=os.listdir(images_path_test)
    number_test.append(len(imges_test))
    print(lst_test[i])
    for m in imges_test:
        img=cv2.imread(images_path_test+'/'+m)
        resized=cv2.resize(img,(224,224))
        images_test.append(resized)
        types_test.append(label_dict[lst_test[i]])
types_arr_test=np.array(types_test)
imges_data_train=np.array(images_train)
imges_data_test=np.array(images_test)
imges_data_train.shape,types_arr_train.shape,imges_data_test.shape,types_arr_test.shape
train_y=to_categorical(types_arr_train,num_classes=6,dtype='int')
test_y=to_categorical(types_arr_test,num_classes=6,dtype='int')
pt.figure(figsize=(20,20))
for i in range(0,64):
    pt.subplot(8,8,i+1)
    ig=random.randint(0,len(imges_data_train))
    pt.imshow(imges_data_train[ig].reshape(224,224,3))
    pt.xticks([])
    pt.yticks([])
    pt.title("Class name {}".format(num_keys[types_train[ig]]))
pt.subplots_adjust(wspace=0.15)
pt.show()
pt.figure(figsize=(20,20))
for i in range(0,64):
    pt.subplot(8,8,i+1)
    ig=random.randint(0,len(imges_data_test))
    pt.imshow(imges_data_test[ig].reshape(224,224,3))
    pt.xticks([])
    pt.yticks([])
    #num=types_arr_train.iloc[idx]
    pt.title("Class name {}".format(num_keys[types_test[ig]]))
pt.subplots_adjust(wspace=0.15)
pt.show()
vgg=VGG16(input_shape=(224,224,3),weights='imagenet',include_top=True)
vgg.summary()
y=vgg.get_layer('fc2').output
y=Dense(6,activation='softmax')(y)
intel=Model(input=vgg.input,output=y)
intel.summary()
for i in intel.layers[:-1]:
    i.trainable=False
intel.summary()
intel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
intel_history=intel.fit(imges_data_train,train_y,epochs=50,batch_size=512,
          validation_data=(imges_data_test,test_y))
pt.plot(intel_history.history['accuracy'],label='train')
pt.plot(intel_history.history['val_accuracy'],label='test')
pt.legend(loc='lower right')
pt.show()
pt.plot(intel_history.history['loss'],label='train')
pt.plot(intel_history.history['val_loss'],label='test')
pt.legend(loc='upper right')
pt.show()
intel_pred=intel.predict(imges_data_test)
intel_output=(intel_pred>0.5)
print(metrics.classification_report(test_y,intel_output))
pt.figure(figsize=(20,20))
for i in range(0,20):
    radm=random.randint(0,len(lst_pred))
    prd=dt_path_pred+'/'+lst_pred[radm]
    imread=cv2.imread(prd)
    read=cv2.resize(imread,(224,224))
    read_array=np.array(read)
    prediction=np.argmax(intel.predict(read_array.reshape(-1,224,224,3)))
    pt.subplot(5,4,i+1)
    pt.imshow(read)
    pt.xticks([])
    pt.yticks([])
    pt.title("predicted:{}".format(num_keys[prediction]))
pt.show()
vgg_tune=VGG16(input_shape=[224,224,3],weights='imagenet',include_top=False)
vgg_tune.summary()
x=vgg_tune.get_layer('block5_pool').output
x=Flatten()(x)
x=Dense(units=256,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(units=128,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(units=32,activation='relu')(x)
x=Dropout(0.1)(x)
y=Dense(units=6,activation='softmax')(x)
intel_fine=Model(input=vgg_tune.input,output=y)
intel_fine.summary()
for layer in intel_fine.layers[:-8]:
    layer.trainable=False
intel_fine.summary()
intel_fine.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
fine_history=intel_fine.fit(imges_data_train,train_y,epochs=25,batch_size=512,
          validation_data=(imges_data_test,test_y))
pt.plot(fine_history.history['accuracy'],label='train')
pt.plot(fine_history.history['val_accuracy'],label='test')
pt.legend(loc='lower right')
pt.show()
pt.plot(fine_history.history['loss'],label='train')
pt.plot(fine_history.history['val_loss'],label='test')
pt.legend(loc='upper right')
pt.show()
fine_output=intel_fine.predict(imges_data_test)
fine_predict=(fine_output>0.5)
print(metrics.classification_report(test_y,fine_predict))
pt.figure(figsize=(20,20))
for i in range(0,20):
    radm=random.randint(0,len(lst_pred))
    prd=dt_path_pred+'/'+lst_pred[radm]
    imread=cv2.imread(prd)
    read=cv2.resize(imread,(224,224))
    read_array=np.array(read)
    prediction=np.argmax(intel_fine.predict(read_array.reshape(-1,224,224,3)))
    pt.subplot(5,4,i+1)
    pt.imshow(read)
    pt.xticks([])
    pt.yticks([])
    pt.title("predicted:{}".format(num_keys[prediction]))
pt.show()
























