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
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img
from keras.applications.inception_v3 import InceptionV3
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
        resized=cv2.resize(img,(150,150))
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
        resized=cv2.resize(img,(150,150))
        images_test.append(resized)
        types_test.append(label_dict[lst_test[i]])
types_arr_test=np.array(types_test)
imges_data_train=np.array(images_train)
imges_data_test=np.array(images_test)
pt.figure(figsize=(20,20))
for i in range(0,64):
    pt.subplot(8,8,i+1)
    ig=random.randint(0,len(imges_data_train))
    pt.imshow(imges_data_train[ig].reshape(150,150,3))
    pt.xticks([])
    pt.yticks([])
    #num=types_arr_train.iloc[idx]
    pt.title("Class name {}".format(num_keys[types_train[ig]]))
pt.subplots_adjust(wspace=0.15)
pt.show()
pt.figure(figsize=(20,20))
for i in range(0,64):
    pt.subplot(8,8,i+1)
    ig=random.randint(0,len(imges_data_test))
    pt.imshow(imges_data_test[ig].reshape(150,150,3))
    pt.xticks([])
    pt.yticks([])
    #num=types_arr_train.iloc[idx]
    pt.title("Class name {}".format(num_keys[types_test[ig]]))
pt.subplots_adjust(wspace=0.15)
pt.show()
imges_data_train=imges_data_train/255
imges_data_test=imges_data_test/255
imges_data_train.shape,types_arr_train.shape,imges_data_test.shape,types_arr_test.shape
train_y=to_categorical(types_arr_train,num_classes=6,dtype='int')
test_y=to_categorical(types_arr_test,num_classes=6,dtype='int')
model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Dropout(0.4))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model_history=model.fit(imges_data_train,train_y,batch_size=256,epochs=25,
                        validation_data=(imges_data_test,test_y))
predict_test=model.predict_classes(imges_data_test)
predict_test_type=to_categorical(predict_test,num_classes=6,dtype='int')
print(metrics.classification_report(test_y,predict_test_type))
pt.plot(model_history.history['accuracy'],label='train')
pt.plot(model_history.history['val_accuracy'],label='test')
pt.legend()
pt.show()
pt.plot(model_history.history['loss'],label='train')
pt.plot(model_history.history['val_loss'],label='test')
pt.legend()
pt.show()
metrics.confusion_matrix(np.argmax(test_y,axis=1),np.argmax(predict_test_type,axis=1))
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
                  directory=dt_path_train ,
        target_size=(150, 150),
        batch_size=256,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        directory=dt_path_test,
        target_size=(150, 150),
        batch_size=256,
        class_mode='categorical')
gen=Sequential()
gen.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
gen.add(Dropout(0.4))
gen.add(MaxPool2D(pool_size=(2,2)))
gen.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
gen.add(Dropout(0.2))
gen.add(MaxPool2D(pool_size=(2,2)))
gen.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
gen.add(Dropout(0.2))
gen.add(Flatten())
gen.add(Dense(units=128,activation='relu'))
gen.add(Dropout(0.4))
gen.add(Dense(units=32,activation='relu'))
gen.add(Dropout(0.2))
gen.add(Dense(units=6,activation='softmax'))
gen.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
gen_history=gen.fit_generator(train_generator,steps_per_epoch=14034//256,epochs=50,
                              validation_data=validation_generator,validation_steps=3000//256)
pt.plot(gen_history.history['accuracy'],label='train')
pt.plot(gen_history.history['val_accuracy'],label='test')
pt.legend()
pt.show()
pt.plot(gen_history.history['loss'],label='train')
pt.plot(gen_history.history['val_loss'],label='test')
pt.legend()
pt.show()
prdct_gen_lbl=gen.predict(imges_data_test)
print(metrics.classification_report(test_y,prdct_lbl))
prdct_lbl=(prdct_gen_lbl>0.5)
inceptn=InceptionV3(input_shape=(150,150,3),weights='imagenet',include_top=False)
for i in inceptn.layers:
    i.trainable=False
x=Flatten()(inceptn.output)
y=Dense(6,activation='softmax')(x)
mdl=Model(input=inceptn.input,output=y)
mdl.summary()
train_trnsfr_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_trsfr_datagen = ImageDataGenerator(rescale=1./255)

train_trsfr_generator = train_trnsfr_datagen.flow_from_directory(
                  directory=dt_path_train ,
        target_size=(150, 150),
        batch_size=256,
        class_mode='categorical')
validation_trsfr_generator = test_trsfr_datagen.flow_from_directory(
        directory=dt_path_test,
        target_size=(150, 150),
        batch_size=256,
        class_mode='categorical')
mdl.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
mdl.fit_generator(train_trsfr_generator,epochs=10,steps_per_epoch=14034//256,
                  validation_data=validation_trsfr_generator,validation_steps=3000//256)





