from keras.applications.densenet import DenseNet121
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.vgg16 import VGG16

IMG_ROW=IMG_COL=224
IMG_CHANNEL=3
IMG_DIR='../input/sample/images/'
labeldata=pd.read_csv('../input/sample_labels.csv')
print(labeldata.head())
labellist=[]
for label in labeldata['Finding Labels']:
    if(label=='No Finding'):
        labellist.append(0)
    else:
        labellist.append(1)
labeldf=to_categorical(labellist)
train_imgpath,valid_imgpath,train_label,valid_label=train_test_split(labeldata['Image Index'],labeldf,
                                                                   test_size=0.15)
def read_img(imgpath):
    img=cv2.imread(imgpath)
    img=cv2.resize(img,(IMG_ROW,IMG_COL))
    return img
def train_gen(batch_size=40):
    while(True):
        imglist=np.zeros((batch_size,IMG_ROW,IMG_COL,IMG_CHANNEL))
        labellist=np.zeros((batch_size,2))
        for i in range(batch_size):
            rndid=random.randint(0,len(train_imgpath)-1)
            img=read_img(IMG_DIR+train_imgpath[train_imgpath.index[rndid]])
            label=train_label[rndid]
            imglist[i]=img
            labellist[i]=label
        yield (imglist,labellist)

def valid_gen():
    imglist=np.zeros((len(valid_imgpath),IMG_ROW,IMG_COL,IMG_CHANNEL))
    labellist=np.zeros((len(valid_imgpath),2))
    for i in range(len(valid_imgpath)):
        img=read_img(IMG_DIR+valid_imgpath[valid_imgpath.index[i]])
        label=valid_label[i]
        imglist[i]=img
        labellist[i]=label
    return (imglist,labellist)
validdata=valid_gen()
labelset=set()
for labelstr in labeldata['Finding Labels']:
    labelarr=labelstr.split('|')
    for label in labelarr:
        if(label not in labelset):
            labelset.add(label)
labellist=list(labelset)
labelsequence=np.zeros((len(labeldata),len(labellist)))
for i,labelstr in enumerate(labeldata['Finding Labels']):
    labelarr=labelstr.split('|')
    for label in labelarr:
        index1=labellist.index(label)
        labelsequence[i][index1]=1
print(labelsequence[:5])
def build_model():
    model = DenseNet121(include_top=False, input_shape = (IMG_ROW,IMG_COL,IMG_CHANNEL),
                                          weights=None)
    new_output =GlobalAveragePooling2D()(model.output)
    new_output=Dense(128,activation='relu')(new_output)
    
    new_output = Dense(2,activation='softmax')(new_output)
    model = Model(model.inputs, new_output)
    return model
model=build_model()
model.compile(metrics=['accuracy'],loss='categorical_crossentropy',
             optimizer='Adam')
callbacks=[
    ModelCheckpoint('model.h5',monitor='val_loss',save_best_only=True,verbose=1),
    ReduceLROnPlateau(monitor='val_loss',patience=3,min_lr=1e-8,verbose=1)
]
history=model.fit_generator(train_gen(),epochs=30,
                            steps_per_epoch=100,
                            validation_data=validdata,
                           callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])