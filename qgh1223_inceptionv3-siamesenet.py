from keras.applications.vgg16 import VGG16
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import matplotlib.pyplot as plt
import random
from keras.models import Model,Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenetv2 import MobileNetV2

IMG_ROW=IMG_COL=144
BASE_DIR='../input/train/'
labelpath='../input/train.csv'
traindata=pd.read_csv(labelpath)
def build_siamese_model( img_shape,lr,branch_model,activation='sigmoid'):

    optim  = Adam(lr=lr)
    
    
    
    mid        = 32
    xa_inp     = Input(shape=branch_model.output_shape[1:])
    xb_inp     = Input(shape=branch_model.output_shape[1:])
    x1         = Lambda(lambda x : x[0]*x[1])([xa_inp, xb_inp])
    x2         = Lambda(lambda x : x[0] + x[1])([xa_inp, xb_inp])
    x3         = Lambda(lambda x : K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4         = Lambda(lambda x : K.square(x))(x3)
    x          = Concatenate()([x1, x2, x3, x4])
    x          = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x          = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x          = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x          = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x          = Flatten(name='flatten')(x)
    x          = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')
    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    x          = head_model([xa, xb])
    model      = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=[ 'accuracy'])
    return model,head_model
def kind_list(imgdata):
    kindlist=imgdata.groupby('Id').size()
    return kindlist.index

def fetch_all_kind_list(imgdata):
    kindlist=kind_list(imgdata)
    kindimgpathlist=[]
    for kind in kindlist:
        kindimgpathlist.append(list(imgdata['Image'][imgdata['Id']==kind]))
    return kindimgpathlist,kindlist

def fetch_kind_list_split(kindimgpathlist,split_size=0.8):
    trainkindimgpathlist=[]
    validkindimgpathlist=[]
    for pathlist in kindimgpathlist:
        if(len(pathlist)<=3):
            trainkindimgpathlist.append(pathlist)
            validkindimgpathlist.append(pathlist)
        else:
            trainkindimgpathlist.append(pathlist[:int(len(pathlist)*split_size)])
            validkindimgpathlist.append(pathlist[int(len(pathlist)*split_size):])
    return trainkindimgpathlist,validkindimgpathlist
def imgarr(imgpath):
    img=cv2.imread(imgpath)
    return img
def siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,kindimgpathlist,
                    contrast_times=5,batch_size=50):
    while(True):
        imglist1=[]
        imglist2=[]
        labellist=[]
        for i in range(batch_size):
            for j in range(contrast_times):
                rndid=random.randint(0,len(kindimgpathlist)-1)
                if(i%2==0):
                    #print(len(kindimgpathlist[rndid]))
                    pair=np.random.randint(0,len(kindimgpathlist[rndid]),2)
                    imgpath1=kindimgpathlist[rndid][pair[0]]
                    imgpath2=kindimgpathlist[rndid][pair[1]]
                    labellist.append(1)
                else:
                    rndid1=random.randint(0,len(kindimgpathlist[rndid])-1)
                    imgpath1=kindimgpathlist[rndid][rndid1]
                    index1=random.choice([num for num in range(len(kindimgpathlist)) if num not in [rndid]])
                    rndid2=random.randint(0,len(kindimgpathlist[index1])-1)
                    imgpath2=kindimgpathlist[index1][rndid2]
                    labellist.append(0)
                img1=imgarr(BASE_DIR+imgpath1)
                img2=imgarr(BASE_DIR+imgpath2)
                img1=cv2.resize(img1,(IMG_ROW,IMG_COL))
                img2=cv2.resize(img2,(IMG_ROW,IMG_COL))
                imglist1.append(img1)
                imglist2.append(img2)
        yield ([np.asarray(imglist1),np.asarray(imglist2)],np.asarray(labellist))
img_shape=(IMG_ROW,IMG_COL,3)
modelfn=InceptionV3(weights=None,
                   input_shape=img_shape,
                   classes=300)
model,head_model = build_siamese_model(img_shape,64e-5,modelfn)
model.summary()
model.summary()
model.compile(optimizer=Adam(0.001),metrics=['accuracy'],
              loss=['binary_crossentropy'])
callbacks=[
    ReduceLROnPlateau(monitor='val_loss',patience=5,min_lr=1e-9,verbose=1,mode='min'),
    ModelCheckpoint('siamese.h5',monitor='val_loss',save_best_only=True,verbose=1)
]
kindimgpathlist,kindlist=fetch_all_kind_list(traindata)
trainkindimgpathlist,validkindimgpathlist=fetch_kind_list_split(kindimgpathlist)
history=model.fit_generator(siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,
                                            trainkindimgpathlist,batch_size=30),
                            steps_per_epoch=20,
                            epochs=80,
                            validation_data=siamese_img_gen(BASE_DIR,IMG_ROW,IMG_COL,
                                                            validkindimgpathlist,contrast_times=10,batch_size=5),
                            validation_steps=20,
                            callbacks=callbacks)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])
plt.show()
modelfn.save('mobile_encoder.h5')