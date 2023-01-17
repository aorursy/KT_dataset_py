import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import random
from keras.models import Model,Sequential
from keras.layers.core import Activation,Reshape,Permute
from keras.layers.convolutional import Convolution2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input,merge,Conv2D,Concatenate
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.losses import binary_crossentropy
from keras import backend as K
positionpath='../input/list_landmarks_align_celeba.csv'
IMG_DIR='../input/img_align_celeba/img_align_celeba/'
IMG_ROW=IMG_COL=64
positiondata=pd.read_csv(positionpath)
print(positiondata.head())
print(positiondata.columns)
column1=positiondata.columns[1:]
color1=[(255,0,0),(0,255,0),(0,0,255),
       (255,255,0),(255,0,255)]
def draw_box(index1):
    imgpath=IMG_DIR+positiondata['image_id'][index1]
    img=cv2.imread(imgpath)
    for i in range(len(column1)):
        if(i%2==0):
            x1=positiondata[column1[i]][index1]
            y1=positiondata[column1[i+1]][index1]
            img=cv2.rectangle(img,(x1-10,y1-10),
                             (x1+10,y1+10),
                              color1[int(i/2)])
        else:
            continue
    return img
def generate_maskarr(index1):
    imgpath=IMG_DIR+positiondata['image_id'][index1]
    img=cv2.imread(imgpath)
    maskarr=np.zeros((img.shape[0],img.shape[1],img.shape[2]))
    for i in range(len(column1)):
        if(i%2==0):
            x1=positiondata[column1[i]][index1]
            y1=positiondata[column1[i+1]][index1]
            for j in range(10):
                for k in range(10):
                    maskarr[x1-j][y1-k]=color1[int(i/2)]
                    maskarr[x1+j][y1+k]=color1[int(i/2)]
        else:
            continue
    img=cv2.resize(img,(IMG_ROW,IMG_COL))
    maskarr=cv2.resize(maskarr,(IMG_ROW,IMG_COL))
    return img,maskarr
def imgarr_trainmaskarr(positiondata,index1):
    index1=positiondata.index[index1]
    imgpath=IMG_DIR+positiondata['image_id'][index1]
    img=cv2.imread(imgpath)
    img1=cv2.resize(img,(IMG_ROW,IMG_COL))
    maskarr=np.zeros((IMG_ROW*IMG_COL,len(color1)))
    for i in range(len(column1)):
        if(i%2==0):
            centerx=int(positiondata[column1[i]][index1]*IMG_ROW/img1.shape[0])
            centery=int(positiondata[column1[i]][index1]*IMG_COL/img1.shape[1])
            startx=centerx-int(10*IMG_ROW/img1.shape[0])
            starty=centery-int(10*IMG_ROW/img.shape[1])
            endx=centerx+int(10*IMG_ROW/img1.shape[0])
            endy=centery+int(10*IMG_ROW/img.shape[1])
            for j in range(starty*IMG_ROW+startx,endy*IMG_ROW+endx):
                if(j<IMG_ROW*IMG_COL):
                    maskarr[j][int(i/2)]=1
    return img1,maskarr
for i in range(10):
    rnd_id=random.randint(0,len(positiondata)-1)
    f,axes=plt.subplots(1,2,figsize=(15,2))
    j=0
    for ax in axes:
        if(j==0):
            img=cv2.imread(IMG_DIR+positiondata['image_id'][rnd_id])
            ax.imshow(img/255)
        else:
            img=draw_box(rnd_id)
            ax.imshow(img/255)
        j+=1
img,maskarr=imgarr_trainmaskarr(positiondata,1)
print(img.shape)
print(maskarr.shape)
def train_gen(batch_size=100):
    while(True):
        traindata=positiondata.sample(n=int(len(positiondata)*0.5))
        imgs=[]
        masks=[]
        for i in range(batch_size):
            rnd_id=random.randint(0,len(traindata)-1)
            img,maskarr=imgarr_trainmaskarr(traindata,rnd_id)
            imgs.append(img)
            masks.append(maskarr)
        yield (np.asarray(imgs),np.asarray(masks))
def test_data():
    test_positiondata=positiondata.sample(n=int(len(positiondata)*0.01))
    imgs=[]
    masks=[]
    for i in range(len(test_positiondata)):
        img,maskarr=imgarr_trainmaskarr(test_positiondata,i)
        imgs.append(img)
        masks.append(maskarr)
    return np.asarray(imgs),np.asarray(masks)
testimgs,testmasks=test_data()
def build_model(img_w, img_h, filters):
    n_labels = len(color1)

    kernel = 3

    encoding_layers = [
        Conv2D(64, (kernel, kernel), input_shape=(img_h, img_w, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),

        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]

    autoencoder =Sequential()
    autoencoder.encoding_layers = encoding_layers

    for l in autoencoder.encoding_layers:
        autoencoder.add(l)

    decoding_layers = [
        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(512, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(256, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(128, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        UpSampling2D(),
        Convolution2D(64, (kernel, kernel), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Convolution2D(n_labels, (1, 1), padding='valid', activation="sigmoid"),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Reshape((n_labels, img_h * img_w)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))

    #with open('model_5l.json', 'w') as outfile:
    #    outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))
    autoencoder.summary()
    return autoencoder
model=build_model(IMG_ROW,IMG_COL,10)
optimizer=SGD(lr=0.001,momentum=0.9,decay=0.0005)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,
              metrics=['accuracy'])
callbacks=[
    ReduceLROnPlateau(patience=3,monitor='val_loss',verbose=1),
    ModelCheckpoint('face_segnet_model.h5',save_best_only=True)
]
history=model.fit_generator(train_gen(),validation_data=(testimgs,testmasks),
                           epochs=20,callbacks=callbacks,steps_per_epoch=100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','valid'])