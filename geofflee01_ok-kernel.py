#import a bunch of rubbish



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow

import csv as csv

from IPython.display import display

import cv2 

from matplotlib import pyplot as plt

import operator

from keras.models import Model

from keras.layers import *

from keras.activations import relu

from keras.preprocessing.image import ImageDataGenerator

from keras import backend

import os

from random import shuffle 

    

#LOSS AND METRICS FOR LEARNING ALGORITHM

def gacc(y_true, y_pred): #general accuracy

    return 1-(backend.mean(backend.abs(y_pred - y_true), axis=-1))



def right_acc(y_true, y_pred): #general accuracy

    return 1-(backend.mean(backend.minimum(backend.abs((backend.round(y_pred*4) - y_true*4)*4),1), axis=-1))



def mean_squared_error2(y_true, y_pred): #custom loss algorithm, actually a modded MAE

    return backend.mean(backend.maximum((backend.abs(y_pred - y_true))-.02,0), axis=-1)



def std_error(y_true, y_pred): #std of the error

    return backend.std((backend.abs(y_pred - y_true)))



def p_meanX(y_true, y_pred): #mean of all predictions - should ideally be .5

    return backend.mean(y_pred, axis=-1)



def p_stdX(y_true, y_pred): #std of all predictions - should be ~0.32 to 0.35

    return backend.std(y_pred)



# Aaaaaaaaaaaaaaaaa
## Callable functions -- just shrink and hide this junk



def adjust_gamma(image, gamma=1.0): #old version of gamma correction

    i = 0

    invGamma = 1.0 / gamma

    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)



def adjust_gamma2(img, gamma=1.0): #new colour balancer, normalises colour balance

    new = [20,80,120]

    newimge = np.zeros(512*512*3)

    newimge = np.reshape(newimge,(512,512,3))

    newimge[:,:,0] = img[:,:,0]*(min(2,(new[0] / np.mean(img[128:384,:,0]))))

    newimge[:,:,1] = img[:,:,1]*(min(2,(new[1] / np.mean(img[128:384,:,1]))))

    newimge[:,:,2] = img[:,:,2]*(min(2,(new[2] / np.mean(img[128:384,:,2]))))

    newimge = np.clip(newimge,0,254)

    img = newimge.astype(np.uint8)

    return img



def autocrop(image, threshold=10): #is meant to get rid of the back borders (works most times)

    if len(image.shape) == 3:

        flatImage = np.max(image, 2)

    else:

        flatImage = image

    assert len(flatImage.shape) == 2



    rows = np.where(np.max(flatImage, 0) > threshold)[0]

    if rows.size:

        cols = np.where(np.max(flatImage, 1) > threshold)[0]

        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]

    else:

        image = image[:1, :1]

    return image



def im_resize(img): #crops to square, also calls autocrop,

    img = autocrop(img)

    height,width, x = img.shape

    margin = 0

    if(width>=height):

        margin = int(np.floor((width-height)/2))

        #print(height, width, margin)

        crop_img = img[0:height,(width-height-margin):(height+margin)]

    else:

        margin = int(np.floor((width-height)/2))

        #print(height, width, margin)

        crop_img = img[(height-width-margin):(width+margin),0:width]

    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)

    return img



#im_avg returns a tuple of the inner area of the cropped 512x512 image

def im_avg(img):

    return cv2.mean(img[100:412,100:412])
a = Input(shape=(512,512,3))

x = Conv2D(128, (3, 3), strides=2, activation='relu')(a)

x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)

x1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x)

x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x3 = Conv2D(128, (5, 5), activation='relu', padding='same')(x)

x3a = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(x)

x = concatenate([x,x1,x2,x3,x3a], axis=-1)

x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

x = Conv2D(128, (1, 1), activation='relu')(x)

x4 = Conv2D(64, (1, 1), activation='relu', padding='same')(x)

x5 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x6 = Conv2D(64, (5, 5), activation='relu', padding='same')(x)

x6a = MaxPooling2D(pool_size=(3, 3), strides=1,padding='same')(x)

x = concatenate([x,x4,x5,x6,x6a], axis=-1)

x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

x = Conv2D(64, (1, 1), activation='elu')(x)

x7 = Conv2D(64, (1, 1), activation='elu', padding='same')(x)

x8 = Conv2D(64, (3, 3), activation='elu', padding='same')(x)

x9 = Conv2D(64, (5, 5), activation='elu', padding='same')(x)

x9a = MaxPooling2D(pool_size=(3, 3), strides=1,padding='same')(x)

x = concatenate([x,x7,x8,x9,x9a], axis=-1)

x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

x = Conv2D(64, (1, 1), activation='elu')(x)

x10 = Conv2D(64, (1, 1), activation='elu', padding='same')(x)

x11 = Conv2D(64, (3, 3), activation='elu', padding='same')(x)

x12 = Conv2D(64, (5, 5), activation='elu', padding='same')(x)

x12a = MaxPooling2D(pool_size=(3, 3), strides=1,padding='same')(x)

x = concatenate([x,x11,x12,x10,x12a], axis=-1)

x = Conv2D(32, (1, 1), activation='elu')(x)

x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

x = AveragePooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)

x = Flatten()(x)

b = Dense(1,activation='linear')(x)

model = Model(inputs=a, outputs=b)

#Running a model

batch_size = 32

epochs = 300



datagen = ImageDataGenerator(

    horizontal_flip=True,

    vertical_flip=True

 #   brightness_range=(0.7,1.3),

 #   channel_shift_range=.3

)



model.compile(loss=mean_squared_error2, optimizer='adam', metrics=[right_acc, std_error,p_meanX,p_stdX ])

#datagen.fit(train_images)



#model.fit(x=train_images, y=train_labels, epochs=50, verbose =0,validation_split=0.1, shuffle=True) 
### Need to make a dataframe to do dataframe access of images and classications



# probably should have done this in the image gen script



for dirname, _, filenames in os.walk('../input/12kimages/data/data/0/'): pass

dirname="../input/12kimages/data/data/0/"

datalabels0 = pd.DataFrame()

datalabels0['name'] = filenames

datalabels0['name'] = '0/' + datalabels0['name'].astype(str)

datalabels0['diagnosis'] = 0



for dirname, _, filenames in os.walk('../input/12kimages/data/data/1/'): pass

dirname="../input/12kimages/data/data/1/"

datalabels1 = pd.DataFrame()

datalabels1['name'] = filenames

datalabels1['name'] = '1/' + datalabels1['name'].astype(str)

datalabels1['diagnosis'] = .25



for dirname, _, filenames in os.walk('../input/12kimages/data/data/2/'): pass

dirname="../input/12kimages/data/data/2/"

datalabels2 = pd.DataFrame()

datalabels2['name'] = filenames

datalabels2['name'] = '2/' + datalabels2['name'].astype(str)

datalabels2['diagnosis'] = .5



for dirname, _, filenames in os.walk('../input/12kimages/data/data/3/'): pass

dirname="../input/12kimages/data/data/3/"

datalabels3 = pd.DataFrame()

datalabels3['name'] = filenames

datalabels3['name'] = '3/' + datalabels3['name'].astype(str)

datalabels3['diagnosis'] = .75



for dirname, _, filenames in os.walk('../input/12kimages/data/data/4/'): pass

dirname="../input/12kimages/data/data/4/"

datalabels4 = pd.DataFrame()

datalabels4['name'] = filenames

datalabels4['name'] = '4/' + datalabels4['name'].astype(str)

datalabels4['diagnosis'] = 1



datalabels = pd.DataFrame().append([datalabels0, datalabels1, datalabels2, datalabels3, datalabels4],ignore_index = True)

datalabels['name'] = datalabels['name'].astype(str)

#datalabels
type(datalabels['name'][3])
the_flow = datagen.flow_from_dataframe(datalabels, directory='../input/12kimages/data/data',x_col='name',y_col='diagnosis',class_mode='raw',target_size=(512, 512), batch_size=32)

hist = model.fit_generator(the_flow,

                   use_multiprocessing=False, shuffle=True, workers=1,steps_per_epoch=100,epochs=epochs,verbose =2)
real_pred = abs(model.predict(train_images))*4

pred = np.round(real_pred)

real = np.round(abs(np.asarray(train_labels))*4)



for i in range(100):

    print(real[i], pred[i],real_pred[i])
for dirname, _, filenames in os.walk('/kaggle/input/aptos2019-blindness-detection/train_images'):

    pass

dirname="/kaggle/input/aptos2019-blindness-detection/train_images"

    

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv', header=0)

train['diagnosis_2'] = 0

train['diagnosis_3'] = 0

bad = [0,0,0,0,0]

#del train_images

#del train_labels

count = 0

accurat = 0

inaccurat = 0

#nimg = {}

for filename in filenames:

    if filename[(len(filename)-4):] == ".png":

        img = cv2.imread(os.path.join(dirname, filename))

        new_img = im_resize(img)

        thisav = im_avg(new_img)

        dgamma = thisav[0]+thisav[1]+thisav[2]

        new_img = adjust_gamma2(new_img,(((400) / dgamma)))

        #new_img = np.divide(new_img,255.0)

        count = count+1

        k = np.reshape(new_img,(1,512,512,3))

        

        real_pred = min(4,max(0,model.predict(k)[0][0]*4))

        pred = int(real_pred)  #np.round(real_pred)

        real = train[train.id_code==filename[:-4]].diagnosis.item()

        train.loc[train.id_code==filename[:-4],'diagnosis_2']=int(pred)

        #train.loc[train.id_code==filename[:-4],'diagnosis_3']=real_pred

        #print(real, " ", pred, " ",real_pred)

        if int(pred) == real:

            accurat +=1

        else:

            inaccurat +=1

            bad[real] += 1

        if count % 100 == 0: 

            print(count, " - Good: ",accurat, " n Bad: ",inaccurat, " - Bad: ",bad)

 

train
for dirname, _, filenames in os.walk('/kaggle/input/aptos2019-blindness-detection/test_images'):

    pass

dirname="/kaggle/input/aptos2019-blindness-detection/test_images"

    

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv', header=0)

test['diagnosis'] = 0



test_images = []

test_labels = []

count = 0

#nimg = {}

for filename in filenames:

    if filename[(len(filename)-4):] == ".png":

        img = cv2.imread(os.path.join(dirname, filename))

        new_img = im_resize(img)

        thisav = im_avg(new_img)

        dgamma = thisav[0]+thisav[1]+thisav[2]

        new_img = adjust_gamma2(new_img,(((400) / dgamma)))

        #new_img = np.divide(new_img,255.0)

        count = count+1

        #nimg[filename[:-4]] = new_img

        k = np.reshape(new_img,(1,512,512,3))

       # print(np.shape(k), np.shape(test_images))

        #if len(test_images) ==0:

        #    test_images = np.copy(k)

        #else:

        #    test_images=np.vstack([test_images,k])

        #test_labels.append(test[test.id_code==filename[:-4]].diagnosis.item())

        real_pred = min(4,max(0,model.predict(k)[0][0]*4))

        pred = int(real_pred)  #np.round(real_pred)

        test.loc[test.id_code==filename[:-4],'diagnosis']=int(pred)

       # print(pred, int(pred))

        

        if count % 50 == 0: 

            print(count)

     #   if count % 1000 == 0: 

            #break

#test_labels = np.asarray(test_labels)    

test.to_csv('submission.csv',index=False)

test

#pred = np.round(abs(model.predict(test_images))*4)

#print(pred)