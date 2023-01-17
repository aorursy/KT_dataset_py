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
imgdir = '/kaggle/input/the-oxfordiiit-pet-dataset/images/images'
anndir = '/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls'
##imglist = sorted(os.listdir(imgdir))
"""for img in imglist:
    if '.mat' in img:
        imglist.remove(img)"""
annlist = sorted(os.listdir(anndir))
imglist = [str(img[:-3]+'jpg') for img in annlist]
imglist[-1005]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import random
import cv2
randimg = random.sample(imglist,1)[0]
img=mpimg.imread('/kaggle/input/the-oxfordiiit-pet-dataset/images/images/'+randimg)
imgplot = plt.imshow(img)
plt.show()
import bs4 as bs
fn = '/kaggle/input/the-oxfordiiit-pet-dataset/annotations/annotations/xmls/'+randimg[:-3]+'xml'
with open(fn,'r') as f:
    fstr = f.read()
soup = bs.BeautifulSoup(fstr)
print('label',': ',soup.find('name').text)
xmin = int(soup.xmin.text)
ymin = int(soup.ymin.text)
xmax = int(soup.xmax.text)
ymax = int(soup.ymax.text)
img1 = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),2)
plt.imshow(img1)
plt.show()
from scipy import misc
from skimage.transform import resize

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def datagen(imgdir,anndir,imglist):
    imgarr = []
    c = 0
    for path in imglist:
        compath = imgdir+'/'+path
        img = mpimg.imread(compath)
        img = resize(img,(112,112,3))
        ##img_gs = rgb2gray(img)
        imgarr.append(img)
        c = c+1
        if c%1000==0:
            print(c)
    return np.array(imgarr)
f = 1
if f==0:
    imgarr = datagen(imgdir,anndir,imglist)
else:
    imgarr = np.load('/kaggle/input/pet-localization/datax.npy')
imgarr.shape
def ygen(anndir,annlist):
    yarr = []
    for i in range(len(annlist)):
        path = anndir+'/'+annlist[i]
        with open(path,'r') as f:
            fstr = f.read()
        soup = bs.BeautifulSoup(fstr)
        h = float(soup.height.text)
        w = float(soup.width.text)
        xmin = int(soup.xmin.text)/w
        ymin = int(soup.ymin.text)/h
        xmax = int(soup.xmax.text)/w
        ymax = int(soup.ymax.text)/h
        yarr.append(np.array([xmin,ymin,xmax,ymax]))
    return np.array(yarr)
yarr = ygen(anndir,annlist)
yarr.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(imgarr, yarr, test_size=0.05, random_state=12)
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,BatchNormalization,Dropout,Add,Input,Lambda,Concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import keras.backend as K
from keras.optimizers import Adam
from keras.applications import VGG16,Xception,ResNet50,MobileNetV2,InceptionV3,VGG19
from keras.utils import plot_model
from keras.losses import Huber
initializer = keras.initializers.RandomNormal(mean=0., stddev=1.)
xception = Xception(include_top=False, weights='imagenet', pooling='max')
resnet = ResNet50(include_top=False, weights='imagenet', pooling='max')
vgg = VGG16(include_top=False,weights='imagenet',pooling='max')
vgg19 = VGG19(include_top=False,weights='imagenet',pooling='max')
mnet = MobileNetV2(include_top=False,weights='imagenet',pooling='max')
inception = InceptionV3(include_top=False,weights='imagenet',pooling='max')
input_imgs  = Input((112,112,3))

##vgg = VGG16(include_top=False,weights='imagenet', input_tensor=input_imgs, pooling='max')
##vgg_last = vgg(input_imgs)
##vgg19_last = vgg19(input_imgs)
xnet_last = xception(input_imgs)
##res_last = resnet(input_imgs)
##mnet_last = mnet(input_imgs)
##inet_last = inception(input_imgs)

#flat_1 = Flatten(name='flat_3')(vgg_last)
dense_1 = Dense(256,activation='relu',name='dense_1')(xnet_last)
dense_1 = Dropout(0.5,name='drop_1')(dense_1)
dense_2 = Dense(128,activation='relu',name='dense_2')(dense_1)
dense_2 = Dropout(0.5,name='drop_2')(dense_2)
output = Dense(4,activation='relu',name='output')(dense_2)

model = Model(inputs=input_imgs,outputs=output,name='CNN Model')
##plot_model(model,  show_shapes=True)
for layer in model.layers:
	layer.trainable = False
model.get_layer('dense_1').trainable = True
model.get_layer('dense_2').trainable = True
model.get_layer('output').trainable = True
##adam = tf.keras.optimizers.Adam(lr=0.001)
adam = keras.optimizers.Adam(lr=0.0002)
model.compile(loss='mse',optimizer=adam,metrics=['mse'])
model.fit(X_train,y_train,validation_split=0.05,epochs=20,batch_size=64,verbose=1)
import random
def random_test(X,y,n):
    m = X.shape[0]
    rin = random.sample(list(range(m)),n)
    plt.figure(figsize=(20,10))
    for i,tid in enumerate(rin):
        img1 = np.copy(X[tid])
        pred_0 = model.predict(img1.reshape((1,112,112,3)))
        xmin = int(pred_0[0][0]*112)
        ymin = int(pred_0[0][1]*112)
        xmax = int(pred_0[0][2]*112)
        ymax = int(pred_0[0][3]*112)
        ytest = y[tid]*112
        ytest = ytest.astype(int)
        img1 = cv2.rectangle(img1,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        img1 = cv2.rectangle(img1,(ytest[0],ytest[1]),(ytest[2],ytest[3]),(0,0,255),2)
        plt.subplot(100+n*10+i+1)
        plt.imshow(img1)
random_test(X_test,y_test,6)
random_test(X_train,y_train,6)
model.save('vgg19_model.h5')
