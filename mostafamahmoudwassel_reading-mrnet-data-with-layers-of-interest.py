# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation,MaxPooling2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os
from keras.layers import Input

from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.optimizers import *
from keras.models import Model,Sequential
from keras.layers import *
from keras.activations import *
from keras.callbacks import *
import numpy as np
import pandas as pd 
from numpy import zeros, newaxis

labels_train=[]
labels_train_abnormal = pd.DataFrame.to_numpy(pd.read_csv("../input/mrnet-v1/MRNet-v1.0/train-abnormal.csv")) 
labels_train_acl = pd.DataFrame.to_numpy(pd.read_csv("../input/mrnet-v1/MRNet-v1.0/train-acl.csv")) 
labels_train_men = pd.DataFrame.to_numpy(pd.read_csv("../input/mrnet-v1/MRNet-v1.0/train-meniscus.csv")) 
labels_train.append(1)
labels_train.append(0)
labels_train.append(0)

for i in range (0,1129):
    labels_train.append(labels_train_abnormal[i,1])
    labels_train.append(labels_train_acl[i,1])
    labels_train.append(labels_train_men[i,1])
labels_train=np.array(labels_train).reshape(-1, 3)
labels_train

X_Train_ax = []
Y_Train_ax=[]
X_Train_cor=[]
Y_Train_cor=[]
X_Train_sag=[]
Y_Train_sag=[]

for patient_ID in range(1130):
    label=labels_train[patient_ID]
    if(patient_ID<10):
        pathd1='../input/mrnet-v1/MRNet-v1.0/train/' + 'axial/000'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/train/' + 'coronal/000'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/train/' + 'sagittal/000'+str(patient_ID)+'.npy'
    elif(patient_ID<100):
        pathd1='../input/mrnet-v1/MRNet-v1.0/train/' + 'axial/00'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/train/' + 'coronal/00'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/train/' + 'sagittal/00'+str(patient_ID)+'.npy'
    elif(patient_ID<1000):
        pathd1='../input/mrnet-v1/MRNet-v1.0/train/' + 'axial/0'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/train/' + 'coronal/0'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/train/' + 'sagittal/0'+str(patient_ID)+'.npy'
    else:
        pathd1='../input/mrnet-v1/MRNet-v1.0/train/' + 'axial/'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/train/' + 'coronal/'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/train/' + 'sagittal/'+str(patient_ID)+'.npy'

    d1Image = np.load(pathd1) # this part for reading 3 prespectives of each patient with multi-layers each
    d2Image = np.load(pathd2)
    d3Image = np.load(pathd3)
    startd1=int((d1Image.shape[0]/2)-8)  #this part to capture layers of interest 
    endd1=int((d1Image.shape[0]/2)+8)
    startd2=int((d2Image.shape[0]/2)-8)
    endd2=int((d2Image.shape[0]/2)+8)
    startd3=int((d3Image.shape[0]/2)-8)
    endd3=int((d3Image.shape[0]/2)+8)

    
    image_tensor=d1Image[startd1:endd1,:,:].reshape(256,256,16)
    X_Train_ax.append(image_tensor)
    Y_Train_ax.append(label)
    image_tensor2=d2Image[startd2:endd2,:,:].reshape(256,256,16)
    X_Train_cor.append(image_tensor2)
    Y_Train_cor.append(label)
    image_tensor3=d3Image[startd3:endd3,:,:].reshape(256,256,16)
    X_Train_sag.append(image_tensor3)
    Y_Train_sag.append(label)

    
print(np.asarray(X_Train_ax).shape)
print(np.asarray(Y_Train_ax).shape)
print(np.asarray(X_Train_cor).shape)
print(np.asarray(Y_Train_cor).shape)
print(np.asarray(X_Train_sag).shape)
print(np.asarray(Y_Train_sag).shape)
X_Train_ax = np.array(X_Train_ax)
Y_Train_ax = np.array(Y_Train_ax)
X_Train_cor = np.array(X_Train_cor)
Y_Train_cor = np.array(Y_Train_cor)
X_Train_sag = np.array(X_Train_sag)
Y_Train_sag = np.array(Y_Train_sag)

labels_test=[]
labels_test_abnormal = pd.DataFrame.to_numpy(pd.read_csv("../input/mrnet-v1/MRNet-v1.0/valid-abnormal.csv")) 
labels_test_acl = pd.DataFrame.to_numpy(pd.read_csv("../input/mrnet-v1/MRNet-v1.0/valid-acl.csv")) 
labels_test_men = pd.DataFrame.to_numpy(pd.read_csv("../input/mrnet-v1/MRNet-v1.0/valid-meniscus.csv")) 
labels_test.append(0)
labels_test.append(0)
labels_test.append(0)

for i in range (118):
    labels_test.append(labels_test_abnormal[i,1])
    labels_test.append(labels_test_acl[i,1])
    labels_test.append(labels_test_men[i,1])
labels_test=np.array(labels_test).reshape(-1, 3)
labels_test

from PIL import Image
X_Test_ax = []
Y_Test_ax=[]
X_Test_cor=[]
Y_Test_cor=[]
X_Test_sag=[]
Y_Test_sag=[]

for patient_ID in range(119):
    label=labels_test[patient_ID]
    patient_ID=1130+patient_ID
    if(patient_ID<10):
        pathd1='../input/mrnet-v1/MRNet-v1.0/valid/' + 'axial/000'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/valid/' + 'coronal/000'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/valid/' + 'sagittal/000'+str(patient_ID)+'.npy'
    elif(patient_ID<100):
        pathd1='../input/mrnet-v1/MRNet-v1.0/valid/' + 'axial/00'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/valid/' + 'coronal/00'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/valid/' + 'sagittal/00'+str(patient_ID)+'.npy'
    elif(patient_ID<1000):
        pathd1='../input/mrnet-v1/MRNet-v1.0/valid/' + 'axial/0'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/valid/' + 'coronal/0'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/valid/' + 'sagittal/0'+str(patient_ID)+'.npy'
    else:
        pathd1='../input/mrnet-v1/MRNet-v1.0/valid/' + 'axial/'+str(patient_ID)+'.npy'
        pathd2='../input/mrnet-v1/MRNet-v1.0/valid/' + 'coronal/'+str(patient_ID)+'.npy'
        pathd3='../input/mrnet-v1/MRNet-v1.0/valid/' + 'sagittal/'+str(patient_ID)+'.npy'

    d1Image = np.load(pathd1)
    d2Image = np.load(pathd2)
    d3Image = np.load(pathd3)
    startd1=int((d1Image.shape[0]/2)-8)
    endd1=int((d1Image.shape[0]/2)+8)
    startd2=int((d2Image.shape[0]/2)-8)
    endd2=int((d2Image.shape[0]/2)+8)
    startd3=int((d3Image.shape[0]/2)-8)
    endd3=int((d3Image.shape[0]/2)+8)

    
    image_tensor=d1Image[startd1:endd1,:,:].reshape(256,256,16)
    X_Test_ax.append(image_tensor)
    Y_Test_ax.append(label)
    image_tensor2=d2Image[startd2:endd2,:,:].reshape(256,256,16)
    X_Test_cor.append(image_tensor2)
    Y_Test_cor.append(label)
    image_tensor3=d3Image[startd3:endd3,:,:].reshape(256,256,16)
    X_Test_sag.append(image_tensor3)
    Y_Test_sag.append(label)

    
    
    
print(np.asarray(X_Test_ax).shape)
print(np.asarray(Y_Test_ax).shape)
print(np.asarray(X_Test_cor).shape)
print(np.asarray(Y_Test_cor).shape)
print(np.asarray(X_Test_sag).shape)
print(np.asarray(Y_Test_sag).shape)
X_Test_ax = np.array(X_Test_ax)
Y_Test_ax = np.array(Y_Test_ax)
X_Test_cor = np.array(X_Test_cor)
Y_Test_cor = np.array(Y_Test_cor)
X_Test_sag = np.array(X_Test_sag)
Y_Test_sag = np.array(Y_Test_sag)

