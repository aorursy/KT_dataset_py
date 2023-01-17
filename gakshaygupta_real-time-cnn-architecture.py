# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from multiprocessing import pool
import matplotlib.pyplot as plt
from tensorflow import keras 
import sklearn 
import scipy 
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
!pip install pyunpack
!pip install patool

from pyunpack import Archive
Archive("../data/competitions/facial-keypoints-detection/test.zip").extractall("../data/competitions/facial-keypoints-detection")
Archive("../data/competitions/facial-keypoints-detection/training.zip").extractall("../data/competitions/facial-keypoints-detection")
print(os.listdir("../data/competitions/facial-keypoints-detection"))
train_data=pd.read_csv("../data/competitions/facial-keypoints-detection/training.csv")
test_data=pd.read_csv("../data/competitions/facial-keypoints-detection/test.csv")
# Any results you write to the current directory are saved as output.

train_data.dropna(inplace=True)
train_data.info()
X_train=np.array(train_data.iloc[:,-1])
X_test=np.array(test_data.iloc[:,-1])
#for train set
X_train_d=np.zeros([len(X_train),96,96])
for i,j in enumerate(X_train):
    c=np.array([int(x) for x in j.strip().split()])
    
    X_train_d[i]=c.reshape([96,96]) 
X_train=X_train_d.copy()    
#for test set
X_test_d=np.zeros([len(X_test),96,96])
for i,j in enumerate(X_test):
    c=np.array([int(x) for x in j.strip().split()])
   
    X_test_d[i]=c.reshape([96,96]) 
X_test=X_test_d.copy()    
#For Output
Y_train=train_data.iloc[:,:-1]
Y_test=train_data.iloc[:,-1]

##test train split
from sklearn.cross_validation import train_test_split
a_train, a_test, b_train, b_test = train_test_split(X_train,Y_train, test_size=0.1, random_state=42)
def affine_transform(img,i_points,f_points):
    rows,cols = img.shape
    x=[i_points[2*i] for i in range(15)]
    y=[i_points[2*i-1] for i in range(1,15+1)]
    i_pt=list(zip(x,y))
    x=[f_points[2*i] for i in range(15)]
    y=[f_points[2*i-1] for i in range(1,15+1)]
    f_pt=list(zip(x,y))
    pts1 = np.float32(i_pt)
    pts2 = np.float32(f_pt)
    pts=np.array(i_pt)
    ones=np.ones([15,3])
    
    ones[:,:2]=pts
    pts=ones.T
   
    M = cv2.getAffineTransform(pts1[[0,1,14]],pts2[[0,1,14]])
    transform=np.matmul(M,pts)
    dst = cv2.warpAffine(img.copy(),M,(cols,rows))
    return dst,transform.T
def face_pt_plotter(img,pt):
    imgd=img.copy()
    x=[pt[2*i] for i in range(15)]
    y=[pt[2*i-1] for i in range(1,15+1)]
    pts=zip(x,y)
    for i in pts:
        imgd[int(i[1])][int(i[0])]=255
    return imgd    
#plotting the images
arr=[]
for i in range(len(b_train)):
    img,trans=affine_transform(a_train[400],b_train.values[400],b_train.values[i])
    arr.append([img,trans])

#plotting the images with transformation
fig,axis=plt.subplots(nrows=5,ncols=5,figsize=[10,10])
count=0

for i in range(5):
    for j in range(5):
        axis[i,j].imshow(face_pt_plotter(arr[count+100][0],arr[count+100][1].reshape(30)))
        count+=1
        
#print(Y_train.values[np.nonzero(Y_train.values[:,0]>float(68)),0])
print(np.nonzero(b_train.values[:,0]>71)[0].shape)
print(np.nonzero(b_train.values[:,0]<60)[0].shape)
median=np.median(b_train.values[:,0])
print(median)
var=np.var(b_train.values[:,0])
mean=np.mean(b_train.values[:,0])
print(var,mean)
plt.hist(b_train.values[:,0])
T_points=b_train.values[np.concatenate([np.nonzero(b_train.values[:,0]>mean+var),np.nonzero(b_train.values[:,0]<mean-var)],axis=1).reshape(-1),:]
print(T_points.shape)
#plotting image with T_points
arr=[]
for i in range(len(T_points)):
    img,trans=affine_transform(a_train[0],b_train.values[0],T_points[i])
    arr.append([img,trans])
fig,axis=plt.subplots(nrows=15,ncols=5,figsize=[50,50])
count=0
for i in range(19):
    for j in range(5):
        try:
         axis[i,j].imshow(face_pt_plotter(arr[count][0],arr[count][1].reshape(30)))
        except IndexError:
            pass
        count+=1
        
import random
random.seed(1)
##augmented data creation
A_train=[]
B_train=[]
for j in range(len(b_train)):
        for i in np.random.choice(83,20):
                img,transform=affine_transform(a_train[j],b_train.values[j],T_points[i])
                A_train.append(img)
                B_train.append(transform.reshape(30))
            
print(len(A_train),len(B_train))

a_new=np.concatenate([a_train,np.array(A_train)],axis=0)
print(a_new.shape)
b_new=np.concatenate([b_train,np.array(B_train)],axis=0)
print(b_new.shape)
## model
from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D,Dense
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(30)(x)

    model = Model(img_input, output)
    return model
#training the model
batch_size = 32
num_epochs = 100
input_shape = (96, 96, 1)
verbose = 1
num_classes = 30
patience = 50
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='mse',
              metrics=["mae",'accuracy'])
model.summary()

# callbacks

early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4), verbose=1)
model_checkpoint = ModelCheckpoint("{epoch}-{val_loss}.h5", 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint,early_stop, reduce_lr]
#model.fit(a_new.reshape(-1,96,96,1),b_new,batch_size=batch_size,epochs=num_epochs, verbose=1, callbacks=callbacks,validation_data=[a_test.reshape(-1,96,96,1),b_test])
prediction=model.predict(np.concatenate([X_test.reshape(-1,96,96,1),np.zeros([1,96,96,1])],axis=0))

#saving the predictions
import pickle
with open("predicitons.plk","wb") as a:
    pickle.dump(prediction,a)
#plotting the test images with predicted points
fig,axis=plt.subplots(nrows=15,ncols=5,figsize=[50,50])
count=0
for i in range(15):
    for j in range(5):
        try:
         axis[i,j].imshow(face_pt_plotter(arr[count][0],arr[count][1].reshape(30)))
        except IndexError:
            pass
        count+=1
        
plt.imshow(face_pt_plotter(X_test[-7],prediction[-7]),cmap="gray")
##creating the submission file 
idlook=pd.read_csv("../data/competitions/facial-keypoints-detection/IdLookupTable.csv")
temp=pd.DataFrame(prediction,columns=list(Y_train.columns))
rowid=np.arange(1,53491)
len(rowid)
temp.head()

features=[]
for index,row in idlook.iterrows():
    features.append(temp.iloc[[row["ImageId"]-1]][row["FeatureName"]].values)
features=np.concatenate(features)
d={"RowId":rowid,"Location":x}
submission=pd.DataFrame(d)
submission.to_csv("facial_keypoint_detection.csv",index=False)