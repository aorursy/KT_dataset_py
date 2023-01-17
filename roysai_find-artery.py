import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize

from scipy.interpolate import griddata
import numpy as np
import csv
import pandas as pd

import glob

import time

from keras. models import Model
from keras.layers import Input, BatchNormalization, Add, Concatenate, UpSampling3D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, ZeroPadding3D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import backend as K

import tensorflow as tf
def parse_data(img_data, mask):
    """
    Points:0 is X axis
    Points:1 is Y axis
    Points:2 is Z axis
    """
    #to normalize Points:0, Points:1, Points:2
    unique_0=list(np.unique(img_data['Points:0']))
    unique_1=list(np.unique(img_data['Points:1']))
    unique_2=list(np.unique(img_data['Points:2']))
    
    dims=(len(unique_0), len(unique_1), len(unique_2))
    print (dims)
    
    array=np.zeros(dims)
    label=np.zeros(dims)
    
    for i in range (len(img_data)):
        point0=unique_0.index(img_data.iloc[i]['Points:0'])
        point1=unique_1.index(img_data.iloc[i]['Points:1'])
        point2=unique_2.index(img_data.iloc[i]['Points:2'])
        #print (point0, point1, point2)
        array[point0, point1, point2]=img_data.iloc[i]['ImageScalars']
        label[point0, point1, point2]=mask.iloc[i]['ImageScalars']
    
    return array, label

def load_vti(file_name, steps=5):
    
    """
    Load csv data. Convert the data from vti format to numpy format.
    """
    img_data=pd.read_csv(file_name+'.csv')
    mask=pd.read_csv(file_name+'Level.csv')
    
    #normalize ImageScalar to be between 0 and 1
    img_data.ImageScalars=img_data.ImageScalars/(max(img_data.ImageScalars))

    dims=(len(unique_0), len(unique_1), len(unique_2))
    array=np.zeros(dims)
    
    print (img_data)
    #only take slice between -5 and 5, ~middle of the volume, to simplify testing
    #mask=mask[(img_data['Points:2']>-5) & (img_data['Points:2']<5)]
    img_data=img_data[(img_data['Points:2']>-5) & (img_data['Points:2']<5)]
    
    return parse_data(img_data, mask)
#To load csv data to numpy. Overall will take about 2 hours for the process. Only run this part if 
#need to generate new npy file.

"""

path='../input/'
allfile=glob.glob(path+"*img.csv")
for file in allfile:
    start=time.time()
    img_data=pd.read_csv(file)
    #img_data.head()

    mask=pd.read_csv(file[:-4]+'Level.csv')
    
    mask['ImageScalars'][mask['ImageScalars']>0]=0
    mask['ImageScalars'][mask['ImageScalars']<0]=1
    img, label=parse_data(img_data, mask)
    name=file[9:-4]
    np.save(name+".npy", img)
    np.save(name+"Level.npy", label)
    
    print ("{} save completed, took {} seconds".format(file, time.time()-start))
    
"""
""
#load pre-saved array.
path="../input/numpy-data/"
allnpyfile=glob.glob(path+'*img.npy')
DIMS=[48,64,96] #user define size. Consideration is taken to make maxpooling convenient
m=len(allnpyfile) #number of training and validation files
path="../input/numpy-data/"
allnpyfile=glob.glob(path+'*img.npy')
DIMS=[48,64,96] #user define size. Consideration is taken to make maxpooling convenient
m=len(allnpyfile) #number of training and validation files

img_array=np.zeros([m]+DIMS)
label_array=np.zeros([m]+DIMS)

img_array_flip=np.zeros([m]+DIMS)
label_array_flip=np.zeros([m]+DIMS)

ori_shape=[0]*m
for i, file in enumerate(allnpyfile):
    #load and resize image file
    temp1=np.load(file)
    ori_shape[i]=temp1.shape

    img_array[i]=resize(temp1, DIMS, preserve_range=True)
    
    #load and resize label file
    name=file[:-4]+'Level.npy'
    temp2=np.load(name)
    label_array[i]=resize(temp2, DIMS, preserve_range=True)
        
#sanity check
i=10
plt.subplot(221)
sample=np.load(allnpyfile[i])
middle=int(sample.shape[0]/2)
plt.imshow(sample[middle,:,:])

plt.subplot(222)
plt.imshow(img_array[i, 25, :,:])

plt.subplot(223)
sample=np.load(allnpyfile[i][:-4]+'Level.npy')
plt.imshow(sample[middle,:,:])

plt.subplot(224)
plt.imshow(label_array[i, 25, :, :])

plt.show()

#check heat map
heatmap=np.mean(label_array, axis=0)
plt.imshow(heatmap[20,:,:])
plt.show()
img_array=np.expand_dims(img_array, axis=-1)
label_array=np.expand_dims(label_array, axis=-1)

print (img_array.shape, label_array.shape)
img_dims=img_array.shape[1:]
def unet():
    n=8
    inputs=Input(img_dims) #48x64x96x1
    
    c1=Conv3D(n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1=Conv3D(n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1=BatchNormalization(axis=-1)(c1)
    p1=MaxPooling3D()(c1) #24x32x48x8
    
    c2=Conv3D(2*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2=Conv3D(2*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2=BatchNormalization(axis=-1)(c2)
    p2=MaxPooling3D()(c2) #12x16x24x16
    
    c3=Conv3D(4*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3=Conv3D(4*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3=BatchNormalization(axis=-1)(c3)
    p3=MaxPooling3D()(c3) #6x8x12x32
    
    c4=Conv3D(8*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4=Conv3D(8*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4=BatchNormalization(axis=-1)(c4)
    p4=MaxPooling3D()(c4) #3x4x6x64
    
    c5=Conv3D(16*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(p4) #3x4x6x128
    c5=Conv3D(16*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5=BatchNormalization(axis=-1)(c5)
    c5=Dropout(0.3)(c5)
       
    u6=Conv3DTranspose(8*n, kernel_size=2, strides=2, padding='same')(c5) #6x8x12x64
    u6=concatenate([u6, c4 ], axis=-1)
    u6=Conv3D(8*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    u6=Conv3D(8*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    u6=BatchNormalization(axis=-1)(u6)
    
    u7=Conv3DTranspose(4*n, kernel_size=2, strides=2, padding='same')(u6) #12x16x24x32
    u7=concatenate([u7, c3], axis=-1)
    u7=Conv3D(4*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    u7=Conv3D(4*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    u7=BatchNormalization(axis=-1)(u7)
    
    u8=Conv3DTranspose(2*n, kernel_size=2, strides=2, padding='same')(u7)
    u8=concatenate([u8, c2])
    u8=Conv3D(2*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    u8=Conv3D(2*n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    u8=BatchNormalization(axis=-1)(u8)
    
    u9=Conv3DTranspose(n, kernel_size=2, strides=2, padding='same')(u8)
    u9=concatenate([u9, c1]) 
    u9=Conv3D(n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    u9=Conv3D(n, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    u9=BatchNormalization(axis=-1)(u9)

    outputs=Conv3D(1, 1, activation='sigmoid')(u9)

    model=Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    return model

    
def mean_iou(y_true, y_pred):
    #intersection over union
    y_pred_=tf.to_int32(y_pred>0.5)
    score, up_opt=tf.metrics.mean_iou(y_true, y_pred_, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score=tf.identity(score)   
    
    return score
    
#use the first 16 cases for training, the remaining for cross-validation
train_X=img_array[:16]
train_Y=label_array[:16]

validate_X=img_array[16:]
validate_Y=label_array[16:]
"""
Skip this part if load from pre-trained model
"""
earlystopper=EarlyStopping(patience=40, verbose=1)
reduce_lr=ReduceLROnPlateau(factor=0.1, patience=4, min_lr=1e-8, verbose=1)

model=unet()
#First train with adam optimizer. then fine tune with sgd optimizer. using sgd always yield better result, but much slower compared to adam.
#We cut short the sgd learning by doing adam for head start.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.fit(train_X, train_Y, epochs=500, validation_data=(validate_X, validate_Y))

sgd = optimizers.SGD(lr=1e-3, decay=1e-2, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[mean_iou])
model.fit(train_X, train_Y, epochs=10, validation_data=(validate_X, validate_Y), callbacks=[earlystopper, reduce_lr])
model.save_weights('find_artery_unet')

"""
The author has trained the model for few hundred of epochs.
Uncomment to load author's pre-trained model instead.
"""
""
model=unet()
model.load_weights('../input/find-artery-weights/find_artery_unet')
def iou(y_true, y_pred, threshold):
    y_pred_=np.zeros(y_pred.shape)
    y_pred_[y_pred>threshold]=1
    
    intersection=np.sum(y_true*y_pred_)
    union=np.sum(y_true)+np.sum(y_pred_)-intersection
    
    return intersection/union
#sanity check
pred=model.predict(validate_X)

i=1 
n=25#slicing index n from image i
plt.subplot(121)
sample=np.squeeze(pred[i][n,:,:])
plt.title('prediction')
plt.imshow(sample)

plt.subplot(122)
sample=np.squeeze(validate_Y[i][n,:,:])
plt.title('ground truth')
plt.imshow(sample)

plt.show()
#find average iou among the validation case
pred_iou=[]
threshold=0.5
for i in range (0,4):
    pred_iou.append(iou(validate_Y[i], pred[i], threshold))
    
print (pred_iou)
print (np.mean(pred_iou))
#Use the following function to return the prediction to original size
def return_to_ori(pred, ori_shape):
    """
    restore the image to original shape.
    """
    return resize(pred, ori_size, preserve_range=True)
    
        

