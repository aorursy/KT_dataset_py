#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Import usual libraries

import cv2, os

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

from zipfile import ZipFile

from sklearn.utils import shuffle



import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

import keras, sys, time, warnings

from keras.models import *

from keras.layers import *

import pandas as pd

from keras import optimizers



warnings.filterwarnings("ignore")

sns.set_style("whitegrid", {'axes.grid' : False})



print("python {}".format(sys.version))

print("keras version {}".format(keras.__version__)); del keras

print("tensorflow version {}".format(tf.__version__))





dir_data = "/kaggle/input/seg-data/dataset1/dataset1"

dir_seg = dir_data + "/annotations_prepped_train/"

dir_img = dir_data + "/images_prepped_train/"



VGG_Weights_path = "/kaggle/input/seg-data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
def getImageArr( path , width , height ):

        img = cv2.imread(path, 1)

        img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1

        return img



def getSegmentationArr( path , nClasses ,  width , height  ):



    seg_labels = np.zeros((  height , width  , nClasses ))

    img = cv2.imread(path, 1)

    img = cv2.resize(img, ( width , height ))

    img = img[:, : , 0]



    for c in range(nClasses):

        seg_labels[: , : , c ] = (img == c ).astype(int)

    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))

    return seg_labels



input_height , input_width = 224 , 224

output_height , output_width = 224 , 224



ldseg = np.array(os.listdir(dir_seg))

  ## pick the first image file

fnm = ldseg[0]

print(fnm)



  ## read in the original image and segmentation labels

seg = cv2.imread(dir_seg + fnm ) # (360, 480, 3)

img_is = cv2.imread(dir_img + fnm )

print("seg.shape={}, img_is.shape={}".format(seg.shape,img_is.shape))



## Check the number of labels

mi, ma = np.min(seg), np.max(seg)

n_classes = ma - mi + 1

print("minimum seg = {}, maximum seg = {}, Total number of segmentation classes = {}".format(mi,ma, n_classes))



images = os.listdir(dir_img)

images.sort()

segmentations  = os.listdir(dir_seg)

segmentations.sort()

      

X = []

Y = []

for im , seg in zip(images,segmentations) :

    X.append( getImageArr(dir_img + im , input_width , input_height )  )

    Y.append( getSegmentationArr( dir_seg + seg , n_classes , output_width , output_height )  )



X, Y = np.array(X) , np.array(Y)

print(X.shape,Y.shape)

def FCN_VGG_8( nClasses ,  input_height=224, input_width=224):

    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,

    ## which makes the input_height and width 2^5 = 32 times smaller

    assert input_height%32 == 0

    assert input_width%32 == 0

    IMAGE_ORDERING =  "channels_last" 



    img_input = Input(shape=(input_height,input_width, 3)) ## Assume 224,224,3

    

    ## Block 1

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)

    f1 = x

    

    # Block 2

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)

    f2 = x



    # Block 3

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)

    pool3 = x



    # Block 4

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)

    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)## (None, 14, 14, 512) 



    # Block 5

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(pool4)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)

    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)## (None, 7, 7, 512)



    #x = Flatten(name='flatten')(x)

    #x = Dense(4096, activation='relu', name='fc1')(x)

    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)

    # assuming that the input_height = input_width = 224 as in VGG data

    

    #x = Dense(4096, activation='relu', name='fc2')(x)

    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)   

    # assuming that the input_height = input_width = 224 as in VGG data

    

    #x = Dense(1000 , activation='softmax', name='predictions')(x)

    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)

    # assuming that the input_height = input_width = 224 as in VGG data

    

    

    vgg  = Model(  img_input , pool5  )

    vgg.load_weights(VGG_Weights_path) ## loading VGG weights for the encoder parts of FCN8

    

    n = 4096

    o = ( Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)

    conv7 = ( Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    

    

    ## 4 times upsamping for pool4 layer

    conv7_4 = Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(conv7)

    ## (None, 224, 224, 10)

    ## 2 times upsampling for pool411

    pool411 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)

    S1 = Add(name="add1")([pool411, conv7_4 ])



    pool411_2 = (Conv2DTranspose( nClasses , kernel_size=(2,2) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING ))(S1)

    

    pool311 = ( Conv2D( nClasses , ( 1 , 1 ) , activation='relu' , padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)

        

    

    o = Add(name="add2")([pool411_2, pool311])



    o = Conv2DTranspose( nClasses , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False, data_format=IMAGE_ORDERING )(o)

    o = (Activation('softmax'))(o)

    

    model = Model(img_input, o)



    return model



model = FCN_VGG_8(nClasses     = n_classes,  

             input_height = 224, 

             input_width  = 224)

model.summary()
train_rate = 0.85

index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)

index_test  = list(set(range(X.shape[0])) - set(index_train))

                            

X, Y = shuffle(X,Y)

X_train, y_train = X[index_train],Y[index_train]

X_test, y_test = X[index_test],Y[index_test]

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)



sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])



hist1 = model.fit(X_train,y_train,

                  validation_data=(X_test,y_test),

                  batch_size=32,epochs=200,verbose=2)



model.save("fcn_vgg_200.h5")