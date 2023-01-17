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


from PIL import Image
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

    d1Image = np.load(pathd1)
    d2Image = np.load(pathd2)
    d3Image = np.load(pathd3)
    startd1=int((d1Image.shape[0]/2)-4)
    endd1=int((d1Image.shape[0]/2)+4)
    startd2=int((d2Image.shape[0]/2)-4)
    endd2=int((d2Image.shape[0]/2)+4)
    startd3=int((d3Image.shape[0]/2)-4)
    endd3=int((d3Image.shape[0]/2)+4)

    
    image_tensor=d1Image[startd1:endd1,:,:].reshape(256,256,8)
    X_Train_ax.append(image_tensor)
    Y_Train_ax.append(label)
    image_tensor2=d2Image[startd2:endd2,:,:].reshape(256,256,8)
    X_Train_cor.append(image_tensor2)
    Y_Train_cor.append(label)
    image_tensor3=d3Image[startd3:endd3,:,:].reshape(256,256,8)
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
    startd1=int((d1Image.shape[0]/2)-4)
    endd1=int((d1Image.shape[0]/2)+4)
    startd2=int((d2Image.shape[0]/2)-4)
    endd2=int((d2Image.shape[0]/2)+4)
    startd3=int((d3Image.shape[0]/2)-4)
    endd3=int((d3Image.shape[0]/2)+4)

    
    image_tensor=d1Image[startd1:endd1,:,:].reshape(256,256,8)
    X_Test_ax.append(image_tensor)
    Y_Test_ax.append(label)
    image_tensor2=d2Image[startd2:endd2,:,:].reshape(256,256,8)
    X_Test_cor.append(image_tensor2)
    Y_Test_cor.append(label)
    image_tensor3=d3Image[startd3:endd3,:,:].reshape(256,256,8)
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

Y_Test_ax_ab=Y_Test_ax[:,0]
Y_Train_ax_ab=Y_Train_ax[:,0]
Y_Test_ax_acl=Y_Test_ax[:,1]
Y_Train_ax_acl=Y_Train_ax[:,1]
Y_Test_ax_men=Y_Test_ax[:,2]
Y_Train_ax_men=Y_Train_ax[:,2]

Y_Test_cor_ab=Y_Test_cor[:,0]
Y_Train_cor_ab=Y_Train_cor[:,0]
Y_Test_cor_acl=Y_Test_cor[:,1]
Y_Train_cor_acl=Y_Train_cor[:,1]
Y_Test_cor_men=Y_Test_cor[:,2]
Y_Train_cor_men=Y_Train_cor[:,2]

Y_Test_sag_ab=Y_Test_sag[:,0]
Y_Train_sag_ab=Y_Train_sag[:,0]
Y_Test_sag_acl=Y_Test_sag[:,1]
Y_Train_sag_acl=Y_Train_sag[:,1]
Y_Test_sag_men=Y_Test_sag[:,2]
Y_Train_sag_men=Y_Train_sag[:,2]

InputShape=(256,256,8)
NumClasses=2
def VGG():
    modelVGG = Sequential()
    modelVGG.add(Conv2D(input_shape=InputShape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    modelVGG.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    modelVGG.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    modelVGG.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    modelVGG.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    modelVGG.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    modelVGG.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    modelVGG.add(Flatten())
    modelVGG.add(Dense(units=4096,activation="relu"))
    modelVGG.add(Dense(units=4096,activation="relu"))
    modelVGG.add(Dense(units=1, activation="sigmoid"))
    return modelVGG
modelVGG1=VGG()
modelVGG1.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG1.fit(X_Train_ax, Y_Train_ax_ab, epochs=10, batch_size=5)
print(modelVGG1.evaluate(X_Test_ax, Y_Test_ax_ab, verbose=2))
modelVGG1.save_weights('modelVGG1.h5')

modelVGG2=VGG()
modelVGG2.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG2.fit(X_Train_ax, Y_Train_ax_acl, epochs=10, batch_size=5)
print(modelVGG2.evaluate(X_Test_ax, Y_Test_ax_acl, verbose=2))
modelVGG2.save_weights('modelVGG2.h5')

modelVGG3=VGG()
modelVGG3.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG3.fit(X_Train_ax, Y_Train_ax_men, epochs=10, batch_size=5)
print(modelVGG3.evaluate(X_Test_ax, Y_Test_ax_men, verbose=2))
modelVGG3.save_weights('modelVGG3.h5')

modelVGG4=VGG()
modelVGG4.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG4.fit(X_Train_cor, Y_Train_cor_ab, epochs=10, batch_size=5)
print(modelVGG4.evaluate(X_Test_cor, Y_Test_cor_ab, verbose=2))
modelVGG4.save_weights('modelVGG4.h5')

modelVGG5=VGG()
modelVGG5.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG5.fit(X_Train_cor, Y_Train_cor_acl, epochs=10, batch_size=5)
print(modelVGG5.evaluate(X_Test_cor, Y_Test_cor_acl, verbose=2))
modelVGG5.save_weights('modelVGG5.h5')

modelVGG6=VGG()
modelVGG6.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG6.fit(X_Train_cor, Y_Train_cor_men, epochs=10, batch_size=5)
print(modelVGG6.evaluate(X_Test_cor, Y_Test_cor_men, verbose=2))
modelVGG6.save_weights('modelVGG6.h5')

modelVGG7=VGG()
modelVGG7.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG7.fit(X_Train_sag, Y_Train_sag_ab, epochs=10, batch_size=5)
print(modelVGG7.evaluate(X_Test_sag, Y_Test_sag_ab, verbose=2))
modelVGG7.save_weights('modelVGG7.h5')

modelVGG8=VGG()
modelVGG8.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG8.fit(X_Train_sag, Y_Train_sag_acl, epochs=10, batch_size=5)
print(modelVGG8.evaluate(X_Test_sag, Y_Test_sag_acl, verbose=2))
modelVGG8.save_weights('modelVGG8.h5')

modelVGG9=VGG()
modelVGG9.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelVGG9.fit(X_Train_sag, Y_Train_sag_men, epochs=10, batch_size=5)
print(modelVGG9.evaluate(X_Test_sag, Y_Test_sag_men, verbose=2))
modelVGG9.save_weights('modelVGG9.h5')

def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
    channels = int(channels * alpha)

    x = Conv2D(channels,
               kernel_size=(3, 3),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_act'.format(name))(x)
    return x


def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn1'.format(name))(x)
    x = Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = BatchNormalization(name='{}_bn2'.format(name))(x)
    x = Activation('relu', name='{}_act2'.format(name))(x)
    return x


def MobileNet(shape, num_classes, alpha=1.0, include_top=True, weights=None):
    x_in = Input(shape=shape)

    x = get_conv_block(x_in, 32, (2, 2), alpha=alpha, name='initial')

    layers = [
        (64, (1, 1)),
        (128, (2, 2)),
        (128, (1, 1)),
        (256, (2, 2)),
        (256, (1, 1)),
        (512, (2, 2)),
        *[(512, (1, 1)) for _ in range(5)],
        (1024, (2, 2)),
        (1024, (2, 2))
    ]

    for i, (channels, strides) in enumerate(layers):
        x = get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))

    if include_top:
        x = GlobalAvgPool2D(name='global_avg')(x)
        x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

modelMobileNet1=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet1.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet1.fit(X_Train_ax, Y_Train_ax_ab, epochs=10, batch_size=5)
print(modelMobileNet1.evaluate(X_Test_ax, Y_Test_ax_ab, verbose=2))
modelMobileNet1.save_weights('modelMobileNet1.h5')

modelMobileNet2=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet2.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet2.fit(X_Train_ax, Y_Train_ax_acl, epochs=10, batch_size=5)
print(modelMobileNet2.evaluate(X_Test_ax, Y_Test_ax_acl, verbose=2))
modelMobileNet2.save_weights('modelMobileNet2.h5')

modelMobileNet3=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet3.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet3.fit(X_Train_ax, Y_Train_ax_men, epochs=10, batch_size=5)
print(modelMobileNet3.evaluate(X_Test_ax, Y_Test_ax_men, verbose=2))
modelMobileNet3.save_weights('modelMobileNet3.h5')

modelMobileNet4=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet4.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet4.fit(X_Train_cor, Y_Train_cor_ab, epochs=10, batch_size=5)
print(modelMobileNet4.evaluate(X_Test_cor, Y_Test_cor_ab, verbose=2))
modelMobileNet4.save_weights('modelMobileNet4.h5')

modelMobileNet5=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet5.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet5.fit(X_Train_cor, Y_Train_cor_acl, epochs=10, batch_size=5)
print(modelMobileNet5.evaluate(X_Test_cor, Y_Test_cor_acl, verbose=2))
modelMobileNet5.save_weights('modelMobileNet5.h5')

modelMobileNet6=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet6.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet6.fit(X_Train_cor, Y_Train_cor_men, epochs=10, batch_size=5)
print(modelMobileNet6.evaluate(X_Test_cor, Y_Test_cor_men, verbose=2))
modelMobileNet6.save_weights('modelMobileNet6.h5')

modelMobileNet7=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet7.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet7.fit(X_Train_sag, Y_Train_sag_ab, epochs=10, batch_size=5)
print(modelMobileNet7.evaluate(X_Test_sag, Y_Test_sag_ab, verbose=2))
modelMobileNet7.save_weights('modelMobileNet7.h5')

modelMobileNet8=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet8.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])

modelMobileNet8.fit(X_Train_sag, Y_Train_sag_acl, epochs=10, batch_size=5)
print(modelMobileNet8.evaluate(X_Test_sag, Y_Test_sag_acl, verbose=2))
modelMobileNet8.save_weights('modelMobileNet8.h5')

modelMobileNet9=MobileNet(shape=InputShape,num_classes=NumClasses)
modelMobileNet9.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelMobileNet9.fit(X_Train_sag, Y_Train_sag_men, epochs=10, batch_size=5)
print(modelMobileNet9.evaluate(X_Test_sag, Y_Test_sag_men, verbose=2))
modelMobileNet9.save_weights('modelMobileNet9.h5')

import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os
import gc

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, recall_score, precision_score


from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils, to_categorical
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import keras.backend as K

#identity_block

def identity_block(X, f, filters, stage, block):

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a',padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (256, 256, 8)):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='b')
    X = identity_block(X, 3, [32, 32, 128], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=3, block='d')

    # Stage 4 
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='d')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='e')
    X = identity_block(X, 3, [128, 128, 512], stage=4, block='f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [256,256, 1024], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256,256, 1024], stage=5, block='b')
    X = identity_block(X, 3, [256,256, 1024], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), name='avg_pool')(X)
    

    # output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

modelRN1 = ResNet50(input_shape = (256, 256, 8))
modelRN1.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN1.fit(X_Train_ax, Y_Train_ax_ab, epochs=10, batch_size=5)
print(modelRN1.evaluate(X_Test_ax, Y_Test_ax_ab, verbose=2))
modelRN1.save_weights('modelRN1.h5')

modelRN2 = ResNet50(input_shape = (256, 256, 8))
modelRN2.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN2.fit(X_Train_ax, Y_Train_ax_acl, epochs=10, batch_size=5)
print(modelRN2.evaluate(X_Test_ax, Y_Test_ax_acl, verbose=2))
modelRN2.save_weights('modelRN2.h5')

modelRN3 = ResNet50(input_shape = (256, 256, 8))
modelRN3.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN3.fit(X_Train_ax, Y_Train_ax_men, epochs=10, batch_size=5)
print(modelRN3.evaluate(X_Test_ax, Y_Test_ax_men, verbose=2))
modelRN3.save_weights('modelRN3.h5')

modelRN4 = ResNet50(input_shape = (256, 256, 8))
modelRN4.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN4.fit(X_Train_cor, Y_Train_cor_ab, epochs=10, batch_size=5)
print(modelRN4.evaluate(X_Test_cor, Y_Test_cor_ab, verbose=2))
modelRN4.save_weights('modelRN4.h5')

modelRN5 = ResNet50(input_shape = (256, 256, 8))
modelRN5.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN5.fit(X_Train_cor, Y_Train_cor_acl, epochs=10, batch_size=5)
print(modelRN5.evaluate(X_Test_cor, Y_Test_cor_acl, verbose=2))
modelRN5.save_weights('modelRN5.h5')

modelRN6 = ResNet50(input_shape = (256, 256, 8))
modelRN6.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN6.fit(X_Train_cor, Y_Train_cor_men, epochs=10, batch_size=5)
print(modelRN6.evaluate(X_Test_cor, Y_Test_cor_men, verbose=2))
modelRN6.save_weights('modelRN6.h5')

modelRN7 = ResNet50(input_shape = (256, 256, 8))
modelRN7.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN7.fit(X_Train_sag, Y_Train_sag_ab, epochs=10, batch_size=5)
print(modelRN7.evaluate(X_Test_sag, Y_Test_sag_ab, verbose=2))
modelRN7.save_weights('modelRN7.h5')

modelRN8 = ResNet50(input_shape = (256, 256, 8))
modelRN8.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN8.fit(X_Train_sag, Y_Train_sag_acl, epochs=10, batch_size=5)
print(modelRN8.evaluate(X_Test_sag, Y_Test_sag_acl, verbose=2))
modelRN8.save_weights('modelRN8.h5')

modelRN9 = ResNet50(input_shape = (256, 256, 8))
modelRN9.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.002, momentum=0.9),metrics=["accuracy"])
modelRN9.fit(X_Train_sag, Y_Train_sag_men, epochs=10, batch_size=5)
print(modelRN9.evaluate(X_Test_sag, Y_Test_sag_men, verbose=2))
modelRN9.save_weights('modelRN9.h5')

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



from PIL import Image
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

    d1Image = np.load(pathd1)
    d2Image = np.load(pathd2)
    d3Image = np.load(pathd3)
    startd1=int((d1Image.shape[0]/2)-1)
    endd1=int((d1Image.shape[0]/2)+2)
    startd2=int((d2Image.shape[0]/2)-1)
    endd2=int((d2Image.shape[0]/2)+2)
    startd3=int((d3Image.shape[0]/2)-1)
    endd3=int((d3Image.shape[0]/2)+2)

    
    image_tensor=d1Image[startd1:endd1,:,:].reshape(256,256,3)
    X_Train_ax.append(image_tensor)
    Y_Train_ax.append(label)
    image_tensor2=d2Image[startd2:endd2,:,:].reshape(256,256,3)
    X_Train_cor.append(image_tensor2)
    Y_Train_cor.append(label)
    image_tensor3=d3Image[startd3:endd3,:,:].reshape(256,256,3)
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
    startd1=int((d1Image.shape[0]/2)-1)
    endd1=int((d1Image.shape[0]/2)+2)
    startd2=int((d2Image.shape[0]/2)-1)
    endd2=int((d2Image.shape[0]/2)+2)
    startd3=int((d3Image.shape[0]/2)-1)
    endd3=int((d3Image.shape[0]/2)+2)

    
    image_tensor=d1Image[startd1:endd1,:,:].reshape(256,256,3)
    X_Test_ax.append(image_tensor)
    Y_Test_ax.append(label)
    image_tensor2=d2Image[startd2:endd2,:,:].reshape(256,256,3)
    X_Test_cor.append(image_tensor2)
    Y_Test_cor.append(label)
    image_tensor3=d3Image[startd3:endd3,:,:].reshape(256,256,3)
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


Y_Test_ax_ab=Y_Test_ax[:,0]
Y_Train_ax_ab=Y_Train_ax[:,0]
Y_Test_ax_acl=Y_Test_ax[:,1]
Y_Train_ax_acl=Y_Train_ax[:,1]
Y_Test_ax_men=Y_Test_ax[:,2]
Y_Train_ax_men=Y_Train_ax[:,2]

Y_Test_cor_ab=Y_Test_cor[:,0]
Y_Train_cor_ab=Y_Train_cor[:,0]
Y_Test_cor_acl=Y_Test_cor[:,1]
Y_Train_cor_acl=Y_Train_cor[:,1]
Y_Test_cor_men=Y_Test_cor[:,2]
Y_Train_cor_men=Y_Train_cor[:,2]

Y_Test_sag_ab=Y_Test_sag[:,0]
Y_Train_sag_ab=Y_Train_sag[:,0]
Y_Test_sag_acl=Y_Test_sag[:,1]
Y_Train_sag_acl=Y_Train_sag[:,1]
Y_Test_sag_men=Y_Test_sag[:,2]
Y_Train_sag_men=Y_Train_sag[:,2]
import tensorflow as tf
pretrained_model = tf.keras.applications.ResNet50(input_shape=(256,256,3), include_top=False)
pretrained_model.trainable = False

modelTL = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
    
])



for layer in modelTL.layers[:4]:
    layer.trainable=False
for layer in modelTL.layers[4:]:
    layer.trainable=True



modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Nadam',metrics=["accuracy"])
modelTL.fit(X_Train_sag, Y_Train_sag_men, epochs=10, batch_size=32)
print(modelTL.evaluate(X_Test_sag, Y_Test_sag_men, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Nadam',metrics=["accuracy"])
modelTL.fit(X_Train_sag, Y_Train_sag_acl, epochs=10, batch_size=32)
print(modelTL.evaluate(X_Test_sag, Y_Test_sag_acl, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Nadam',metrics=["accuracy"])
modelTL.fit(X_Train_sag, Y_Train_sag_ab, epochs=10, batch_size=10)
print(modelTL.evaluate(X_Test_sag, Y_Test_sag_ab, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Adamax',metrics=["accuracy"]) #more epochs
modelTL.fit(X_Train_cor, Y_Train_cor_men, epochs=50, batch_size=32)
print(modelTL.evaluate(X_Test_cor, Y_Test_cor_men, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Adamax',metrics=["accuracy"]) #more epochs
modelTL.fit(X_Train_cor, Y_Train_cor_acl, epochs=10, batch_size=32)
print(modelTL.evaluate(X_Test_cor, Y_Test_cor_acl, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Adamax',metrics=["accuracy"]) #more epochs
modelTL.fit(X_Train_cor, Y_Train_cor_ab, epochs=10, batch_size=32)
print(modelTL.evaluate(X_Test_cor, Y_Test_cor_ab, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Adamax',metrics=["accuracy"]) #more epochs
modelTL.fit(X_Train_ax, Y_Train_ax_men, epochs=50, batch_size=32)
print(modelTL.evaluate(X_Test_ax, Y_Test_ax_men, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Adamax',metrics=["accuracy"]) #more epochs
modelTL.fit(X_Train_ax, Y_Train_ax_acl, epochs=10, batch_size=32)
print(modelTL.evaluate(X_Test_ax, Y_Test_ax_acl, verbose=2))

modelTL.compile(loss=keras.losses.binary_crossentropy,
              optimizer='Adamax',metrics=["accuracy"]) #more epochs
modelTL.fit(X_Train_ax, Y_Train_ax_ab, epochs=10, batch_size=32)
print(modelTL.evaluate(X_Test_ax, Y_Test_ax_ab, verbose=2))

