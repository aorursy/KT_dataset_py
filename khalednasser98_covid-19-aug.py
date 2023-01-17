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
import cv2 
import matplotlib.pyplot as plt


# khaled edits
from imgaug import augmenters as iaa
metaData = pd.read_csv("../input/coronahack-chest-xraydataset/Chest_xray_Corona_Metadata.csv")
metaData
normalMetaData=metaData.loc[metaData['Label'] == 'Normal']
normalMetaData
PnemoniaMetaData=metaData.loc[metaData['Label'] == 'Pnemonia']
PnemoniaMetaData
VirusPnemoniaMetaData=PnemoniaMetaData.loc[PnemoniaMetaData['Label_1_Virus_category'] == 'Virus']
VirusPnemoniaMetaData
BacteriaPnemoniaMetaData=PnemoniaMetaData.loc[PnemoniaMetaData['Label_1_Virus_category'] == 'bacteria']
BacteriaPnemoniaMetaData
normalMetaDataTrain=normalMetaData.loc[normalMetaData['Dataset_type'] == 'TRAIN']
normalMetaDataTrain
normalMetaDataTest=normalMetaData.loc[normalMetaData['Dataset_type'] == 'TEST']
normalMetaDataTest
PnemoniaMetaDataTrain=PnemoniaMetaData.loc[PnemoniaMetaData['Dataset_type'] == 'TRAIN']
PnemoniaMetaDataTrain
PnemoniaMetaDataTest=PnemoniaMetaData.loc[PnemoniaMetaData['Dataset_type'] == 'TEST']

PnemoniaMetaDataTest
# augmentation functions
'''
prototype of what we can implement in the future
those functions are my own prefer , but we can edit a lot of course 
i %1 before update aug -> 86 accuracy
i %1 after update aug  -> recal 100% bad accuracy , overfitting
'''
def zoom(image):
  zoom = iaa.Affine(scale=(1, 1.5))          # this scale of zoom in image ratio of 30%
  image = zoom.augment_image(image)
  return image

def pan(image):                              # image shifted over x-y axis helps to highlights region 
  pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})        # 10% to left or right and same ratio to up-down
  image = pan.augment_image(image)
  return image

def brightness(image):                       # change brightness by multiplying pixels as mask
    brightness = iaa.Multiply((0.7, 1.7))    # 0.2-1.2 is limits of multipler 
    image = brightness.augment_image(image)  # note : if value is less than 1, means darker image
    return image

def flip(image):                             # flipping only about x_axis
    image = cv2.flip(image,1)               
    return image

def random_augment(image):
    if np.random.rand() < 0.1:
        image = pan(image)
    if np.random.rand() < 0.6:
        image = zoom(image)
    if np.random.rand() < 0.6:
        image = brightness(image)
  # if np.random.rand() < 0.1:
    #   image = flip(image)
    
    return image
X_Test_level1 = []
Y_Test_level1= []

X_val_level1 = []
Y_val_level1= []

for i in range (0,390):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(PnemoniaMetaDataTest.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    #if i %2 :
    X_Test_level1.append(reshapedimage)
    Y_Test_level1.append(1.0)
    #else :
     #   X_val_level1.append(reshapedimage)
      #  Y_val_level1.append(1.0)
for i in range (0,234):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(normalMetaDataTest.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    #if i %2 :
    X_Test_level1.append(reshapedimage)
    Y_Test_level1.append(0.0)
    #else :
     #   X_val_level1.append(reshapedimage)
      #  Y_val_level1.append(0.0)


X_Test_level1=np.array(X_Test_level1)
Y_Test_level1=np.array(Y_Test_level1)

X_val_level1=np.array(X_val_level1)
Y_val_level1=np.array(Y_val_level1)

X_Train_level1 = []
Y_Train_level1= []
for i in range (0,1500):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+ str(PnemoniaMetaDataTrain.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    if i% 1 == 0:
        aug_image = random_augment(reshapedimage)
        X_Train_level1.append(aug_image)
        Y_Train_level1.append(1.0)
    X_Train_level1.append(reshapedimage)
    Y_Train_level1.append(1.0)
for i in range (0,1342):
    path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'+ str(normalMetaDataTrain.iloc[i]['X_ray_image_name'])
    reshapedimage =cv2.resize(cv2.imread(path, 1), (224,224))
    if i% 1 == 0:
        aug_image = random_augment(reshapedimage)
        X_Train_level1.append(aug_image)
        Y_Train_level1.append(0.0)
    X_Train_level1.append(reshapedimage)
    Y_Train_level1.append(0.0)
    


X_Train_level1=np.array(X_Train_level1)
Y_Train_level1=np.array(Y_Train_level1)
print(X_Train_level1.shape)
print(X_Test_level1.shape)
print(X_val_level1.shape)

# test /debug cell
'''
path='../input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'+ str(PnemoniaMetaDataTest.iloc[1]['X_ray_image_name'])
 
original_image = cv2.resize(cv2.imread(path, 1), (224,224))
aug_image = random_augment(original_image)
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title('Original Image')

axs[1].imshow(aug_image)
axs[1].set_title('Brightness altered image ')
'''
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_Train_level1),
                                                 Y_Train_level1)

InputShape=(224,224,3)
act='relu'
def VGG():
    inputs = Input(shape=InputShape)

# First conv block
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

# Second conv block
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

# Third conv block
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

# Fourth conv block
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

# Fifth conv block
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)

# FC layer
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.7)(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.3)(x)

# Output layer
    output = Dense(units=1, activation='sigmoid')(x)

# Creating model and compiling
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer='adam',metrics=["binary_accuracy"])
    checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')

    return model
modelVGG1=VGG()

hist=modelVGG1.fit(X_Train_level1, Y_Train_level1, epochs=50, batch_size=64,
                   class_weight= class_weights,
                   validation_data=(X_val_level1, Y_val_level1),
                   shuffle = 1)
print(modelVGG1.evaluate(X_Test_level1, Y_Test_level1, verbose=1))

from sklearn.metrics import accuracy_score, confusion_matrix

preds = modelVGG1.predict(X_Test_level1)

acc = accuracy_score(Y_Test_level1, np.round(preds))*100
cm = confusion_matrix(Y_Test_level1, np.round(preds))
tn, fp, fn, tp = cm.ravel()

print('CONFUSION MATRIX ------------------')
print(cm)

print('\nTEST METRICS ----------------------')
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100
print('Accuracy: {}%'.format(acc))
print('Precision: {}%'.format(precision))
print('Recall: {}%'.format(recall))
print('F1-score: {}'.format(2*precision*recall/(precision+recall)))

#print('\nTRAIN METRIC ----------------------')
#print('Train acc: {}'.format(hist.history['loss']))

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
