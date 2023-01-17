import numpy as np

import pandas as pd

import os

import gc

import time

print(os.listdir("../input/airbus-ship-detection"))
train=pd.read_csv('../input/airbus-ship-detection/train_ship_segmentations_v2.csv')
train.head()
train['exist_ship'] = train['EncodedPixels'].fillna(0)

train.loc[train['exist_ship']!=0,'exist_ship']=1

del train['EncodedPixels']
print(len(train['ImageId']))

print(train['ImageId'].value_counts().shape[0])

train_gp = train.groupby('ImageId').sum().reset_index()

train_gp.loc[train_gp['exist_ship']>0,'exist_ship']=1
train_gb = train.groupby(['ImageId']).sum().reset_index()

train_gb.loc[train_gb['exist_ship']>0, 'exist_ship'] = 1
print(train_gb['exist_ship'].value_counts())
train_gb = train_gb.sort_values(by='exist_ship')

train_gb = train_gb.drop(train.index[0:100000])
train_0 = train_gb[train_gb['exist_ship']==0].sample(24000,random_state=100)

train_0_15 = train_0.iloc[0:5000]#0,15000

train_1 = train_gb[train_gb['exist_ship']==1].sample(20000,random_state=100)

train_1_15 = train_1.iloc[0:5000]#0,15000

test_0 = train_0.iloc[500:1200]#15000，23000

test_1 = train_1.iloc[500:800]#15000，18000

train_sample =pd.concat([train_0_15,train_1_15])

test_sample =pd.concat([test_0,test_1])
train_path = '../input/airbus-ship-detection/train_v2/'

test_path = '../input/airbus-ship-detection/test_v2/'
X = np.empty(shape=(len(train_sample), 256,256,3),dtype=np.uint8)

y = np.empty(shape=len(train_sample),dtype=np.uint8)
from PIL import Image
for index, image in enumerate(train_sample['ImageId']):

    image_array= Image.open(train_path + image).resize((256,256)).convert('RGB')

    X[index] = image_array

    y[index]=train_sample[train_sample['ImageId']==image]['exist_ship'].iloc[0]



print(X.shape)

print(y.shape)
test_X = np.empty(shape=(len(test_sample), 256,256,3),dtype=np.uint8)

test_Y = np.empty(shape=len(test_sample),dtype=np.uint8)

for index, image in enumerate(test_sample['ImageId']):

    image_array= Image.open(train_path + image).resize((256,256)).convert('RGB')

    test_X[index] = image_array

    test_Y[index]=test_sample[test_sample['ImageId']==image]['exist_ship'].iloc[0]



print(test_X.shape)

print(test_Y.shape)
from sklearn.preprocessing import OneHotEncoder

targets =y.reshape(len(y),-1)

enc = OneHotEncoder()

enc.fit(targets)

y = enc.transform(targets).toarray()

print(y.shape)



targetss =test_Y.reshape(len(test_Y),-1)

enc1 = OneHotEncoder()

enc1.fit(targetss)

test_Y = enc1.transform(targetss).toarray()

print(test_Y.shape)
import keras.applications
dir(keras.applications)
from keras.applications.resnet50 import ResNet50 as ResModel
from keras.applications.resnet50 import ResNet50 as ResModel

#from keras.applications.vgg16 import VGG16 as VGG16Model

img_width, img_height = 256, 256

model = ResModel(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.models import Sequential, Model 

from keras import backend as K

for layer in model.layers:

    layer.trainable = False



x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

predictions = Dense(2, activation="softmax")(x)



# creating the final model 

model_final = Model(input = model.input, output = predictions)
def recall(y_true, y_pred):

    # Calculates the recall召回率

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def precision(y_true, y_pred):

    #"""精确率"""

    tp= K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives

    pp= K.sum(K.round(K.clip(y_pred, 0, 1))) # predicted positives

    precision = tp/ (pp+ K.epsilon())

    return precision

    
from keras import optimizers

epochs = 10

lrate = 0.001

decay = lrate/epochs

#adam = optimizers.Adam(lr=lrate,beta_1=0.9, beta_2=0.999, decay=decay)

sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model_final.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[precision, recall])

model_final.summary()
model_final.fit(X, y, epochs=10, batch_size=50)

#score = model_final.evaluate(test_X, test_Y, batch_size=50)



model_final.save('ResNet_transfer_ship.h5')
gc.collect()