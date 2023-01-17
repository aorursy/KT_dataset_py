# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.listdir('../input/plantseedlings/input/input/plant-seedlings-classification/train/')



# Any results you write to the current directory are saved as output.
import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)
print(os.listdir('../input/seed-augumented/seedling_data/train'))

import fnmatch

import os

import numpy as np

import pandas as pd

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

import cv2

np.random.seed(21)



path = '../input/plantseedlings/input/input/plant-seedlings-classification/train/'

train_label = []

train_img = []

label2num = {'Loose Silky-bent':0, 'Charlock':1, 'Sugar beet':2, 'Small-flowered Cranesbill':3,

             'Common Chickweed':4, 'Common wheat':5, 'Maize':6, 'Cleavers':7, 'Scentless Mayweed':8,

             'Fat Hen':9, 'Black-grass':10, 'Shepherds Purse':11}

for i in os.listdir(path):

    label_number = label2num[i]

    new_path = path+i+'/'

    for j in fnmatch.filter(os.listdir(new_path), '*.png'):

        img = cv2.imread(new_path + j, cv2.IMREAD_COLOR)

        img = cv2.resize(img.copy(), (200,200), interpolation = cv2.INTER_AREA)

        #temp_img = image.load_img(new_path+j)#, target_size=(200,200))

        #temp_img = temp_img - temp_img.mean()

        

        #img = cv2.imread(new_path + j)

        #temp_img = cv2.resize(img,(200,200))

        ## convert to hsv

        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



        ## mask of green (36,25,25) ~ (86, 255,255)

        #mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))

        #mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))



        ## slice the green

        #imask = mask>0

        #green = np.zeros_like(img, np.uint8)

        #green[imask] = img[imask]

        

        

        #green = cv2.resize(green, (200,200), interpolation = cv2.INTER_AREA)

        #plt.imshow(green)

        temp = image.img_to_array(img)

        temp1 = 0

        temp1 = temp / 255

        temp = (temp1 - temp1.mean())/temp1.std()

        train_label.append(label_number)

        #temp_img = image.img_to_array(temp_img)

        #temp_img = temp_img/255

        #temp_img = temp_img - temp_img.mean() / temp_img.std()

        train_img.append(temp)



train_img = np.array(train_img)



train_y=pd.get_dummies(train_label)

train_y = np.array(train_y)

#train_img=preprocess_input(train_img)



print('Training data shape: ', train_img.shape)

print('Training labels shape: ', train_y.shape)
%matplotlib inline

import matplotlib.pyplot as plt

#plt.imshow(train_img[1])

#print(train_img[1])

img = cv2.imread(os.path.join(path, 'Fat Hen',os.listdir(path + 'Fat Hen/')[0]), cv2.IMREAD_COLOR)

img = cv2.resize(img.copy(), (200,200), interpolation = cv2.INTER_AREA)

temp = image.img_to_array(img)

#print(img)\



#temp = temp / 255

#temp = (temp - temp.mean())/temp.std()

plt.imshow(train_img[2])

print(train_img[2])
import keras

from keras.models import Sequential,Model

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator



def vgg16_model(num_classes=None):



    model = VGG16(weights='imagenet', include_top=False,input_shape=(200,200,3))

    model.layers.pop()

    model.layers.pop()

    model.layers.pop()



    model.outputs = [model.layers[-1].output]



    model.layers[-2].outbound_nodes= []

    x=Conv2D(256, kernel_size=(2,2),strides=2)(model.output)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)    

    x=Conv2D(128, kernel_size=(2,2),strides=1)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)    

    x=Conv2D(128, kernel_size=(2,2),strides=1)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x=Flatten()(x)

    x=Dense(num_classes, activation='softmax')(x)



    model=Model(model.input,x)



    for layer in model.layers[:15]:



        layer.trainable = False





    return model
def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision





def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def fscore(y_true, y_pred):

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:

        return 0



    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    f_score = 2 * (p * r) / (p + r + K.epsilon())

    return f_score
from keras import backend as K

num_classes=12

model = vgg16_model(num_classes)

model.compile(optimizer="nadam", loss='categorical_crossentropy', metrics=['accuracy',fscore])

model.summary()

#Split training data into rain set and validation set

from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.1, random_state=42)
from keras.callbacks import ModelCheckpoint

epochs = 10

batch_size = 32

# model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)



model_checkpoint = ModelCheckpoint('./model61.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5',

                           monitor='val_loss',

                             verbose=1,

                            save_best_only=True,

                            mode='min',

                             save_weights_only=False)





model.fit(X_train,Y_train,

          batch_size=128,

          epochs=20,

          verbose=1, shuffle=True, validation_data=(X_valid,Y_valid), callbacks=[model_checkpoint])
from keras.applications import resnet50

base_model = resnet50.ResNet50(weights= 'imagenet', include_top=False, input_shape= (200,200,3))
from keras.layers import GlobalAveragePooling2D

#base_model.layers.pop()

#base_model.layers.pop()

#base_model.layers.pop()



#base_model.outputs = [base_model.layers[-1].output]

#base_model.layers[-2].outbound_nodes= []



#base_model.outputs

x = GlobalAveragePooling2D()(base_model.output)

x = Dropout(0.7)(x)

predictions = Dense(12, activation= 'softmax')(x)

model_res = Model(inputs = base_model.input, outputs = predictions)
model_res.summary()
from keras.optimizers import SGD, Adam

#sgd = SGD(lr=0.001, momentum=0.9, decay=1e-8, nesterov=False)

adam = Adam(lr=0.0001)

model_res.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
model_checkpoint = ModelCheckpoint('./model61.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}_resnet50.h5',

                           monitor='val_loss',

                             verbose=1,

                            save_best_only=True,

                            mode='min',

                             save_weights_only=False)



model_res.fit(X_train,Y_train,batch_size = 128 , epochs=20,validation_data=(X_valid,Y_valid),callbacks=[model_checkpoint])
#from keras.applications import InceptionV3

#model_v3 = InceptionV3.