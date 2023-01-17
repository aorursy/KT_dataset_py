# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2

from keras import backend as K

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model,Sequential,Input

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization,Activation,GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns

from keras.applications import DenseNet121



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))
sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
train.head()
x = train['image_id']
img_size=100
train_image=[]

for name in train['image_id']:

    path='/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    train_image.append(image)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(train_image[i])
test.head()
test_image=[]

for name in test['image_id']:

    path='/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    test_image.append(image)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(test_image[i])
#from keras.preprocessing.image import img_to_array

X_Train = np.ndarray(shape=(len(train_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in train_image:

    #X_Train[i]=img_to_array(image)

    X_Train[i]=train_image[i]

    i=i+1

X_Train=X_Train/255

print('Train Shape: {}'.format(X_Train.shape))
X_Test = np.ndarray(shape=(len(test_image), img_size, img_size, 3),dtype = np.float32)

i=0

for image in test_image:

    #X_Test[i]=img_to_array(image)

    X_Test[i]=test_image[i]

    i=i+1

    

X_Test=X_Test/255

print('Test Shape: {}'.format(X_Test.shape))
y = train.copy()

del y['image_id']

y.head()
y_train = np.array(y.values)

print(y_train.shape,y_train[0])
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, y_train, test_size=0.2, random_state=42)
def build_densenet():

    densenet = DenseNet121(weights='imagenet', include_top=False)



    input = Input(shape=(img_size, img_size, 3))

    x = Conv2D(3, (3, 3), padding='same')(input)

    

    x = densenet(x)

    

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)



    # multi output

    output = Dense(4,activation = 'softmax', name='root')(x)

 



    # model

    model = Model(input,output)

    

    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    

    return model
model = build_densenet()

annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)

checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)

# Generates batches of image data with data augmentation

datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations

                        width_shift_range=0.2, # Range for random horizontal shifts

                        height_shift_range=0.2, # Range for random vertical shifts

                        zoom_range=0.2, # Range for random zoom

                        horizontal_flip=True, # Randomly flip inputs horizontally

                        vertical_flip=True) # Randomly flip inputs vertically



datagen.fit(X_train)

# Fits the model on batches with real-time data augmentation

hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),

               steps_per_epoch=X_train.shape[0] // 32,

               epochs=80,

               verbose=1,

               callbacks=[annealer, checkpoint],

               validation_data=(X_val, Y_val))
predict = model.predict(X_Test)

all_predict = np.ndarray(shape = (test.shape[0],4),dtype = np.float32)

for i in range(0,test.shape[0]):

    for j in range(0,4):

        if predict[i][j]==max(predict[i]):

            all_predict[i][j] = 1

        else:

            all_predict[i][j] = 0 
healthy = [y_test[0] for y_test in all_predict]

multiple_diseases = [y_test[1] for y_test in all_predict]

rust = [y_test[2] for y_test in all_predict]

scab = [y_test[3] for y_test in all_predict]
df = {'image_id':test.image_id,'healthy':healthy,'multiple_diseases':multiple_diseases,'rust':rust,'scab':scab}
data = pd.DataFrame(df)

data.tail()
data.to_csv('submission.csv',index = False)