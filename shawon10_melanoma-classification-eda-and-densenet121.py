# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

import time, gc

import cv2

from keras import backend as K

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization,Activation

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns

from keras.models import Model,Sequential, Input, load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.applications import DenseNet121
import os

import gc

import json

import math

import cv2

import PIL

from PIL import Image

import numpy as np



import matplotlib.pyplot as plt

import pandas as pd



import scipy

from tqdm import tqdm

%matplotlib inline

from keras.preprocessing import image
sample_submission = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")

test = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
train.head()
plt.hist(train['target'])

plt.title('Frequency Histogram of Melanoma')

plt.figure(figsize=(12, 12))

plt.show()
#def preprocess_image(image_path, desired_size=imSize):

    #im = Image.open(image_path)

    #im = im.resize((desired_size, )*2, resample=Image.LANCZOS)

    

    #return im
#N = train.shape[0]//3

#x_train = np.empty((N, imSize, imSize, 3), dtype=np.uint8)

#for i, image_id in enumerate(tqdm(train['image_name'])):

    #if i==N:

        #break

    #x_train[i, :, :, :] = preprocess_image(

       # f'../input/siim-isic-melanoma-classification/jpeg/train/{image_id}.jpg'

    #)
x = train['image_name']

train_malignant=train[train['target'] == 1]

train_benign=train[train['target'] == 0]

train_benign=train_benign[0:584]

img_size=64
train_malignant.head()
train_benign.head()
train_malignant.shape
train_benign.shape
train_balanced = pd.concat([train_benign, train_malignant])

train_balanced.head()
train_balanced.tail()
train_balanced.shape
plt.hist(train_balanced['target'])

plt.title('Frequency Histogram of Balanced Melanoma')

plt.figure(figsize=(12, 12))

plt.show()
train_image=[]

for i,name in enumerate(tqdm(train_balanced['image_name'])):

    path='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    train_image.append(image)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(train_image[i])
test.head()
test_image=[]

for i,name in enumerate(tqdm(test['image_name'])):

    path='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'+name+'.jpg'

    img=cv2.imread(path)

    image=cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_AREA)

    test_image.append(image)
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

for i in range(4):

    ax[i].set_axis_off()

    ax[i].imshow(test_image[i])
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
y = train_balanced['target']

y.tail()
from keras.utils.np_utils import to_categorical

y_train = np.array(y.values)

y_train = to_categorical(y_train, num_classes=2)

print(y_train.shape,y_train[1100])

print(y_train[3])
EPOCHS = 80

SIZE=64

N_ch=3

BATCH_SIZE = 64
def build_densenet():

    densenet = DenseNet121(weights='imagenet', include_top=False)



    input = Input(shape=(SIZE, SIZE, N_ch))

    x = Conv2D(3, (3, 3), padding='same')(input)

    

    x = densenet(x)

    

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)



    # multi output

    output = Dense(2,activation = 'softmax', name='root')(x)

 



    # model

    model = Model(input,output)

    

    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    

    return model
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, y_train, test_size=0.2, random_state=42)
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

hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),

               steps_per_epoch=X_train.shape[0] // BATCH_SIZE,

               epochs=EPOCHS,

               verbose=1,

               callbacks=[annealer, checkpoint],

               validation_data=(X_val, Y_val))
final_loss, final_accuracy = model.evaluate(X_val, Y_val)

print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))
predict = model.predict(X_Test)

print(predict)

result=[]

disease_class=['0','1']

for i in range(len(predict)):

    ind=np.argmax(predict[i])

    result.append(disease_class[ind])
sample_submission["target"]= result

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head()