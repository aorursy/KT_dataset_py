# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/dog-breed-identification'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, cv2, random, time, shutil, csv

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from tqdm import tqdm

np.random.seed(42)

%matplotlib inline 



import keras

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model,Sequential

from keras.layers import Dense, Dropout, InputLayer, Input, Flatten,MaxPooling2D,Conv2D,Activation,GlobalAveragePooling2D

from keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import load_img,img_to_array
def get_num_files(path):

    if not os.path.exists(path):

        return 0

    return sum([len(files) for r, d, files in os.walk(path)])
#Data Paths

train_dir = '/kaggle/input/dog-breed-identification/train'

test_dir = '/kaggle/input/dog-breed-identification/test'
#Count/Print train and test samples.

data_size = get_num_files(train_dir)

test_size = get_num_files(test_dir)

print('Data samples size: ', data_size)

print('Test samples size: ', test_size)
#Read train labels.

labels_dataframe = pd.read_csv('../input/dog-breed-identification/labels.csv')

#Read sample_submission file to be modified by pridected labels.

sample_df = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')

#Incpect labels_dataframe.

sample_df.head()
len(labels_dataframe['breed'])
#Create list of alphabetically sorted labels.

dog_breeds = sorted(list(set(labels_dataframe['breed'])))

n_classes = len(dog_breeds)

print(n_classes)

dog_breeds[:10]
#Map each label string to an integer label.

class_to_num = dict(zip(dog_breeds, range(n_classes)))
def images_to_array(data_dir, labels_dataframe, img_size = (224,224,3)):

    '''

    1- Read image samples from certain directory.

    2- Risize it, then stack them into one big numpy array.

    3- Read sample's label form the labels dataframe.

    4- One hot encode labels array.

    5- Shuffle Data and label arrays.

    '''

    images_names = labels_dataframe['id']

    images_labels = labels_dataframe['breed']

    data_size = len(images_names)

    #initailize output arrays.

    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)

    y = np.zeros([data_size,1], dtype=np.uint8)

    #read data and lables.

    for i in tqdm(range(data_size)):

        image_name = images_names[i]

        img_dir = os.path.join(data_dir, image_name+'.jpg')

        img_pixels = load_img(img_dir,target_size=img_size)

        X[i] = img_pixels

        

        image_breed = images_labels[i]

        y[i] = class_to_num[image_breed]

    

    #One hot encoder

    y = to_categorical(y)

    #shuffle    

    ind = np.random.permutation(data_size)

    X = X[ind]

    y = y[ind]

    print('Ouptut Data Size: ', X.shape)

    print('Ouptut Label Size: ', y.shape)

    return X, y
img_size = (224,224,3)

X, y = images_to_array(train_dir, labels_dataframe, img_size)

X.shape
# mdl=Sequential()

# mdl.add(Conv2D(16,2,input_shape=(224,224,3),activation='relu'))

# mdl.add(MaxPooling2D())

# mdl.add(Conv2D(32,2,activation='relu'))

# mdl.add(MaxPooling2D())

# mdl.add(Conv2D(64,2,activation='relu'))

# mdl.add(MaxPooling2D())

# mdl.add(Dense(1000,activation='relu'))

# mdl.add(GlobalAveragePooling2D())

# mdl.add(Dense(120,activation='softmax'))

# mdl.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# mdl.fit(X,y,epochs=10,batch_size=16,verbose=2,validation_split=0.2)
from keras import applications
model=applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(224,224,3))

model.summary()
x=model.output

x=GlobalAveragePooling2D()(x)

#x=Dense(1024,activation='relu')(x)

predictions=Dense(n_classes,activation='softmax')(x)

md2=Model(model.input,predictions)

for layers in model.layers:

    layers.trainable=False
md2.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

md2.fit(X,y,epochs=10,verbose=2,validation_split=0.2,batch_size=32)
for layers in md2.layers[:249]:

    layers.trainable=False

for layers in md2.layers[249:]:

    layers.trainable=True
from keras.optimizers import SGD
md2.compile(optimizer=SGD(lr=0.00001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

md2.fit(X,y,epochs=10,verbose=2,validation_split=0.2,batch_size=16)
def images_to_array(data_dir, sample_df, img_size = (224,224,3)):

    '''

    1- Read image samples from certain directory.

    2- Risize it, then stack them into one big numpy array.

    3- Read sample's label form the labels dataframe.

    4- One hot encode labels array.

    5- Shuffle Data and label arrays.

    '''

    images_names = sample_df['id']

    #images_labels = labels_dataframe['breed']

    data_size = len(images_names)

    #initailize output arrays.

    xts= np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)

    yts = np.zeros([data_size,1], dtype=np.uint8)

    #read data and lables.

    for i in tqdm(range(data_size)):

        image_name = images_names[i]

        img_dir = os.path.join(data_dir, image_name+'.jpg')

        img_pixels = load_img(img_dir,target_size=img_size)

        xts[i] = img_pixels

        

#         image_breed = images_labels[i]

#         y[i] = class_to_num[image_breed]

    

#     #One hot encoder

#     y = to_categorical(y)

#     #shuffle    

#     ind = np.random.permutation(data_size)

#     X = X[ind]

#     y = y[ind]

#     print('Ouptut Data Size: ', X.shape)

#     print('Ouptut Label Size: ', y.shape)

    return xts
xts=images_to_array(test_dir, sample_df, img_size = (224,224,3))
pre=md2.predict(xts)
pre.shape
sub=pd.DataFrame(data=pre,columns=sample_df.drop('id',axis=1).columns)
sub.head()
sub['id']=sample_df['id']
sub.head()
sub=sub[sample_df.columns]
sub.head()
sub.to_csv('submission.csv',index=False)
sub.head()
sample_df.head()