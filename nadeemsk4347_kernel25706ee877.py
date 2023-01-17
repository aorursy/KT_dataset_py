# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

 

#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder



#preprocess.

from keras.preprocessing.image import ImageDataGenerator



#dl libraraies

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.callbacks import ReduceLROnPlateau



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

 

import tensorflow as tf

import random as rn



# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os                   

from random import shuffle  

from zipfile import ZipFile

from PIL import Image



#TL pecific modules

from keras.applications.vgg16 import VGG16
X = np.load('/kaggle/input/module4/X.npy')

Y = np.load('/kaggle/input/module4/Z.npy')



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.10,random_state=42)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=42)

base_model=VGG16(include_top=False, weights=None,input_shape=(50,50,3), pooling='avg')

 
weights_path = '/kaggle/input/trans-learn-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model.load_weights(weights_path)
base_model.summary()
model=Sequential()

model.add(base_model)



model.add(Dense(256,activation='relu'))

model.add(Dense(2,activation='softmax'))
epochs=50

batch_size=30

red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
model.summary()
base_model.trainable=False # setting the VGG model to be untrainable.
model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])



history = model.fit(x=x_train, y=y_train,

                    validation_data=(x_val, y_val),

                    batch_size=30,

                    epochs=10,

                    verbose=1)
model.save('transfer_wala_model.h5')

np.save("x_test.npy", x_test)

np.save("y_test.npy", y_test)
for i in range (len(base_model.layers)):

    print (i,base_model.layers[i])

  

for layer in base_model.layers[11:]:

    layer.trainable=True

for layer in base_model.layers[0:11]:

    layer.trainable=False

  
model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train,

                    validation_data=(x_val, y_val),

                    batch_size=30,

                    epochs=10,

                    verbose=1)
model.save('transfer_with_fine_tuning.h5')