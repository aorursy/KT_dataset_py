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

from keras import regularizers

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical

from keras.callbacks import ReduceLROnPlateau



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.layers import InputLayer

 

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

import keras.preprocessing.image as img

from keras.applications.resnet50 import ResNet50
os.listdir('../input/mydataset/mydataset/ptrain')
def make_train_data(label,DIR):

    for img in tqdm(os.listdir(DIR)):

        try:

            path = os.path.join(DIR,img)

            img = cv2.imread(path,cv2.IMREAD_COLOR)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

            X.append(np.array(img))

            Z.append(str(label))

        except:

            pass

        

def assign_label(img,label):

    return label


#ax=[]

IMG_SIZE=256

Plain='../input/mydataset/mydataset/ptrain/normal'

Pothole='../input/mydataset/mydataset/ptrain/pothole'



X=[]

Z=[]

make_train_data('Plain',Plain)

make_train_data('Pothole',Pothole)
fig,ax=plt.subplots(2,5)

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0)

fig.set_size_inches(10,10)

for i in range(2):

    for j in range (5):

        l=rn.randint(0,len(Z))

        ax[i,j].imshow(X[l][:])

        ax[i,j].set_title(Z[l])

        ax[i,j].set_aspect('equal')
le=LabelEncoder()

Y=le.fit_transform(Z)

Y=to_categorical(Y,2)

print(Y)

X=np.array(X)



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1337)



np.random.seed(42)

rn.seed(42)
base_model=ResNet50(include_top=False, weights='imagenet',input_shape=(256,256,3), pooling='max')

base_model.summary()
model=Sequential()

model.add(base_model)

model.add(Dropout(0.20))

model.add(Dense(2048,activation='relu'))

model.add(Dense(1024,activation='relu'))

model.add(Dense(512,activation='relu'))

model.add(Dense(2,activation='softmax'))
epochs=50

batch_size=128

red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=2, verbose=1)

base_model.trainable=True 

model.compile(optimizer=Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
History = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test,y_test))
model.save('model.hdf5')
from IPython.display import FileLink

FileLink(r'model.hdf5')