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

from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,CSVLogger,ReduceLROnPlateau,LearningRateScheduler



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,GlobalAveragePooling2D

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
lookup = dict()

reverselookup = dict()

count = 0

for j in os.listdir('../input/leapgestrecog/leapGestRecog/00/'):

    if not j.startswith('.'): # If running this code locally, this is to 

                              # ensure you aren't reading in hidden folders

        lookup[j] = count

        reverselookup[count] = j

        count = count + 1

lookup
x_data = []

y_data = []

IMG_SIZE = 150

datacount = 0 # We'll use this to tally how many images are in our dataset

for i in range(0, 10): # Loop over the ten top-level folders

    for j in os.listdir('../input/leapgestrecog/leapGestRecog/0' + str(i) + '/'):

        if not j.startswith('.'): # Again avoid hidden folders

            count = 0 # To tally images of a given gesture

            for k in os.listdir('../input/leapgestrecog/leapGestRecog/0' + 

                                str(i) + '/' + j + '/'):

                                # Loop over the images

                path = '../input/leapgestrecog/leapGestRecog/0' + str(i) + '/' + j + '/' + k

                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

                arr = np.array(img)

                x_data.append(arr) 

                count = count + 1

            y_values = np.full((count, 1), lookup[j]) 

            y_data.append(y_values)

            datacount = datacount + count

x_data = np.array(x_data, dtype = 'float32')

y_data = np.array(y_data)

y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
# check some image

fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=rn.randint(0,len(y_data))

        ax[i,j].imshow(x_data[l])

        ax[i,j].set_title(reverselookup[y_data[l,0]])

        

plt.tight_layout()
y_data=to_categorical(y_data)

x_data = x_data.reshape((datacount, IMG_SIZE, IMG_SIZE, 1))

x_data = x_data/255
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.25,random_state=42)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,1)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dense(10, activation = "softmax"))
batch_size=128

epochs=10
checkpoint = ModelCheckpoint(

    './base.model',

    monitor='val_loss',

    verbose=1,

    save_best_only=True,

    mode='min',

    save_weights_only=False,

    period=1

)

earlystop = EarlyStopping(

    monitor='val_loss',

    min_delta=0.001,

    patience=30,

    verbose=1,

    mode='auto'

)

tensorboard = TensorBoard(

    log_dir = './logs',

    histogram_freq=0,

    batch_size=16,

    write_graph=True,

    write_grads=True,

    write_images=False,

)



csvlogger = CSVLogger(

    filename= "training_csv.log",

    separator = ",",

    append = False

)



reduce = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=3,

    verbose=1, 

    mode='auto'

)



callbacks = [checkpoint,tensorboard,csvlogger,reduce]
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
History = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test),callbacks=callbacks)
plt.plot(History.history['loss'])

plt.plot(History.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('Loss')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()
plt.plot(History.history['acc'])

plt.plot(History.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend(['train', 'test'])

plt.show()