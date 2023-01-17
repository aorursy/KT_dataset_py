# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



###########################################################

###########################################################

##-LIBRARY-LOADING

import numpy as np

from glob import glob

import cv2

import fnmatch

import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

 

import sklearn

import keras

from sklearn.model_selection import train_test_split #, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

#from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

from keras.callbacks import Callback#, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

#from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

 

import matplotlib.pylab as plt

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

###########################################################

###########################################################



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
###########################################################

###########################################################

##-DATA-LOADING

Patch = glob('/kaggle/input/IDC_regular_ps50_idx5/**/*.png', recursive=True)

for filename in Patch[0:10]:

    print(filename)

###########################################################

###########################################################
classZero = fnmatch.filter(Patch, '*class0.png')

classOne = fnmatch.filter(Patch, '*class1.png')

###########################################################

###########################################################

##-DATA-PREPROCESSING  

x = []

y = []

 

WIDTH = 50

HEIGHT = 50

 

for img in Patch[0:70000]:

    full_size_image = cv2.imread(img)

    x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))

    if img in classZero:

        y.append(0)

    elif img in classOne:

        y.append(1)

 

df = pd.DataFrame()

df["images"]=x

df["labels"]=y

 

x=np.array(x)

x=x/255.0

 

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2) ##train-test

 

Y_trainH = to_categorical(Y_train, num_classes = 2) ##Label-

Y_testH = to_categorical(Y_test, num_classes = 2)   ##-Encoding

 

#Imbalanced Data

# Make Data 1D for compatability upsampling methods

X_trainFlat = X_train.reshape(X_train.shape[0], 7500)

X_testFlat = X_test.reshape(X_test.shape[0], 7500)

 

ros = RandomUnderSampler(ratio='auto')

X_trainRos, Y_trainRos = ros.fit_sample(X_trainFlat, Y_train)

X_testRos, Y_testRos = ros.fit_sample(X_testFlat, Y_test)

Y_trainRosH = to_categorical(Y_trainRos, num_classes = 2)  #Label- 

Y_testRosH = to_categorical(Y_testRos, num_classes = 2)    #-Encoding

 

for i in range(len(X_trainRos)):

    height, width, channels = 50,50,3

    X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)

for i in range(len(X_testRos)):

    height, width, channels = 50,50,3

    X_testRosReshaped = X_testRos.reshape(len(X_testRos),height,width,channels)

 

from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

print("Old Class Weights: ",class_weight)

from sklearn.utils import class_weight

class_weight2 = class_weight.compute_class_weight('balanced', np.unique(Y_trainRos), Y_trainRos)

print("New Class Weights: ",class_weight2)

###########################################################

###########################################################

###########################################################

###########################################################

 

class MetricsCheckpoint(Callback):

    """Callback that saves metrics after each epoch"""

    def __init__(self, savepath):

        super(MetricsCheckpoint, self).__init__()

        self.savepath = savepath

        self.history = {}

    def on_epoch_end(self, epoch, logs=None):

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        np.save(self.savepath, self.history)

 

classes = 2

epochs = 10

img_rows,img_cols=50,50

input_shape = (img_rows, img_cols, 3)

model = Sequential()

model.add(Conv2D(16, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=input_shape,strides=2))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5)) 

model.add(Dense(classes, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(), ##SGD, RMSprop, Adam, Adagrad, Adadelta

              metrics=['accuracy'])

datagen = ImageDataGenerator(featurewise_center=False,  samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False,

                             zca_whitening=False, rotation_range=15, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True)  



 



 

history = model.fit_generator(datagen.flow(X_trainRosReshaped,Y_trainRosH, batch_size=128),

                    steps_per_epoch=len(X_trainRosReshaped) / 32, epochs=epochs,class_weight=class_weight2,

                    validation_data = [X_trainRosReshaped, Y_trainRosH],callbacks = [MetricsCheckpoint('logs')])
score = model.evaluate(X_testRosReshaped, Y_testRosH, verbose=0)

print('accuracy:', score[1])

 
y_pred = model.predict(X_testRosReshaped)

class_ = {1: 'Positive', 0: 'Negative'}

classification_report = sklearn.metrics.classification_report(np.where(Y_testRosH > 0)[1], np.argmax(y_pred, axis=1), target_names=list(class_.values()))

print(classification_report, sep='')  