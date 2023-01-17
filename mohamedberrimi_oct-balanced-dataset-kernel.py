# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import os

from glob import glob

import matplotlib.pyplot as plt

import random

import cv2

import pandas as pd

import numpy as np

import matplotlib.gridspec as gridspec

import seaborn as sns

import zlib

import itertools

import sklearn

import itertools

import scipy

import skimage

from skimage.transform import resize

import csv

from tqdm import tqdm

from sklearn import model_selection

from sklearn.model_selection import train_test_split, learning_curve,KFold,cross_val_score,StratifiedKFold

from sklearn.utils import class_weight

from sklearn.metrics import confusion_matrix

import keras

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras import models, layers, optimizers

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.utils import class_weight

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop

from keras.models import Sequential, model_from_json

from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D,MaxPooling2D,AveragePooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K

from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score



from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(

  rescale = 1./255,

    validation_split=0.1

  )

 

training_set = train_datagen.flow_from_directory('../input/oct_balanced_version/Balanced',

  target_size = (140, 140),

  batch_size = 28860,

  classes=["DME","CNV","NORMAL","DRUSEN"],

  subset='training')

 

validation_set = train_datagen.flow_from_directory('../input/oct_balanced_version/Balanced',

  target_size = (140, 140),

  batch_size = 3204,

  subset='validation',

  classes=["DME","CNV","NORMAL","DRUSEN"])



#test_set = train_datagen.flow_from_directory('../input/test/test',

#  target_size = (190, 190),

#  batch_size = 968,

#  classes=["DME","CNV","NORMAL","DRUSEN"])

 
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

 

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





def plotKerasLearningCurve():

    plt.figure(figsize=(10,5))

    metrics = np.load('logs.npy')[()]

    filt = ['acc'] # try to add 'loss' to see the loss learning curve

    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):

        l = np.array(metrics[k])

        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')

        x = np.argmin(l) if 'loss' in k else np.argmax(l)

        y = l[x]

        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')

        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   

    plt.legend(loc=4)

    plt.axis([0, None, None, None]);

    plt.grid()

    plt.xlabel('Number of epochs')

    plt.ylabel('Accuracy')



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (5,5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



def plot_learning_curve(history):

    plt.figure(figsize=(8,8))

    plt.subplot(1,2,1)

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./accuracy_curve.png')

    plt.subplot(1,2,2)

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./loss_curve.png')

map_characters1 = {0: 'Normal', 1: 'CNV', 2: 'DME', 3: 'DRUSEN'}



x_train.shape
x_test,ytest=next(validation_set)

x_train, y_train = next(training_set)

#x_test, ytest=next(test_set)
m_samples = x_train.shape[0]

m_samplesTest = x_test.shape[0]

X_train = x_train.reshape(m_samples, -1)

X_test = x_test.reshape(m_samplesTest, -1)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score  



RandomForestClassifier=RandomForestClassifier(n_estimators=300)



RandomForestClassifier.fit(X_train,y_train)



y_pred=RandomForestClassifier.predict(X_test)



print('F1 score : ',f1_score(ytest,y_pred,average="weighted"))

print('recall_score : ',recall_score(ytest,y_pred,average="weighted"))

print('precision_score : ',precision_score(ytest,y_pred,average="weighted"))

Accuracy=RandomForestClassifier.score(X_test,ytest)

print('The accuracy of Random Forest classifier is : ',Accuracy)
classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 527, kernel_initializer = 'uniform', activation = 'relu', input_dim = 58800))

# Adding the second hidden layer

classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.9))



classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())



classifier.add(Dropout(0.8))

classifier.add(Dense(units = 300, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())



classifier.add(Dropout(0.8))



# Adding the second hidden layer



classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.7))



# Adding the third hidden layer

classifier.add(Dense(units = 250, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Dropout(0.9))

 # Adding the output layer

classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train,validation_data=(X_test, ytest), batch_size = 50, epochs = 30)

y_pred=classifier.predict(X_test)

y_pred = (y_pred > 0.5) 

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

print('F1 score : ',f1_score(ytest,y_pred,average="micro"))

print('recall_score : ',recall_score(ytest,y_pred,average="weighted"))

print('precision_score : ',precision_score(ytest,y_pred,average="weighted"))
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



# Model Accuracy, how often is the classifier correct?

y_pred = clf.predict(X_test)

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



print("Decision tree Accuracy:",metrics.accuracy_score(ytest, y_pred))

print('F1 score : ',f1_score(ytest,y_pred,average="weighted"))

print('recall_score : ',recall_score(ytest,y_pred,average="weighted"))

print('precision_score : ',precision_score(ytest,y_pred,average="weighted"))
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (4, 4)))

classifier.add(Dropout(0.3))

classifier.add(Conv2D(512, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Conv2D(80, (3, 3), activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(BatchNormalization())



classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(0.3))

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 4, activation = 'sigmoid'))

 
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list1 = [checkpoint]

callbacks_list = callbacks_list1+[keras.callbacks.EarlyStopping(monitor='val_acc', patience=8, verbose=1)]

history=classifier.fit_generator(training_set,

                            steps_per_epoch = 28860/124,

                            validation_data = (valid_X,valid_Y), 

                            validation_steps= 3204/100,

                            epochs = 15,

                            shuffle=True, 

                            callbacks = callbacks_list+[MetricsCheckpoint('logs')])



 
classifier.save(weights.hdf5)