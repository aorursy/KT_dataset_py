# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
''''for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
'''
!pip install tensorflow==1.14.0
!pip install keras==2.3.0
'''
from __future__ import print_function

# Utils
import numpy as np
import argparse
import random, glob
import os, sys, csv
import cv2 as cv
import time, datetime

# Files
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

# Deep learning libraries

import keras
from skimage.io import imread



# Networks
from keras.preprocessing import image
from keras.applications.densenet import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

# Layers
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras import backend as K
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation


# Other
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.models import load_model


import tensorflow as tf
print(tf.__version__)


# Setting seeds for reproducibility
seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)



print(keras.__version__)
NUM_CLASSES = 2
BATCH_SIZE = 64
IMAGE_SIZE = 200
WIDTH = IMAGE_SIZE
HEIGHT = IMAGE_SIZE
 


LABELS = ['NORMAL', 'PNEUMONIA']

EPOCHS = 10
train_dir = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/train'
test_dir = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/test'
val_dir = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/val'
train_normal = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/train/NORMAL'
train_pneumonia = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/train/PNEUMONIA'
normal_cases = glob.glob(train_normal +'/*.jpeg')
pneumonia_cases = glob.glob(train_pneumonia +'/*.jpeg')

# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []

# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    train_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases:
    train_data.append((img, 1))

# Get a pandas dataframe from the data we have in our list 
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# Shuffle the data 
train_data = train_data.sample(frac=1.).reset_index(drop=True)

# Get few samples for both the classes
pneumonia_samples = (train_data[train_data['label']==1]['image'].iloc[:5]).tolist()
normal_samples = (train_data[train_data['label']==0]['image'].iloc[:5]).tolist()

# Concat the data in a single list and del the above two list
samples = pneumonia_samples + normal_samples
del pneumonia_samples, normal_samples

# Plot the data 
f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()
input_path='/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/'
for _set in ['train', 'val', 'test']:
    n_normal = len(os.listdir(input_path + _set + '/NORMAL'))
    n_infect = len(os.listdir(input_path + _set + '/PNEUMONIA'))
    print('Set: {}, normal images: {}, pneumonia images: {}'.format(_set, n_normal, n_infect))
import seaborn as sns
l = []
for i in train_data['label'] :
    if(i  == 0):
        l.append("Normal")
    else:
        l.append("Pneumonia")
sns.set_style('darkgrid')
sns.countplot(l) 
## Specify the values for all arguments to data_generator_with_aug.
#from keras.applications.resnet50 import preprocess_input, decode_predictions


data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                             horizontal_flip = True,rescale=1./255,
                                             width_shift_range = 0.2,
                                             height_shift_range = 0.2,
                                             shear_range = 0.2,
                                             zoom_range = 0.2
                                            )
            
data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input ,rescale=1./255            
                                          )

train_generator = data_generator_with_aug.flow_from_directory(
       directory = train_dir ,
       target_size = (IMAGE_SIZE , IMAGE_SIZE ),
       batch_size = BATCH_SIZE ,
       class_mode = 'categorical')

validation_generator = data_generator_no_aug.flow_from_directory(
       directory = val_dir,
       target_size = (IMAGE_SIZE , IMAGE_SIZE ),
       class_mode = 'categorical')

test_generator = data_generator_no_aug.flow_from_directory(
       directory =  test_dir,
       target_size = (IMAGE_SIZE , IMAGE_SIZE ),
       batch_size = BATCH_SIZE ,
       class_mode = 'categorical')
 
nb_train_samples = train_generator.samples
STEPS = nb_train_samples // BATCH_SIZE
print('Step  = {}'.format(STEPS))
#train_gen.samples // batch_size, 
Metrics = ['accuracy', tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]


lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=1, verbose=1)

K.clear_session()
DenseNet_weights_path = '../input/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'

 

DenseNet_model = Sequential()
DenseNet_model.add(DenseNet121(include_top=False, pooling='avg', weights='imagenet'))

DenseNet_model.add(Dense(NUM_CLASSES, activation='softmax'))

DenseNet_model.layers[0].trainable = False

 
DenseNet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=Metrics)
 
print(DenseNet_model.summary())

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=1, verbose=1)
start_time = datetime.datetime.now();


history = DenseNet_model.fit(
       train_generator, # specify where model gets training data
       epochs = EPOCHS,
       steps_per_epoch=STEPS ,callbacks=[lr_reduce] ,
       validation_data=validation_generator) # specify where model gets validation data


# Evaluate the model
scores = DenseNet_model.evaluate(test_generator)
print("\n%s: %.2f%%" % (DenseNet_model.metrics_names[1], scores[1]*100))

end_time = datetime.datetime.now()
elapsed_time = end_time - start_time

print('Start Time  = {} ; End Time  = {} ; Total Time = {}'.format(start_time,end_time,elapsed_time))
DenseNet_model.save('DenseNet121_model.h5')

print("\n%s: %.2f%%" % (DenseNet_model.metrics_names[0], scores[0]*100))
print("\n%s: %.2f%%" % (DenseNet_model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (DenseNet_model.metrics_names[2], scores[2]*100))
print("\n%s: %.2f%%" % (DenseNet_model.metrics_names[3], scores[3]*100))
print("\n%s: %.2f%%" % (DenseNet_model.metrics_names[4], scores[4]*100))

from sklearn.metrics import accuracy_score, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
preds = DenseNet_model.predict_generator(generator=test_generator) # get proba predictions
test_labels = np.argmax(preds,axis = 1)

 
CM = confusion_matrix(test_generator.classes, test_labels )
fig, ax = plot_confusion_matrix(conf_mat=CM  ,  figsize=(5, 5),cmap = "OrRd"  )
plt.show() 
ax = plt.subplot()
sns.set(font_scale=1.0) #edited as suggested
sns.heatmap(CM, annot=True, ax=ax, cmap="Blues", fmt="g");  # annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('Observed labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['False', 'True']);
ax.yaxis.set_ticklabels(['False', 'True']);
plt.show()
plt.figure( figsize = (15,8)) 
    
plt.subplot(221)  
# Accuracy 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# Loss
plt.figure( figsize = (15,8)) 
plt.subplot(222)  
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()