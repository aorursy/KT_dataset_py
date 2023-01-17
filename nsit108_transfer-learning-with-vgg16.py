# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
print(tf.__version__)
 

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
import tensorflow as tf
import keras
from skimage.io import imread
 


# Networks
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetLarge, NASNetMobile
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

# Setting seeds for reproducibility
seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)
NUM_CLASSES = 2
BATCH_SIZE = 64
IMAGE_SIZE = 200
WIDTH = IMAGE_SIZE
HEIGHT = IMAGE_SIZE
 


LABELS = ['NORMAL', 'PNEUMONIA']

EPOCHS = 8

Metrics = ['accuracy', tf.keras.metrics.AUC(),tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]

train_dir = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/train'
test_dir = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/test'
val_dir = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/val'
train_normal = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/train/NORMAL'
train_pneumonia = '/kaggle/input/chest-xray-pneumo-res/chest_xray_updated_jan20/train/PNEUMONIA'
# Get the list of all the images
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
#resnet_weights_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
#'../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

vgg16_weights_path = '/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, epsilon=0.0001, patience=1, verbose=1)


K.clear_session()


base_model = VGG16(weights=vgg16_weights_path, include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
 

x = base_model.output
x = Flatten()(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)

VGG16_model = Model(inputs=base_model.input, outputs=x)

 

for layer in VGG16_model.layers[0:18]:
    layer.trainable = False
    
 # Create the loss function object using the wrapper function abov
optimizer = keras.optimizers.Adam(lr=0.0001)


VGG16_model.compile(loss='categorical_crossentropy',    
              optimizer=optimizer, 
              metrics=Metrics)


VGG16_model.summary()

start_time = datetime.datetime.now();


VGG16_model.fit_generator(
       train_generator, # specify where model gets training data
       epochs = EPOCHS,
       steps_per_epoch=STEPS,callbacks=[lr_reduce] ,
       validation_data=validation_generator) # specify where model gets validation data


# Evaluate the model
scores = VGG16_model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (VGG16_model.metrics_names[1], scores[1]*100))
end_time = datetime.datetime.now()
elapsed_time = end_time - start_time

print('Start Time  = {} ; End Time  = {} ; Total Time = {}'.format(start_time,end_time,elapsed_time))
test_generator.reset()
preds = VGG16_model.predict_generator(generator=test_generator) # get proba predictions
test_labels = np.argmax(preds,axis = 1)

 
plt.figure( figsize = (15,8)) 
    
plt.subplot(221)  
# Accuracy 
plt.plot(VGG16_model.history.history['accuracy'])
plt.plot(VGG16_model.history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()


# Loss
plt.figure( figsize = (15,8)) 
plt.subplot(222)  
plt.plot(VGG16_model.history.history['loss'])
plt.plot(VGG16_model.history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()
print("\n%s: %.2f%%" % (VGG16_model.metrics_names[0], scores[0]*100))
print("\n%s: %.2f%%" % (VGG16_model.metrics_names[1], scores[1]*100))
print("\n%s: %.2f%%" % (VGG16_model.metrics_names[2], scores[2]*100))
print("\n%s: %.2f%%" % (VGG16_model.metrics_names[3], scores[3]*100))
print("\n%s: %.2f%%" % (VGG16_model.metrics_names[4], scores[4]*100))
import pickle

pickle.dump( VGG16_model,open('Vgg16_model.pkl','wb'))