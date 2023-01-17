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
import os
import tensorflow as tf
from tensorflow import keras
import zipfile
from shutil import copyfile
# to delete files or folder
'''import shutil
shutil.rmtree("/kaggle/working/test1")'''
# list files in input directory
print(os.listdir("../input/"))
# extract all training images
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall(".")

# extract all testing images
with zipfile.ZipFile("../input/dogs-vs-cats/test1.zip","r") as z:
    z.extractall(".")
print('Number of training images : ',len(os.listdir("/kaggle/working/train")))
print('Number of testing images : ',len(os.listdir("/kaggle/working/test1")))
os.listdir("/kaggle/working/train")[:3]
os.mkdir("/kaggle/working/training/")
os.mkdir("/kaggle/working/validation/")
os.mkdir("/kaggle/working/training/dogs")
os.mkdir("/kaggle/working/validation/dogs")
os.mkdir("/kaggle/working/training/cats")
os.mkdir("/kaggle/working/validation/cats")
os.listdir('/kaggle/working/training/')
os.listdir('/kaggle/working/validation/')
def shuffle_split(source, split):
    n = len(os.listdir(source))
    inputfiles = np.array(os.listdir(source))
    np.random.shuffle(inputfiles)
    print(len(inputfiles))
    split_index = int(n*split)
    training, validation = inputfiles[:split_index], inputfiles[split_index:]
    print(len(training), len(validation))
    return training, validation
def copy_file(source,training, validation, split):
    tr, va = shuffle_split(source, split) # shuffle and split files into training and validation
    # training
    for filename in tr:
        filename_split = filename.split('.')[0]
        if filename_split == 'cat':
            copyfile(source+filename, training + 'cats/' + filename)
        else:
            copyfile(source+filename, training + 'dogs/' + filename)
    # validation
    for filename in va:
        filename_split = filename.split('.')[0]
        if filename_split == 'cat':
            copyfile(source+filename, validation + 'cats/' + filename)
        else:
            copyfile(source+filename,validation + 'dogs/' + filename)
TRAINING_DIR = '/kaggle/working/training/'
VALIDATION_DIR = '/kaggle/working/validation/'
TRAIN_SOURCE = '/kaggle/working/train/'

copy_file(TRAIN_SOURCE, TRAINING_DIR, VALIDATION_DIR,0.8)
for dir in [TRAINING_DIR,VALIDATION_DIR]:
    for sub in ['cats','dogs']:
        print(sub,len(os.listdir(dir+sub)))
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model

pretrained_model = InceptionV3(input_shape = (150, 150, 3), include_top = False)
for layer in pretrained_model.layers:
    layer.trainable = False # freeze or lock the underlying layers

pretrained_model.summary()
last_layer = pretrained_model.get_layer('mixed7')

last_output = last_layer.output
print(last_output)
from tensorflow.keras.optimizers import RMSprop

# Take the last output layer and pass it through a DNN
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pretrained_model.input, x)
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data augmentation for training images
train_datagen = ImageDataGenerator(	rescale=1./255,
    rotation_range=40, # rotate image
    width_shift_range=0.2, # move image left or right
    height_shift_range=0.2, # move image up or down
    shear_range=0.2, # skew image
    zoom_range=0.2, # zoom in on image
    horizontal_flip=True, # flip image on horizontal
    fill_mode='nearest') # normalize the data

train_generator = train_datagen.flow_from_directory( 
    TRAINING_DIR,
    target_size = (150,150), 
    batch_size = 128, 
    class_mode = 'binary')

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory( 
    VALIDATION_DIR, 
    target_size = (150,150), 
    batch_size = 32, 
    class_mode = 'binary')
from keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(patience = 2)
history = model.fit(train_generator, 
          steps_per_epoch= 20000//128, 
          epochs = 3, 
          validation_data=validation_generator, 
          validation_steps=5000//32, 
          callbacks = [earlystopping])
## Plot the history of our model

import matplotlib.pyplot as plt
# Get the different results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))

# Plot the training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training and validation accuracy")
plt.figure()

# Plot the training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training and validation loss")
plt.figure()
os.mkdir('/kaggle/working/testing/')
os.mkdir('/kaggle/working/testing/cats/')
os.mkdir('/kaggle/working/testing/dogs/')
def testfiles(source, output):
    for filename in os.listdir(source):
        if filename.split('.')[0] == 'cat':
            copyfile(source+filename, output+'cats/'+filename)
        else:
            copyfile(source+filename, output+'dogs/'+filename)
testfiles('/kaggle/working/test1/','/kaggle/working/testing/')
testing_datagen = ImageDataGenerator(rescale = 1./255)

testing_generator = testing_datagen.flow_from_directory( 
    '/kaggle/working/testing/', 
    target_size = (150,150), 
    batch_size = 128, 
    class_mode = 'binary')
model.evaluate(testing_generator)
model.save('./model_inceptions1.h5') 
