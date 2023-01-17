data_dir = '/kaggle/input/uecfood100/UECFOOD100/'
import os

dirNames = os.listdir(data_dir)



for dirName in dirNames:

    if 'txt' in dirName:

        print(dirName)

        dirNames.remove(dirName)

        

print(dirNames)

print(len(dirNames))
import pandas as pd

df = pd.read_csv(data_dir+'category.txt', sep='\t')

print(df)



labels = df['name'].tolist()
trainFiles = []

trainClasses = []



for dirName in dirNames:

    for file in os.listdir(data_dir+dirName):

        trainFiles.append(data_dir+dirName+"/"+file)

        trainClasses.append(dirName)



print(len(trainFiles), len(trainClasses))
import matplotlib.pyplot as plt

import matplotlib.image as mpimg



plt.imshow(mpimg.imread(trainFiles[0]))
from collections import Counter

import seaborn as sns

sns.set_style("whitegrid")



def plot_equilibre(categories, counts):



    plt.figure(figsize=(12, 8))



    sns_bar = sns.barplot(x=categories, y=counts)

    sns_bar.set_xticklabels(categories, rotation=45)

    plt.title('Equilibre of Training Dataset')

    plt.show()
categories = dirNames

counts = []

[counts.append(trainClasses.count(dirName)) for dirName in dirNames]



plot_equilibre(categories, counts)
!pip install split-folders
import splitfolders

splitfolders.ratio('/kaggle/input/uecfood100/UECFOOD100', output='dataset', seed=1234, ratio=(0.8, 0.1, 0.1) )
import numpy as np

import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical
target_size=(224,224)

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



train_generator = train_datagen.flow_from_directory(

    'dataset/train',

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',    

    shuffle=True,

    seed=42,

    class_mode='categorical')
valid_datagen = ImageDataGenerator(rescale=1./255)



valid_generator = valid_datagen.flow_from_directory(

    'dataset/val',

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,    

    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(

    'dataset/test',

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,     

    class_mode='categorical')
import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Model, save_model

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout 

from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ModelCheckpoint
!pip install -q efficientnet

import efficientnet.tfkeras as efn
num_classes = 100

input_shape = (224,224,3)
# load EfficientNetB7 model with imagenet parameteres

base_model = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)
# freeze the base model (for transfer learning)

base_model.trainable = False
# add two FC layers (with L2 regularization)

x = base_model.output

x = GlobalAveragePooling2D()(x) #2560

#x = BatchNormalization()(x)



#x = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)

x = Dense(512)(x)

#x = Dropout(0.2)(x)



# Output layer

out = Dense(num_classes, activation="softmax")(x)



model = Model(inputs=base_model.input, outputs=out)

model.summary()
# Compile Model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
## set Checkpoint : save best only, verbose on

#checkpoint = ModelCheckpoint("food100_classification.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST =test_generator.n//test_generator.batch_size

num_epochs = 100
# Train Model

history = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=num_epochs, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID) #, callbacks=[checkpoint])
## Save Model

save_model(model, 'food100_efficientnetB7.h5')
## load best model weights if using callback (save-best-only)

#model.load_weights("food100_classification.hdf5")
score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)

print(score)
from sklearn.metrics import classification_report, confusion_matrix



predY=model.predict_generator(test_generator)

y_pred = np.argmax(predY,axis=1)

#y_label= [labels[k] for k in y_pred]

y_actual = test_generator.classes

cm = confusion_matrix(y_actual, y_pred)

print(cm)
print(classification_report(y_actual, y_pred, target_names=labels))