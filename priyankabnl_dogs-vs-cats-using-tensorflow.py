# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from zipfile import ZipFile

from PIL import Image

import random

import cv2

import glob

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile
file_name = "/kaggle/input/dogs-vs-cats/train.zip"

file_test = "/kaggle/input/dogs-vs-cats/test1.zip"



with ZipFile(file_name, 'r') as zip:  

    # extracting all the files 

    print('Extracting all the files now...') 

    zip.extractall('/train')

    print('Done!') 

    

with ZipFile(file_test,'r') as zip:

    print('Extracting all files now...')

    zip.extractall('/test1')

    print('Done!')
print(len(os.listdir('/train/train/')))

print(len(os.listdir('/test1/test1/')))
main_dir = "/kaggle/input/dogs-vs-cats"

train_dir = "/train/train"

test_dir = "/test1/test1"

path_train = os.path.join(main_dir, train_dir)

path_test = os.path.join(main_dir, test_dir)

path_array = []

label=[]



for file in os.listdir(path_train):

    path_array.append(os.path.join(path_train,file))

    if file.startswith("cat"):

        label.append('cat')

    elif file.startswith("dog"):

        label.append('dog') 

    
print(path_array[:5])

print(label[:5])
d = {'path': path_array, 'label': label}

df_train = pd.DataFrame(data=d)

df_train.head()
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(96,96,3)),

                                    tf.keras.layers.MaxPooling2D(2,2),

                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2,2),

                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

                                    tf.keras.layers.MaxPooling2D(2,2),

                                    tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(512, activation='relu'),

                                    tf.keras.layers.Dense(2, activation='sigmoid')

])

model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

model.summary()
IMAGE_HT_WID = 96

train_datagen = ImageDataGenerator(validation_split=0.1 ,rescale = 1.0/255.)

train_generator = train_datagen.flow_from_dataframe(dataframe= df_train,x_col='path',y_col='label',subset="training",batch_size=50,seed=42,shuffle=True, class_mode= 'categorical', target_size = (IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator = train_datagen.flow_from_dataframe(dataframe= df_train,x_col='path',y_col='label',subset="validation",batch_size=50,seed=42,shuffle=True, class_mode= 'categorical', target_size = (IMAGE_HT_WID,IMAGE_HT_WID))
EPOCHS=5

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    verbose=1

)
test_data=[]

id=[]

counter=0

for file in os.listdir(path_test):

    test_data.append(os.path.join(path_test,file))

    

print(test_data[:5])



dtest = {'path': test_data}

df_test = pd.DataFrame(data=dtest)

df_test.head()
test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator=test_datagen.flow_from_dataframe(

                dataframe=df_test,

                x_col="path",

                y_col=None,

                batch_size=50,seed=42,

                shuffle=False,

                class_mode=None,

                target_size=(96,96))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()

pred=model.predict_generator(test_generator,

                steps=STEP_SIZE_TEST,

                verbose=1)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()



plt.show()
predicted_class_indices=np.argmax(pred,axis=1)
len(predicted_class_indices)
predicted_class_indices[:10]
len(predicted_class_indices)
len(df_test)
id = [*range(1,len(df_test)+1)]

dataframe_output=pd.DataFrame({"id":id})

dataframe_output["label"]=predicted_class_indices

dataframe_output.to_csv("submission.csv",index=False)
dataframe_output.head()