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
import zipfile
import shutil
import tensorflow as tf
import os
import random
from shutil import copyfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir='/tmp/flower_class'
os.mkdir(base_dir)
training_dir=os.path.join(base_dir,'training')
os.mkdir(training_dir)
validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
print(os.listdir(base_dir))
train_dandelion=os.path.join(training_dir,'dandelion')
os.mkdir(train_dandelion)
train_daisy=os.path.join(training_dir,'daisy')
os.mkdir(train_daisy)
train_rose=os.path.join(training_dir,'rose')
os.mkdir(train_rose)
train_tulip=os.path.join(training_dir,'tulip')
os.mkdir(train_tulip)
train_sunflower=os.path.join(training_dir,'sunflower')
os.mkdir(train_sunflower)
print(os.listdir(training_dir))
validation_dandelion=os.path.join(validation_dir,'dandelion')
os.mkdir(validation_dandelion)
validation_daisy=os.path.join(validation_dir,'daisy')
os.mkdir(validation_daisy)
validation_rose=os.path.join(validation_dir,'rose')
os.mkdir(validation_rose)
validation_tulip=os.path.join(validation_dir,'tulip')
os.mkdir(validation_tulip)
validation_sunflower=os.path.join(validation_dir,'sunflower')
os.mkdir(validation_sunflower)
print(os.listdir(validation_dir))

flower_dandelion_source='../input/flowers-recognition/flowers/dandelion'
flower_daisy_source='../input/flowers-recognition/flowers/daisy'
flower_rose_source='../input/flowers-recognition/flowers/rose'
flower_tulip_source='../input/flowers-recognition/flowers/tulip'
flower_sunflower_source='../input/flowers-recognition/flowers/sunflower'
print('total no. of tulips are',len(os.listdir(flower_tulip_source)))
print('total no. of roses are',len(os.listdir(flower_rose_source)))
print('total no. of daisies are',len(os.listdir(flower_daisy_source)))
print('total no. of sunflowers are',len(os.listdir(flower_sunflower_source)))
print('total no. of dandelions are',len(os.listdir(flower_dandelion_source)))
tulip_files=os.listdir(flower_tulip_source)
print(tulip_files[3:8])
a=0
def split_data(source,training,testing,split_size):

    listall=os.listdir(source)
    #print(len(listall))
    
    for i in listall:
        
        if  os.path.getsize(os.path.join(source,i)) !=0:
            a=2
        else:
            listall.pop(i)

    listall=random.sample(listall, len(listall))
    #print(len(listall))
    b=len(listall)*split_size
    b=round(b)
    #print(b)
    trainall=listall[0:b]
    valall=listall[b:]
    #print(len(valall))
    [copyfile(source + '/' + i, training + '/'+i) for i in trainall]
    [copyfile(source + '/'+i, testing +'/'+i) for i in valall]
    #for i in listall:
        #if listall.index(i)<b:
            #copyfile(source + '/' + i, training + '/'+i)
        #else:
             #copyfile(source + '/'+i, testing +'/'+i)
split_size=0.9            
split_data(flower_dandelion_source,train_dandelion,validation_dandelion,split_size)                
split_data(flower_daisy_source,train_daisy,validation_daisy,split_size)                
split_data(flower_rose_source,train_rose,validation_rose,split_size)                
split_data(flower_tulip_source,train_tulip,validation_tulip,split_size)                
split_data(flower_sunflower_source,train_sunflower,validation_sunflower,split_size)                
#print(len(os.listdir(validation_sunflower)))
#print(len(os.listdir(train_sunflower)))
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(96, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(96, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
   
    tf.keras.layers.Flatten(), 
    
    tf.keras.layers.Dense(512, activation='relu'), 
   
    tf.keras.layers.Dense(5, activation='softmax') 
])
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])
model.summary()
TRAINING_DIR =training_dir 
train_datagen =ImageDataGenerator(rescale=1./255,
                                 rotation_range=10,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.1,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=14,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
VALIDATION_DIR = validation_dir
validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator =validation_generator =validation_datagen.flow_from_directory(VALIDATION_DIR ,
                                                    batch_size=15,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')