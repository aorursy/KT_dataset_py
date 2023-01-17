# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir("../input/10-monkey-species")
train = os.listdir("../input/10-monkey-species/training/training")
validation = os.listdir("../input/10-monkey-species/validation/validation")
os.listdir('../input/10-monkey-species/training/training')
#filenames  = os.listdir('../input/10-monkey-species/training/training/n1')
#sample = random.choice(filenames)

list1 = ["../input/10-monkey-species/training/training/n0/"
, "../input/10-monkey-species/training/training/n1/"
,"../input/10-monkey-species/training/training/n2/"
, "../input/10-monkey-species/training/training/n3/"
, "../input/10-monkey-species/training/training/n4/"
,"../input/10-monkey-species/training/training/n5/"
,"../input/10-monkey-species/training/training/n6/"
,"../input/10-monkey-species/training/training/n7/"
,"../input/10-monkey-species/training/training/n8/",
"../input/10-monkey-species/training/training/n9/"]

fig = plt.figure(figsize=(12, 15))
fig.set_size_inches(13,13)
plt.style.use("ggplot")
j=1
for i in list1:   
    filenames  = os.listdir(i)
    sample = random.choice(filenames)
    image = load_img(i+sample) #'../input/10-monkey-species/training/training/n1'
    plt.subplot(2,5,j)
    plt.imshow(image)
    plt.xlabel("Monkey Species: n{}".format(j))
    j+=1
plt.tight_layout()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tr_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

train_datagen = tr_datagen.flow_from_directory(directory = "../input/10-monkey-species/training/training",target_size=(150,150),
                                            class_mode='categorical',batch_size=60)
test_datagen = ImageDataGenerator(rescale = 1/255.0)
valid_datagen = test_datagen.flow_from_directory(directory = "../input/10-monkey-species/validation/validation",target_size=(150,150),
                                            class_mode='categorical',batch_size=60)
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPool2D,Dropout,BatchNormalization
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu',input_shape=(150,150,3)))
#model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters = 32,kernel_size = (3,3),activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))
model.summary()
from tensorflow.keras.optimizers import SGD
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

checkpoint = ModelCheckpoint("model_1.h5",monitor = 'val_accuracy',verbose=1,save_best_only = True,save_weights_only = False,
                             mode='auto',period=1)
earlystop = EarlyStopping(monitor ='val_acc',patience=20,min_delta = 0,verbose=1,mode='auto')


batch_size = 60
history = model.fit_generator(generator = train_datagen,steps_per_epoch = len(train_datagen),epochs=150,
                              validation_data = valid_datagen,validation_steps =len(valid_datagen) ,
                              callbacks=[checkpoint,earlystop],verbose=1)#train_number//batch_size,valid_number//batch_size
def plot_learning_curve(history,epochs):
    epochs = np.arange(1,epochs+1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs,history.history['accuracy'])
    plt.plot(epochs,history.history['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()
    
    plt.figure(figsize=(10,6)) 
    plt.plot(epochs,history.history['loss'])
    plt.plot(epochs,history.history['val_loss'])
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend(['Train','Val'],loc='upper left')
    plt.show()
    
plot_learning_curve(history,150)
