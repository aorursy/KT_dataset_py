#### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import itertools

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import random

import os,shutil



src_path="../input"



print(os.listdir(src_path))



#constant value

VALID_SPIT=0.2

IMAGE_SIZE=64

BATCH_SIZE=128





# Any results you write to the current direc



# Any results you write to the current directory are saved as output.
label=[]

data=[]

counter=0

path="../input/train/train"

for file in os.listdir(path):

    image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_COLOR)

    image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

    if file.startswith("cat"):

        label.append(0)

    elif file.startswith("dog"):

        label.append(1)

    try:

        data.append(image_data/255)

    except:

        label=label[:len(label)-1]

    counter+=1

    if counter%1000==0:

        print (counter," image data retreived")



data=np.array(data)

data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],3)

label=np.array(label)

print (data.shape)

print (label.shape)
sns.countplot(label)
from sklearn.model_selection import train_test_split

train_data, valid_data, train_label, valid_label = train_test_split(

    data, label, test_size=0.2, random_state=42)

print(train_data.shape)

print(train_label.shape)

print(valid_data.shape)

print(valid_label.shape)
from keras import Sequential

from keras.layers import *

import keras.optimizers as optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import *

import keras.backend as K
from keras import backend as K

K.set_image_dim_ordering('th')

K.set_image_data_format('channels_last')

from keras import layers

from keras import models

from keras import optimizers

from keras.layers import GlobalAveragePooling2D



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(64, 64, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))





model.add(GlobalAveragePooling2D())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(lr=1e-4),metrics=['acc'])

model.summary()
callack_saver = ModelCheckpoint(

            "model.h5"

            , monitor='val_loss'

            , verbose=0

            , save_weights_only=True

            , mode='auto'

            , save_best_only=True

        )



train_history=model.fit(train_data,train_label,validation_data=(valid_data,valid_label),epochs=40,batch_size=BATCH_SIZE, callbacks=[callack_saver])
import matplotlib.pyplot as plt

acc = train_history.history['acc']

val_acc = train_history.history['val_acc']

loss = train_history.history['loss']

val_loss = train_history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'blue', label='Training acc')

plt.plot(epochs, val_acc, 'red', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'blue', label='Training loss')

plt.plot(epochs, val_loss, 'red', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory("../input/test1",target_size=(64, 64),batch_size=32,class_mode='binary')
from tensorflow.python.keras.models import Sequential

from keras.models import load_model



print("-- Evaluate --")



scores = model.evaluate_generator(

            test_generator, 

            steps = 100)



print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

Y_pred = model.predict(valid_data)

predicted_label=np.round(Y_pred,decimals=2)
import matplotlib.pyplot as plt

%matplotlib inline

from mlxtend.plotting import plot_confusion_matrix



# Get the confusion matrix



CM = confusion_matrix(valid_label, Y_pred.round())

fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(12, 12))

plt.xticks(range(2), ['Cat', 'Dog'], fontsize=16)

plt.yticks(range(2), ['Cat', 'Dog'], fontsize=16)

plt.show()
test_data=[]

id=[]

counter=0

for file in os.listdir("../input/test1/test1"):

    image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_COLOR)

    try:

        image_data=cv2.resize(image_data,(IMAGE_SIZE,IMAGE_SIZE))

        test_data.append(image_data/255)

        id.append((file.split("."))[0])

    except:

        print ("ek gaya")

    counter+=1







test_data=np.array(test_data)

print (test_data.shape)

test_data=test_data.reshape((test_data.shape)[0],(test_data.shape)[1],(test_data.shape)[2],3)

dataframe_output=pd.DataFrame({"id":id})
predicted_labels=model.predict(test_data)

predicted_labels=np.round(predicted_labels,decimals=2)

labels=[1 if value>0.5 else 0 for value in predicted_labels]
dataframe_output["label"]=labels

print(dataframe_output)
#0이 고양이 1이 강아지

import matplotlib.pyplot as plt

%matplotlib inline



for i in range(20):

    plt.figure()

    plt.imshow(test_data[i])

#     plt.xlabel('label')

    plt.xlabel(dataframe_output["label"][i])



    plt.show()