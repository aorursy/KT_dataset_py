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
import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from keras.applications import VGG16

import cv2

import os

import random

import tensorflow as tf

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os

print(os.listdir('../input/chest-xray-pneumonia/chest_xray/train'))
labels = ['NORMAL', 'PNEUMONIA']

img_size = 150

def get_data(data_dir):

    data = [] 

    for label in labels: 

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for img in os.listdir(path):

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size

                data.append([resized_arr, class_num])

            except Exception as e:

                print(e)

    return np.array(data)
data = get_data("/kaggle/input/chest-xray-pneumonia/chest_xray/train")
l = []

for i in data:

    l.append(labels[i[1]])

sns.set_style('darkgrid')

sns.countplot(l)

fig,ax=plt.subplots(3,3)

fig.set_size_inches(15,15)

for i in range(3):

    for j in range (3):

        l=random.randint(0,len(data))

        ax[i,j].imshow(data[l][0])

        ax[i,j].set_title('Case: '+labels[data[l][1]])

        

plt.tight_layout()
x = []

y = []



for feature, label in data:

    x.append(feature)

    y.append(label)
# Normalize the data

x = np.array(x) / 255
# Reshaping the data from 1-D to 3-D as required through input by CNN's 

x = x.reshape(-1, img_size, img_size, 3)

y = np.array(y)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

y = label_binarizer.fit_transform(y)
model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(rate=0.3))



model.add(Conv2D(filters =32, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(Dense(256))

model.add(Dropout(rate=0.2))

model.add(Activation('relu'))

model.add(Dense(2, activation = "softmax"))

model.summary()
batch_size=128

epochs=5



from keras.callbacks import ReduceLROnPlateau

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x)
#di lw classification bin 2

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

data2 = get_data("/kaggle/input/chest-xray-pneumonia/chest_xray/test")
x1 = []

y2 = []



for feature, label in data2:

    x1.append(feature)

    y2.append(label)
# Normalize the data

x1 = np.array(x1) / 255
# Reshaping the data from 1-D to 3-D as required through input by CNN's 

x1 = x1.reshape(-1, img_size, img_size, 3)

y2 = np.array(y2)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

y2 = label_binarizer.fit_transform(y2)
History=model.fit(x,y, batch_size=batch_size,epochs=5)
print("Loss of the model is - " , model.evaluate(x1,y2)[0] , "%")

print("Accuracy of the model is - " , model.evaluate(x1,y2)[1]*100 , "%")
predictions = model.predict_classes(x1)

print(predictions[:5])

print("prediction",predictions.shape)
code={"NORMAL":0,"PNEUMONIA":1}

def getcode (n) :

    for x,y in code.items():

        if n==y:

            return x
plt.figure(figsize=(20,20))

for n,i in enumerate(list(np.random.randint(0,len(x1),36))):

    plt.subplot(6,6,n+1)

    plt.imshow(x1[i])

    plt.axis("off")

    plt.title(getcode(predictions[i]))
y_test_inv = label_binarizer.inverse_transform(y2)
print(classification_report(y_test_inv, predictions, target_names = labels))
cm = confusion_matrix(y_test_inv,predictions)

cm
cm = pd.DataFrame(cm , index = labels , columns = labels)
plt.figure(figsize = (10,10))

sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = labels , yticklabels = labels)
# now storing some properly as well as misclassified indexes'.

i=0

prop_class=[]

mis_class=[]



for i in range(len(y_test_inv)):

    if(y_test_inv[i] == predictions[i]):

        prop_class.append(i)

    if(len(prop_class)==8):

        break



i=0

for i in range(len(y_test_inv)):

    if(y_test_inv[i] != predictions[i]):

        mis_class.append(i)

    if(len(mis_class)==8):

        break
count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x1[prop_class[count]])

        ax[i,j].set_title("Predicted case : "+ labels[predictions[prop_class[count]]] +"\n"+"Actual case : "+ labels[y_test_inv[prop_class[count]]])

        plt.tight_layout()

        count+=1

count=0

fig,ax=plt.subplots(4,2)

fig.set_size_inches(15,15)

for i in range (4):

    for j in range (2):

        ax[i,j].imshow(x1[mis_class[count]])

        ax[i,j].set_title("Predicted case : "+labels[predictions[mis_class[count]]]+"\n"+"Actual case : "+labels[y_test_inv[mis_class[count]]])

        plt.tight_layout()

        count+=1