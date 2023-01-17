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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import cv2

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

# filter warnings

warnings.filterwarnings('ignore')





import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
np.random.seed(1234)

directory="../input/fruits/fruits-360_dataset/fruits-360/Training/"

classes=["Apple Golden 1","Avocado","Banana","Cherry 1","Cocos","Kiwi",

         "Lemon","Mango","Orange"]



all_arrays=[]

img_size=100

for i in classes:

    path=os.path.join(directory,i)

    class_num=classes.index(i)

    for img in os.listdir(path):

        img_array=cv2.imread(os.path.join(path,img),

                             cv2.IMREAD_GRAYSCALE)

        img_array=cv2.resize(img_array,(img_size,img_size))

        all_arrays.append([img_array,class_num])

directory2="../input/fruits/fruits-360_dataset/fruits-360/Test/"

classes2=["Apple Golden 1","Avocado","Banana","Cherry 1","Cocos","Kiwi",

         "Lemon","Mango","Orange"]



all_arrays2=[]

img_size=100

for i in classes2:

    path=os.path.join(directory2,i)

    class_num2=classes2.index(i)

    for img in os.listdir(path):

        img_array2=cv2.imread(os.path.join(path,img),

                             cv2.IMREAD_GRAYSCALE)

        img_array2=cv2.resize(img_array2,(img_size,img_size))

        all_arrays2.append([img_array2,class_num2])
fruits_array_train=[]

for features,label in all_arrays:

    fruits_array_train.append(features)



location=[[1,500,1150],[1500,2000,2500],[3000,3500,4000]]

fruit_names=["Apple","Avocado","Banana","Cherry","Cocos","Kiwi","Lemon","Mango","Orange"]

a=0

b=1

c=2

for i,j,k in location:

    plt.subplots(figsize=(8,8))

    plt.subplot(1,3,1)

    plt.imshow(fruits_array_train[i])

    plt.title(fruit_names[a])

    plt.axis("off")

    plt.subplot(1,3,2)

    plt.imshow(fruits_array_train[j])

    plt.title(fruit_names[b])

    plt.axis("off")

    plt.subplot(1,3,3)

    plt.imshow(fruits_array_train[k])

    plt.title(fruit_names[c])

    plt.axis("off")

    a+=3

    b+=3

    c+=3
import random

random.shuffle(all_arrays)



X_train=[]

Y_train=[]

for features,label in all_arrays:

    X_train.append(features)

    Y_train.append(label)

X_train=np.array(X_train) #arraying



import random

random.shuffle(all_arrays2)



X_test=[]

Y_test=[]

for features,label in all_arrays2:

    X_test.append(features)

    Y_test.append(label)

X_test=np.array(X_test) #arraying
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





model=Sequential()

model.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation="relu",input_shape=(100,100,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=16,kernel_size=(4,4),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters=32,kernel_size=(4,4),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(9,activation="softmax"))

#defining optimizer

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

#compile the model

model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])



epochs=40

batch_size=36
datagen=ImageDataGenerator(featurewise_center=False, #set input mean to 0

                           samplewise_center=False,  #set each sample mean to 0

                           featurewise_std_normalization=False, #divide input datas to std

                           samplewise_std_normalization=False,  #divide each datas to own std

                           zca_whitening=False,  #dimension reduction

                           rotation_range=0.5,    #rotate 5 degree

                           zoom_range=0.5,        #zoom in-out 5%

                           width_shift_range=0.5, #shift 5%

                           height_shift_range=0.5,

                           horizontal_flip=False,  #randomly flip images

                           vertical_flip=False,

                           )

datagen.fit(x_train)



#model fitting

history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),epochs=epochs,

                            validation_data=(x_val,y_val),steps_per_epoch=x_train.shape[0]//batch_size

                           )