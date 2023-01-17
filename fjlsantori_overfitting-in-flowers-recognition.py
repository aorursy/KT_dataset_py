# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.utils.np_utils import to_categorical



flowers_dir = "../input/flowers-recognition/flowers/flowers/"

labels = pd.Categorical(os.listdir(flowers_dir))



print("Labels \n", labels)



from os.path import join, isfile

from os import listdir

flowers_type_path = [join(flowers_dir,flower_type+"/") for flower_type in labels]

print("Paths \n",flowers_type_path)
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



image_size=224

num_classes=5



def resize_flowers(path, img_height=image_size, img_width=image_size):

    label=[]

    flowers=[]

    for i in range(0,len(path)):

        for k in os.listdir(path[i]):

            try:

                img = load_img(path[i]+k,target_size=(img_height, img_width))

                img_array= np.array(img_to_array(img))

                img_output = preprocess_input(img_array)

                flowers.append(img_output)

                label.append(labels[i])

            except:

                None

    return(label, flowers)

etiquetas, imagenes = resize_flowers(flowers_type_path)
from sklearn.preprocessing import LabelEncoder



le=LabelEncoder()

Y=le.fit_transform(etiquetas)

Y=to_categorical(Y,5)

X=np.array(imagenes)

X=X/255
# test split

from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size = 0.25,random_state = 42)



import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
model = Sequential()

#add model layers

model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (image_size,image_size,3)))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

 



model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(num_classes, activation='softmax'))



model.layers[0].trainable = False



model.summary() # print summary my model

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) #compile model
#train the model

batch_size=128

hist = model.fit(x_train,y_train, validation_data=(x_val,y_val), epochs=20, batch_size=batch_size)
score = model.evaluate(x_val,y_val)

score
hist.history

import matplotlib.pyplot as plt



%matplotlib inline



plt.figure

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'],'r')