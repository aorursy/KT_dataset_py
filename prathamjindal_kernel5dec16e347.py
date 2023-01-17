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
        pass

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
csv_files=[i for i in os.listdir("../input/digital-meter/Datasets") if i.endswith(".csv")]
csv_files
dataset=[]
for i in csv_files:
    data=pd.read_csv("../input/digital-meter/Datasets/"+ i,sep=";",index_col=0)
    dataset.append(data)
dataset=pd.concat(dataset,axis=0)
dataset=dataset.replace("X",10)
#dataset
import cv2
image_arr=[]
for i in dataset["image"]:
    
    if i in os.listdir("../input/digital-meter/Datasets/HQ_digital/"):
        img=cv2.imread("../input/digital-meter/Datasets/HQ_digital/"+i)
    elif i in os.listdir("../input/digital-meter/Datasets/LQ_digital/"):
        img=cv2.imread("../input/digital-meter/Datasets/LQ_digital/"+i)
    else:
        img=cv2.imread("../input/digital-meter/Datasets/MQ_digital/"+i)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(144,256))

    img=img/255
    #print(img.shape)
    image_arr.append(np.asarray(img))
image_arr=np.array(image_arr)

digits=dataset[["cadran_1","cadran_2","cadran_3","cadran_4"]]
digits=np.asarray(digits).astype("int")
image_arr=image_arr.reshape(image_arr.shape[0],image_arr.shape[1],image_arr.shape[2],1)
image_arr
digits
from keras.models import Model
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,Flatten
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator

minput=Input((144,256,1))
x = Conv2D(32,(3,3),activation="relu")(minput)
x= MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(0.3)(x)
x=Conv2D(64,(3,3),activation="relu")(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(0.3)(x)
x=Conv2D(128,(3,3),activation="relu")(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(0.3)(x)
x=Flatten()(x)
x=Dense(256,activation="relu")(x)
#x=Dense(128,activation="relu")(x)
d1=Dense(units=11,activation="softmax")(x)
d2=Dense(units=11,activation="softmax")(x)
d3=Dense(units=11,activation="softmax")(x)
d4=Dense(units=11,activation="softmax")(x)
oput=[d1,d2,d3,d4]
model=Model(inputs=minput,outputs=oput)
model.summary()
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(image_arr,digits,test_size=0.2)
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.2,
                             rotation_range=10)
dataGen.fit(X_train)
training=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=32),epochs=50,verbose=1,validation_data=(X_test,y_test))