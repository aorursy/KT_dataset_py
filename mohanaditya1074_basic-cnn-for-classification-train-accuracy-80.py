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
import cv2

import numpy as np

import os

from random import shuffle
train_directory='../input/chest_xray/chest_xray/train'

test_directory='../input/chest_xray/chest_xray/test'

img_size=100

lr=1e-3
def create_train_data():

    test_data=[]

    

    for folder in os.listdir(train_directory):

        

        if folder!='.DS_Store':

        

            path_folder=os.path.join(train_directory,folder)

            label= label_f(folder)#write a function which returns 1 or 0

            

            





            for img in os.listdir(path_folder):

                if img!='.DS_Store':

                    

                    path=os.path.join(path_folder,img)



                    test_sample=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))



                    test_data.append([np.array(test_sample),(label)])

                

        

    

    

    

    shuffle(test_data)

    

    return test_data
def label_f(folder):

    if folder=="NORMAL": return 1

    elif folder=="PNEUMONIA": return 0

    
def create_test_data():

    test_data=[]

    

    for folder in os.listdir(test_directory):

       

        if folder!='.DS_Store':

        

            path_folder=os.path.join(test_directory,folder)

            label= label_f(folder)#write a function which returns 1 or 0

            

           

            

           





            for img in os.listdir(path_folder):

                

               





                path=os.path.join(path_folder,img)



                test_sample=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))



                test_data.append([np.array(test_sample),(label)])

                

                

        

   

    

    return test_data
test_data=create_test_data()
train_data=create_train_data()
train_x=np.array([i[0] for i in train_data]).reshape(-1, img_size, img_size,1)

train_y=[i[1] for i in train_data]





test_x=np.array([i[0] for i in test_data]).reshape(-1, img_size, img_size,1)

test_y=[i[1] for i in test_data]





train_x,test_x=train_x/255.0,test_x/255.0
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation, Flatten,Dense,BatchNormalization,Dropout
model=Sequential()



model.add(Conv2D(64,(5,5),input_shape=train_x.shape[1:]))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Conv2D(64,(3,3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(64))



model.add(Dense(1))

model.add(Activation("sigmoid"))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



history_model=model.fit(train_x,train_y,batch_size=32,validation_split=0.1,epochs=10)
import matplotlib.pyplot as plt
train_loss=history_model.history["loss"]

val_loss=history_model.history["val_loss"]



epoch_count=range(1,len(val_loss)+1)
plt.plot(epoch_count,train_loss,'r--')

plt.plot(epoch_count,val_loss,'b-')

plt.legend(["train loss","val loss"])

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()
model.evaluate(test_x, test_y,  verbose=1)
model1=Sequential()



model1.add(Conv2D(64,(5,5),input_shape=train_x.shape[1:]))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(Activation("relu"))

model1.add(MaxPooling2D(pool_size=(2,2)))

#model1.add(Dropout(0.5, noise_shape=None, seed=None))



model1.add(Conv2D(64,(3,3)))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(Activation("relu"))

model1.add(MaxPooling2D(pool_size=(2,2)))

#model1.add(Dropout(0.5, noise_shape=None, seed=None))



model1.add(Conv2D(96,(3,3)))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(Activation("relu"))

model1.add(MaxPooling2D(pool_size=(2,2)))

#model1.add(Dropout(0.3, noise_shape=None, seed=None))



model1.add(Conv2D(96,(3,3)))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(Activation("relu"))

model1.add(MaxPooling2D(pool_size=(2,2)))

#model1.add(Dropout(0.3, noise_shape=None, seed=None))





model1.add(Conv2D(128,(3,3)))

model1.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))

model1.add(Activation("relu"))

model1.add(MaxPooling2D(pool_size=(2,2)))

#model1.add(Dropout(0.3, noise_shape=None, seed=None))



model1.add(Flatten())



model1.add(Dense(64))



model1.add(Dense(1))

model1.add(Activation("sigmoid"))



model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model1.fit(train_x,train_y,batch_size=32,validation_split=0.1,epochs=8)
model1.evaluate(test_x, test_y,  verbose=1)

train_loss=history.history["loss"]

val_loss=history.history["val_loss"]



epoch_count=range(1,len(val_loss)+1)
plt.plot(epoch_count,train_loss,'r--')

plt.plot(epoch_count,val_loss,'b-')

plt.legend(["train loss","val loss"])

plt.xlabel("epoch")

plt.ylabel("loss")

plt.show()