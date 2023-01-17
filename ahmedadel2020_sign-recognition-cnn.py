# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv3D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.optimizers import SGD


from sklearn.model_selection import train_test_split
import cv2
import glob
import random
import os
def read_images():
    image_list = []
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            i=int(dirname[-1])
            path=os.path.join(dirname, filename)
            im=cv2.imread(path, cv2.COLOR_BGR2RGB)
            im_list=im.tolist()
            im_list={'image':im_list,'lable':i}
            image_list.append(im_list)       
    list_images=[np.array(image_item['image']) for image_item in image_list]
    mean_image=np.mean(list_images,axis=0)
    for i in range(0,len(image_list)):
        image_list[i]['image']=((np.array(image_list[i]['image']) - mean_image) / 255).tolist()
    random.shuffle(image_list)
    random.shuffle(image_list)
    images=[image_item['image'] for image_item in image_list]
    y = [image_item['lable'] for image_item in image_list]
    return {'images':images,'y':y}

data_set=read_images()
X_train, X_test, y_train, y_test = train_test_split(data_set['images'],data_set['y'] , test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print(y_train)
print('done!!')
# Any results you write to the current directory are saved as output.
from keras.layers import Conv2D,MaxPooling2D

def Arch1CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 100,3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def Arch2CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 100,3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
def Arch3CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 100,3)))
    model.add(AveragePooling2D((3,3)))
    model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
def Arch4CNN():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 100,3)))
    model.add(Conv2D(40, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(Conv2D(40, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(Conv2D(40, (3, 3), activation='relu',kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
model=Arch4CNN()
X_train_instance=np.array(X_train)
X_train_instance=X_train_instance.reshape(X_train_instance.shape[0],100,100,3)
# X_val=np.array(X_val)
# X_val=X_val.reshape(X_val.shape[0],100,100,3)
history = model.fit(X_train_instance,y_train, validation_data=(X_val,y_val),epochs=200, batch_size=20)

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
def classified(y_pred):
    y_new_pred=[]
    for i in range(0,len(y_pred)):
        y_new_pred.append(np.argmax(y_pred[i]))
    return y_new_pred
        
X_test_instance=np.array(X_test)
X_test_instance=X_test_instance.reshape(X_test_instance.shape[0],100,100,3)
accuracy = model.evaluate(X_test_instance,y_test, verbose=2)
print("accuracy:",accuracy[1]*100)
y_pred= model.predict(X_test_instance)
y_new_pred=classified(y_pred)
#new_y_true,new_y_pred = binary_value(y_test,y_pred.tolist())
recall_acc=recall_score(y_test,y_new_pred,average='weighted')*100
precision_acc=precision_score(y_test,y_new_pred,average='weighted')*100
f1_acc=f1_score(y_test, y_new_pred, average='weighted')*100
print('recall_acc:',recall_acc)
print('precision_acc:',precision_acc)
print('f1_acc:',f1_acc)

