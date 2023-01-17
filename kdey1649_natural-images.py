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
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,Conv2D,MaxPooling2D,BatchNormalization,Dropout,Flatten
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam, RMSprop
import keras
import cv2
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
def preprocessing(x,path,y,value=0):
    for i in os.listdir(path):
        image_path=path+"/"+i
        image=cv2.imread(image_path,0)
        image=cv2.resize(image,(200,200),interpolation=cv2.INTER_CUBIC)
        image=cv2.GaussianBlur(image,(5,5),cv2.BORDER_DEFAULT)
        x.append(image) 
        y.append(value)
def plt_dynamic(x,vy,ty,ax,label1='Validation_loss',label2='Train_loss'):
    ax.plot(x,vy,'b',label=label1)
    ax.plot(x,ty,'r',label=label2)
    plt.legend()
    plt.grid()
    fig.canvas.draw()
patha='../input/natural-images/natural_images/airplane'
pathc='../input/natural-images/natural_images/car'
pathca='../input/natural-images/natural_images/cat'
pathd='../input/natural-images/natural_images/dog'
pathf='../input/natural-images/natural_images/flower'
pathfr='../input/natural-images/natural_images/fruit'
pathm='../input/natural-images/natural_images/motorbike'
pathp='../input/natural-images/natural_images/person'
y=[]
x=[]
preprocessing(x,patha,y,0)
preprocessing(x,pathc,y,1)
preprocessing(x,pathca,y,2)
preprocessing(x,pathd,y,3)
preprocessing(x,pathf,y,4)
preprocessing(x,pathfr,y,5)
preprocessing(x,pathm,y,6)
preprocessing(x,pathp,y,7)

x=np.array(x)
print(x.shape)
y=np.array(y)
print(y.shape)
x,y=shuffle(x,y)
x_1,x_test,y_1,y_test=train_test_split(x,y,test_size=0.3,shuffle=False)
x_tr,x_cv,y_tr,y_cv=train_test_split(x_1,y_1,test_size=0.005,shuffle=False)
for i in range(0,5):
    plt.title(y[i])
    plt.imshow(x_tr[i])
    plt.show()
print(x_cv.shape)
print(x_tr.shape)
print(x_test.shape)
print(y_tr.shape)
print(y_test.shape)
print(y_cv.shape)
x_tr=x_tr/255
x_test=x_test/255
x_cv=x_cv/255
img_rows,img_cols=200,200
if K.image_data_format() == 'channels_first':
    x_tr= x_tr.reshape(x_tr.shape[0],1,img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_cv = x_cv.reshape(x_cv.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_tr= x_tr.reshape(x_tr.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_cv = x_cv.reshape(x_cv.shape[0],img_rows, img_cols,1)
    input_shape = (img_rows, img_cols, 1)
y_tr=keras.utils.to_categorical(y_tr,8)
y_test=keras.utils.to_categorical(y_test,8)
y_cv=keras.utils.to_categorical(y_cv,8)


datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range = 30,  
        zoom_range = 0.2,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip = True,  
        vertical_flip=False)  


datagen.fit(x_tr)
def structure(activation='relu',dropout=0.4,kernel='glorot_uniform'):
    numclasses=8
    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation=activation,input_shape=input_shape,kernel_initializer=kernel))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(3,3),activation=activation,kernel_initializer=kernel))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation=activation,kernel_initializer=kernel))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,(3,3),activation=activation,kernel_initializer=kernel))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation=activation,kernel_initializer=kernel))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128,activation=activation))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(64,activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(numclasses,activation='softmax'))
    return model;  
model=structure()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
history=model.fit(datagen.flow(x_tr,y_tr,batch_size=16),epochs=25,verbose=1,validation_data=datagen.flow(x_cv,y_cv),callbacks = [learning_rate_reduction])
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print("Test accuracy:",score[1])
score=model.evaluate(x_test,y_test,verbose=0)
print('Test_score',score[0])
print('Test_accuracy',score[1])

fig,ax=plt.subplots(1,1)
ax.set_xlabel('epoch')
ax.set_ylabel('categorical_cross_entropy')

x=list(range(1,25+1))
vy=history.history['val_loss']
ty=history.history['loss']
plt_dynamic(x,vy,ty,ax)
fig,ax=plt.subplots(1,1)
ax.set_xlabel('epoch')
ax.set_ylabel('accuracy')

x=list(range(1,25+1))
ty=history.history['accuracy']
vy=history.history['val_accuracy']
plt_dynamic(x,vy,ty,ax,label2='train_accuracy',label1='val_accuracy')
from sklearn.metrics import classification_report,confusion_matrix
prediction=model.predict_classes(x_test)
y_test=np.argmax(y_test,axis=-1)
cm=confusion_matrix(y_test,prediction)
print(cm)
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test,y_test, verbose=0)
print("%s: %.4f%%" % (loaded_model.metrics_names[1], score[1]*100))

