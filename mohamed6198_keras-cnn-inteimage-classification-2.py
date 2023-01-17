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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
trainpath = '../input/intel-image-classification/seg_train/seg_train/'
testpath = '../input/intel-image-classification/seg_test/seg_test/'
predpath = '../input/intel-image-classification/seg_pred/'
for folder in os.listdir(testpath):
    print(folder)
def shapeCount (Gpath):
    size = []
    pathname = str(Gpath+'*/*.jpg')
    imgsDir = gb.glob(pathname= pathname)
    for imgDir in tqdm(imgsDir):
        img = plt.imread(imgDir)
        size.append(img.shape)
    print("number of Imges = ",len(size))
    return pd.Series(size).value_counts()
print(shapeCount(trainpath))
print(shapeCount(testpath))
print(shapeCount(predpath))
IMAGE_SIZE = 100
code = ['buildings','forest','glacier','mountain','sea','street','seg_pred']
def reArangeData(data):
    # shuffle
    import random
    random.shuffle(data)
    
    # get X,Y
    X = []
    y = []
    for img,lable in data:
        X.append(img)
        y.append(lable)
        
    # convert it to npArray
    return np.array(X),np.array(y)
def loadImages (path):
    Dlist = []
    for folder in os.listdir(path):
        pathname = str(path +folder+'/*.jpg')
        files = gb.glob(pathname= pathname)
        for file in tqdm(files):
            image = cv2.imread(file)
            image_array = cv2.resize(image , (IMAGE_SIZE ,IMAGE_SIZE))
            Dlist.append( [image_array , code.index(folder)] )
    print(len(Dlist))
    return reArangeData(Dlist)
X_train , y_train = loadImages(trainpath)
X_test , y_test = loadImages(testpath)
X_predict , _ = loadImages(predpath)
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(X_train[i])
    plt.axis('off')
    plt.title(code[y_train[i]])
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.title(code[y_test[i]])
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(X_predict[i])
    plt.axis('off')
print('X_train shape is : ',X_train.shape)
print('y_train shape is : ',y_train.shape)
print("---------------------------------------------")
print('X_test shape id : ',X_test.shape)
print('y_test shape id : ',y_test.shape)
print("---------------------------------------------")
print('X_predict shape id : ',X_predict.shape)

relu = tf.nn.relu
softmax = tf.nn.softmax
input_shape = (IMAGE_SIZE , IMAGE_SIZE , 3)
loss = keras.losses.sparse_categorical_crossentropy
epochs = 50
model = keras.models.Sequential([
    keras.layers.Conv2D(128,kernel_size=(3,3) ,
                        activation=relu , input_shape = input_shape ),
    keras.layers.Conv2D(256,kernel_size=(3,3) , activation=relu),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Conv2D(128,kernel_size=(3,3),activation=relu),
    keras.layers.Conv2D(64,kernel_size=(3,3),activation=relu),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=relu),
    keras.layers.Dense(128,activation=relu),
    keras.layers.Dense(64,activation=relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(6,activation=softmax)
])

model.compile(optimizer='adam',
             loss = loss ,
             metrics=['accuracy'])
print(model.summary())

ThisModel = model.fit(X_train, y_train, epochs=epochs,batch_size=64,verbose=1)
Err,Acc = model.evaluate(X_test,y_test)
print("Err : " , Err)
print("Acc : ", Acc)
model3 = keras.models.Sequential([
    keras.layers.Conv2D(128,kernel_size=(3,3) ,
                        activation=relu , input_shape = input_shape ),
    keras.layers.Conv2D(256,kernel_size=(3,3) , activation=relu),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Conv2D(128,kernel_size=(3,3),activation=relu),
    keras.layers.Conv2D(128,kernel_size=(3,3),activation=relu),
    keras.layers.Conv2D(64,kernel_size=(3,3),activation=relu),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=relu),
    keras.layers.Dense(64,activation=relu),
    keras.layers.Dense(64,activation=relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(6,activation=softmax)
])


model3.compile(optimizer='adam',
             loss = loss ,
             metrics=['accuracy'])
print(model3.summary())
relu = tf.nn.relu
softmax = tf.nn.softmax
input_shape = (IMAGE_SIZE , IMAGE_SIZE , 3)
loss = keras.losses.sparse_categorical_crossentropy

model3 = keras.models.Sequential([
    keras.layers.Conv2D(128,kernel_size=(3,3) ,
                        activation=relu , input_shape = input_shape ),
    keras.layers.Conv2D(256,kernel_size=(3,3) , activation=relu),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Conv2D(128,kernel_size=(3,3),activation=relu),
    keras.layers.Conv2D(128,kernel_size=(3,3),activation=relu),
    keras.layers.Conv2D(64,kernel_size=(3,3),activation=relu),
    keras.layers.MaxPool2D(4,4),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation=relu),
    keras.layers.Dense(64,activation=relu),
    keras.layers.Dense(64,activation=relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(6,activation=softmax)
])


model3.compile(optimizer='adam',
             loss = loss ,
             metrics=['accuracy'])
print(model3.summary())
ThisModel3 = model3.fit(X_train, y_train, epochs=epochs,batch_size=64,verbose=1)

Err3,Acc3 = model3.evaluate(X_test,y_test)
print("Err : " , Err3)
print("Acc : ", Acc3)
model2 = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=input_shape),
        keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Conv2D(120,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(80,kernel_size=(3,3),activation='relu'),    
        keras.layers.Conv2D(50,kernel_size=(3,3),activation='relu'),
        keras.layers.MaxPool2D(4,4),
        keras.layers.Flatten() ,    
        keras.layers.Dense(120,activation='relu') ,    
        keras.layers.Dense(100,activation='relu') ,    
        keras.layers.Dense(50,activation='relu') ,        
        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(6,activation='softmax') ,    
        ])
model2.compile(optimizer ='adam',loss=loss,metrics=['accuracy'])

epochs = 50
ThisModel2 = model2.fit(X_train, y_train, epochs=epochs,batch_size=64,verbose=1)
Err2,Acc2 = model2.evaluate(X_test,y_test)
print("Err : " , Err2)
print("Acc : ", Acc2)
y_result1 = model.predict(X_predict)
plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.imshow(X_predict[i])
    plt.axis('off')
    plt.title(code[np.argmax(y_result1[i])])

y_result2 = model2.predict(X_predict)
plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.imshow(X_predict[i])
    plt.axis('off')
    plt.title(code[np.argmax(y_result2[i])])
y_result3 = model3.predict(X_predict)
plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.imshow(X_predict[i])
    plt.axis('off')
    plt.title(code[np.argmax(y_result3[i])])