import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import glob as gb
import cv2
from keras.models import Sequential
import keras
width = 100
hight = 100
image_size = (width , hight)
image_chanels = 3
trainpath = '../input/cat-and-dog/training_set/'
testpath = '../input/cat-and-dog/test_set/'
for folder in  os.listdir(trainpath + 'training_set') : 
    files = gb.glob(pathname= str( trainpath +'training_set//' + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')
for folder in  os.listdir(testpath + 'test_set') : 
    files = gb.glob(pathname= str( testpath +'test_set//' + folder + '/*.jpg'))
    print(f'For Testing data , found {len(files)} in folder {folder}')
size = []
for folder in  os.listdir(trainpath +'training_set') : 
    files = gb.glob(pathname= str( trainpath +'training_set//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()
size = []
for folder in  os.listdir(testpath +'test_set') : 
    files = gb.glob(pathname= str( testpath +'test_set//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()
code = {"cats" : 0 , "dogs" : 1 }

def getcode(n):
    for x , y in code.items():
        if n == y :
            return x
X_train = []
y_train = []
for folder in  os.listdir(trainpath +'training_set') : 
    files = gb.glob(pathname= str( trainpath +'training_set//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (hight,width))
        X_train.append(list(image_array))
        y_train.append(code[folder])
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))
X_test = []
y_test = []
for folder in  os.listdir(testpath +'test_set') : 
    files = gb.glob(pathname= str(testpath + 'test_set//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (hight,width))
        X_test.append(list(image_array))
        y_test.append(code[folder])

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(getcode(y_test[i]))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')
model = Sequential( [
    
        keras.layers.Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(hight,width,3)),
        keras.layers.MaxPool2D(5 , 5),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.50) ,            

        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu'),
                keras.layers.MaxPool2D(3 , 3),

        keras.layers.Dropout(rate=0.3) ,            

        keras.layers.Conv2D(150,kernel_size=(3,3),activation='relu'),
            keras.layers.BatchNormalization(),

        keras.layers.MaxPool2D(3 , 3),

        keras.layers.Dropout(rate=0.2) ,            

        keras.layers.Flatten() ,       
        keras.layers.Dense(120,activation='relu') , 
        keras.layers.Dense(100,activation='relu') ,        

        keras.layers.Dropout(rate=0.5) ,            
        keras.layers.Dense(2,activation='softmax') ,  
    
        
]) 
model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print("Modle Sumarry " , model.summary())
model.fit(X_train, y_train, epochs=40,batch_size=64,verbose=1)

ModelLoss, ModelAccuracy = model.evaluate(X_test, y_test)
print("ModelLoss is : " , ModelLoss)
print("ModelAccuracy is : " , ModelAccuracy )

