import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
#trainpath=("D:/machine learning/competition/intel image processing/seg_train/seg_train/")
#testpath=("D:/machine learning/competition/intel image processing/seg_test/seg_test/")
#predpath=("D:/machine learning/competition/intel image processing/seg_pred/seg_pred/")
trainpath = '../input/intel-image-classification/seg_train/seg_train/'
testpath = '../input/intel-image-classification/seg_test/seg_test/'
predpath = '../input/intel-image-classification/seg_pred/seg_pred/'

for folder in os.listdir(trainpath ):
    files=gb.glob(pathname=str(trainpath + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')
    #print("for training data found {} in {} ".format (len(files),folder))              
    
for folder in os.listdir(testpath ):
    files=gb.glob(pathname=str(testpath + folder + '/*.jpg'))
    print(f'For test data , found {len(files)} in folder {folder}')
    #print("for training data found {} in {} ".format (len(files),folder))              
    
files=gb.glob(pathname=str(predpath + '/*.jpg'))
print(f'For pred data , found {len(files)} ')
    #print("for training data found {} in {} ".format (len(files),folder))   
size=[]
for folder in os.listdir(trainpath ):
    files=gb.glob(pathname=str(trainpath + folder + '/*.jpg'))
    for file in files:
        image=plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()
size=[]
for folder in os.listdir(testpath ):
    files=gb.glob(pathname=str(testpath + folder + '/*.jpg'))
    for file in files:
        image=plt.imread(file)
        size.append(image.shape)

pd.Series(size).value_counts()
code = {'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x
s=100
X_train=[]
y_train=[]
for folder in os.listdir(trainpath ):
    files=gb.glob(pathname=str(trainpath + folder + '/*.jpg'))
    for file in files:
        image=cv2.imread(file)
        image_array=cv2.resize(image,(s,s))
        X_train.append(list(image_array))
        y_train.append(code[folder])
print(f'we have {len(X_train)} in X_train')
X_test=[]
y_test=[]
for folder in os.listdir(testpath):
    files=gb.glob(pathname=str(testpath+folder+'/*.jpg'))
    for file in files:
        image=cv2.imread(file)
        image_array=cv2.resize(image,(s,s))
        X_test.append(list(image_array))
        y_test.append(code[folder])
plt.figure(figsize=(20,20))
for n,i in enumerate (list(np.random.randint(0, len(X_train),36))):
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])
    plt.axis("off")
    plt.title (getcode(y_train[i]))
    
    
X_pred=[]
files = gb.glob(pathname= str(predpath + '/*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    X_pred.append(list(image_array))
len(X_pred)
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
y_test=np.array(y_test)
X_pred=np.array(X_pred)

print(f"X_train shape is {X_train.shape}")
print(f"X_test shape is {X_train.shape}")
print(f"y_train shape is {X_train.shape}")
print(f"y_train shape is {X_train.shape}")
KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(s,s,3)),
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
KerasModel.compile(optimizer ='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(KerasModel.summary())
epochs = 40
ThisModel = KerasModel.fit(X_train, y_train, epochs=epochs,batch_size=64,verbose=1)
ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)
print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))
y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))
y_result=KerasModel.predict(X_pred)
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    plt.title(getcode(np.argmax(y_result[i])))
