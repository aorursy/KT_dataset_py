
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
covid_dir='../input/deep-learning-competition-cs-2020/train/train/COVID19 AND PNEUMONIA/'
normal_dir='../input/deep-learning-competition-cs-2020/train/train/NORMAL'
covid_imgs=[]
labels=[]
img_width=128
img_height=128
for  dirname, _, imgs in os.walk(covid_dir):
    for img in imgs:
        abs_path = os.path.join(dirname, img)
        image=cv2.imread(abs_path)
        image=cv2.resize(image,(img_width,img_height))
        covid_imgs.append(image)
        labels.append(1)
for  dirname, _, imgs in os.walk(normal_dir):
    for img in imgs:
        abs_path = os.path.join(dirname, img)
        image=cv2.imread(abs_path)
        image=cv2.resize(image,(img_width,img_height))        
        covid_imgs.append(image)
        labels.append(0)
imgs=covid_imgs
test_dir='../input/deep-learning-competition-cs-2020/test/test'
test_imgs=[]
img_names=[]
for  dirname, _, imgs in os.walk(test_dir):
    for img in imgs:
        abs_path = os.path.join(dirname, img)
        img_names.append(img)
        image=cv2.imread(abs_path)
        image=cv2.resize(image,(img_width,img_height))
        test_imgs.append(image)
test_imgs=np.array(test_imgs)
print(len(covid_imgs))
X_train, X_test, y_train, y_test = train_test_split(covid_imgs, labels, test_size=0.3, random_state=42,stratify=labels)
x_train=np.array(X_train)
x_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test)
KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(img_width,img_height,3)),
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
        keras.layers.Dense(2,activation='softmax') ,    
        ])

KerasModel.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])
ThisModel = KerasModel.fit(x_train, y_train, epochs=20,batch_size=64,verbose=1,validation_split=0.1)
ModelLoss, ModelAccuracy = KerasModel.evaluate(x_test, y_test)
print(ModelAccuracy)
y_pred = KerasModel.predict(test_imgs)
print(y_pred[10])


data = list(zip(img_names, y_pred))
submission_df = pd.DataFrame(data, columns=['Image','Label'])
submission_df.head()
submission_df.to_csv("Submission.csv",index=False)