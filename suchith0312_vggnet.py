import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import time
import tarfile
from tqdm import tqdm_notebook

from keras.optimizers import *
from keras.layers import *
from keras.models import Model,Sequential
from keras.callbacks import *
from skimage.transform import resize,rotate
from keras.applications.vgg16 import VGG16,preprocess_input

from sklearn.model_selection import train_test_split,StratifiedKFold

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
n_pixels=224
channels=3
train=pd.read_csv("../input/solution.csv")
train.head()
sns.countplot(train['category'])
X=np.zeros((train.shape[0],n_pixels,n_pixels,channels),dtype=np.uint8)
y=pd.get_dummies(train['category']).iloc[:,:].values

print("Reading train images")
for index,name in tqdm_notebook(enumerate(train['id']),total=train.shape[0]):
    x=np.array(load_img(path='../input/training1/training/'+str(name)+".png",color_mode='rgb'))
    x=resize(x,output_shape=(n_pixels,n_pixels,channels),preserve_range=True,mode='constant')
    X[index]=x
    
print("Shape=",X.shape,y.shape)
def augment(X,y):
    m=X.shape[0]
    print("Flipping")
    return np.append(X,[np.fliplr(x) for x in X],axis=0),np.append(y,[i for i in y],axis=0)

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.20,random_state=42)

print("Training Shape",X_train.shape,y_train.shape)
print("Validation Augmentaion",X_val.shape,y_val.shape)

print("Training Augmentation:")
X_train,y_train=augment(X_train,y_train)
print("After Augmentation",X_train.shape,y_train.shape)
print("Validation Augmentation:")
X_val,y_val=augment(X_val,y_val)
print("After Augmentation",X_val.shape,y_val.shape)

vgg=VGG16(include_top=False,weights='imagenet',input_shape=((224,224,3)))
add_model=Sequential()
add_model.add(Lambda(preprocess_input,input_shape=(224,224,3)))
add_model.add(vgg)
add_model.add(Conv2D(64,(3,3),padding='valid'))
add_model.add(Flatten())
add_model.add(Dense(512,activation='relu'))
add_model.add(Dense(128,activation='relu'))
add_model.add(Dense(6,activation='softmax'))
add_model.layers[1].trainable=False
add_model.summary()
batch_size=16
epochs=2
name="Firstmodel.h5"
checkpointer=ModelCheckpoint(name,monitor='val_acc',save_best_only=True,mode='max',verbose=1)
earlystopper=EarlyStopping(monitor='val_acc',patience=10,mode='max',verbose=1)
add_model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
add_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,
          callbacks=[checkpointer,earlystopper],validation_data=(X_val,y_val))





