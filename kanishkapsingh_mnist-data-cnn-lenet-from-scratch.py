# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten,BatchNormalization
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from skimage import exposure
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.utils import to_categorical
from keras.optimizers import SGD
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
print(os.listdir("../input"))
from tqdm import tqdm
# Defining a function to image preprocess
def preprocess(data):
    for im in tqdm(range(data.shape[0])):
        im_array=data[im].reshape(28,28)
        im_array=exposure.adjust_sigmoid(exposure.adjust_sigmoid(exposure.adjust_sigmoid(exposure.adjust_sigmoid(im_array))))
        data[im]=im_array.reshape(28,28,1)
    return data
# load the image data and split the train data into train and validation data
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
testIndex=test.index
ylabel_train=train['label']
train=train.drop('label',axis=1).values.reshape(train.shape[0],28,28,1)
test=test.values.reshape(test.shape[0],28,28,1)
print('Adjusting training and test images')
train=train/255.0
test=test/255.0
X_train,X_test,y_train,y_test=train_test_split(train,to_categorical(ylabel_train),test_size=0.2,random_state=42)
print(X_train.shape,X_test.shape)
# Using Image Data Generator for Image augmentation
trainDatagen=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,fill_mode='nearest')
valDatagen=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,fill_mode='nearest')
# Create the architecture of the CNN model to be implemented
model=Sequential()
model.add(Conv2D(64,3,activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(128,3,activation='relu'))
model.add(MaxPooling2D((3,3),strides=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(256,3,activation='relu'))
model.add(Conv2D(512,1,activation='relu'))
model.add(MaxPooling2D((3,3),strides=(3,3)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# Setting the callback functions
estop=EarlyStopping(monitor='val_acc',patience=10,verbose=1)
lrreduce=ReduceLROnPlateau(monitor='val_acc',verbose=1,factor=0.5,min_lr=1e-10,patience=3)
# Fit and validate the model
bSize=128
model.fit_generator(trainDatagen.flow(x=X_train,y=y_train,batch_size=bSize),steps_per_epoch=X_train.shape[0]*20/bSize,epochs=500,validation_data=(X_test,y_test),callbacks=[estop,lrreduce])
#Generate predictions on the test data
predictions=model.predict(test)
predictions_df=pd.DataFrame({'ImageId':testIndex+1,'Label':np.argmax(predictions,axis=1)})
predictions_df[['ImageId','Label']].to_csv('Submission_CNN_LargeAugmentation.csv',index=False)
# Saving the model
model.save('MNIST_CNN_LargeAugmentation.kps')


