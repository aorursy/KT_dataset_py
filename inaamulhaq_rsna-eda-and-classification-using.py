import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pydicom 
import glob, pylab
import cv2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# We have to take the train CSV file and view it 
traindata=pd.read_csv('../input/stage_1_train_labels.csv')
print(traindata.iloc[0])
traindata.sample(5)
classlabels=pd.read_csv('../input/stage_1_detailed_class_info.csv')
#print(classlabels.sample(5))
#traindata=ftrainingdata
ftrainingdata=traindata.merge(classlabels)
#print(ftrainingdata)
inputtd = np.ndarray(shape=(1000,28, 28), dtype=np.float32)
inputtd.shape
outputtd=ftrainingdata['Target'][0:1000]
x=0
for index in range (0,1000):
 a=index   
 patientId=ftrainingdata['patientId'][a]
 dcmfile='../input/stage_1_train_images/%s.dcm'% patientId
 dcmdata=pydicom.dcmread(dcmfile)
 im=dcmdata.pixel_array
 #crop=cv2.resize(im, (28, 28))
 #pylab.imshow(crop, cmap=pylab.cm.gist_gray)
 #inputtd=cv2.resize(im, (28, 28))
 inpu=np.array(cv2.resize(im, (28, 28)))
 inputtd[x] = inpu
 x += 1    
inn=inputtd 
inn= inn.reshape(1000,28,28,1)
inn.shape
import keras 
outputtd= keras.utils.to_categorical(outputtd,num_classes=2)
x_train, x_test, y_train, y_test = train_test_split(inn,outputtd,test_size=0.5,random_state=4)
model = Sequential()
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
model.summary()
model.fit(x_train,y_train,validation_data=(x_test,y_test))
model.evaluate(x_test,y_test)