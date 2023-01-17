# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from imageio import imread
import PIL
from skimage.transform import resize
from skimage import color
from matplotlib import pyplot as plt
# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Convolution2D,BatchNormalization,Flatten,Dense,Dropout,MaxPool2D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
dirList=os.listdir('../input/flowers/flowers/')
imageData=[]
imageLabel=[]
validExtensionsList=['jpg','tif','png','bmp']
for d in dirList:
    print('Processing...',d)
    for f in os.listdir('../input/flowers/flowers/'+d):
        ext=f.split('.')[-1]
        if ext in validExtensionsList:
            data=imread('../input/flowers/flowers/'+d+'/'+f)
            #color.rgb2gray(imread('../input/flowers/flowers/'+d+'/'+f))
            resized_data=resize(data,(100,100))
            imageData.append(resized_data)
            imageLabel.append(d)
        else:
            pass
print('DONE!')
np.shape(imageData),np.shape(imageLabel)
idx=np.random.randint(len(imageData))
plt.imshow(imageData[idx])
plt.xlabel(imageLabel[idx])
plt.show()
X=np.array(imageData)
X=X.reshape(X.shape[0],X.shape[1],X.shape[2],3)
y=np.array(imageLabel)
num_classes=len(set(y))
le=LabelEncoder()
y=le.fit_transform(y)
y=to_categorical(num_classes=num_classes,y=y)
input_shape=(X.shape[1],X.shape[2],X.shape[3])
np.shape(X),np.shape(y)
def createCNNModel():
    model=Sequential()
    model.add(Convolution2D(64,3,input_shape=input_shape,activation='relu'))
    model.add(Convolution2D(64,3,input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(Convolution2D(128,3,input_shape=input_shape,activation='relu'))
    model.add(Convolution2D(128,3,input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(Convolution2D(256,3,input_shape=input_shape,activation='relu'))
    model.add(Convolution2D(256,3,input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(Convolution2D(512,3,input_shape=input_shape,activation='relu'))
    model.add(Convolution2D(512,3,input_shape=input_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(num_classes,activation='softmax'))
    return model
model=createCNNModel()
model.summary()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test),np.shape(X_val),np.shape(y_val)
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(X_train)
epochs=100
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#model.fit_generator(datagen.flow(X_train, y_train),validation_data=(X_val,y_val),steps_per_epoch=len(X_train) / 10, epochs=epochs,verbose=1)
hist=model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=50,epochs=epochs,verbose=1)
model.evaluate(X_test,y_test)
plt.plot(range(len(hist.history['acc'])),hist.history['acc'])
plt.plot(range(len(hist.history['loss'])),hist.history['loss'])
plt.xlabel('epoch')
plt.show()
idxtest=np.random.randint(len(X_test))
testImg=X_test[idxtest]
plt.imshow(testImg,cmap='gray')
testImg=testImg.reshape(1,100,100,3)
pred=le.inverse_transform(np.argmax(model.predict(testImg)))
actual=le.inverse_transform(np.argmax(y_test[idxtest]))
print("Actual:",actual," Predicted:",pred)
plt.show()