# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout,BatchNormalization

from keras.preprocessing.image  import load_img,ImageDataGenerator

from keras.preprocessing.image import img_to_array

from keras.utils import np_utils
train = pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')
train.head()

X_train= train.drop(['label'],axis=1)
y_train= train['label']
del train
sns.countplot(y_train)
X_train.isnull().any().describe(), test.isnull().any().describe()
X_train=X_train.astype('float32')/255

test=test.astype('float32')/255

num_classes=len(np.unique(y_train))
img=X_train.iloc[0].as_matrix()

plt.imshow(img.reshape((28,28)))
y_train=keras.utils.to_categorical(y_train,num_classes)
X_train=X_train.values.reshape(-1,28,28,1)

test= test.values.reshape(-1,28,28,1)
X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,test_size=0.15)
print(X_train.shape)
plt.imshow(X_train[1][:,:,0],cmap='gray')
model= Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding= 'same',activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Conv2D(filters=32,kernel_size=4,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Conv2D(filters=64,kernel_size=4,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))

model.add(BatchNormalization())

model.add(Conv2D(filters=128,kernel_size=2,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(num_classes,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
epochs=250

batch_size=256
datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True,rotation_range=0.5,zoom_range=0.1)

datagen.fit(X_train)
checkpoint =model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),epochs=epochs,validation_data= (X_valid,y_valid),verbose=2,steps_per_epoch=X_train.shape[0]/batch_size)    
score=model.evaluate(X_valid,y_valid,verbose=1)

print('\n', 'Test accuracy:', round(score[1]*100,2))
results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mysubmission.csv",index=False)