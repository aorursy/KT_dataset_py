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
df = pd.read_csv('../input/train.csv')
df.head()
df['imagepath'] = df["Image"].map(lambda x:'../input/train/'+x)
labels=df.Id #using the ids of whale as labels to our model
from sklearn import preprocessing
from keras.utils import np_utils

labels1= preprocessing.LabelEncoder()
labels1.fit(labels)
encodedlabels = labels1.transform(labels) #integer encoding
clearalllabels = np_utils.to_categorical(encodedlabels) #one hot encoding


df.head()
import cv2
def imageProcessing(imagepath,name): 
    img=cv2.imread(imagepath)
    height,width,channels=img.shape
    if channels !=1:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img.astype(np.float)
    img = cv2.resize(img,(70,70))
    img = img-img.min()//img.max()-img.min()#step 3
    img=img*255#step 4
    return img
t=[]
    
for i in range(0,9850):
    t.append(imageProcessing(df.imagepath[i],df.Image[i]))
t = np.asarray(t)
t=t.reshape(-1,70,70,1)
t.shape
#Same for test images
from glob import glob
path_to_images = '../input/test/*.jpg'
images=glob(path_to_images)
test=[]
for i in images:
    img = cv2.imread(i)
    height,width,channels=img.shape
    if channels !=1:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=img.astype(np.float)
    img = cv2.resize(img,(70,70))
    img = img-img.min()//img.max()-img.min()
    img=img*255

    test.append(cv2.resize(img,(70,70)))
    # Get image label (folder name)

t.shape
test=np.asarray(test)
test.shape

from keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255,
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True)
#training the image preprocessing
image_gen.fit(t) # fit t to the imageGenerator to let the magic happen

t.shape
clearalllabels.shape
test.shape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Activation
from keras import optimizers

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(70,70,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4251,activation='softmax'))

optimizer=optimizers.SGD()


model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.summary()

#model.fit_generator(image_gen.flow(t,clearalllabels, batch_size=128),steps_per_epoch=  9850,epochs=10,verbose=1)
