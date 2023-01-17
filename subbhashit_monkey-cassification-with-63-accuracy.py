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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
from PIL import Image
l=[]
image=[]
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n0/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n0')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n1/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n1')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n2/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n2')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n3/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n3')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n4/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n4')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n5/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n5')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n6/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n6')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n7/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n7')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n8/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n8')
for i in glob.glob('/kaggle/input/10-monkey-species/training/training/n9/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    image.append(np.array(r))
    l.append('n9')
len(image)
l=pd.DataFrame(l)
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
labels=la.fit_transform(l)
labels
image=np.array(image)
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,10))
ax=figure.add_subplot(121)
ax.imshow(image[0])
bx=figure.add_subplot(122)
bx.imshow(image[60])
np.save("Images",image)
np.save("labels",labels)
image=np.load("Images.npy",allow_pickle=True)
labels=np.load("labels.npy",allow_pickle=True)
s=np.arange(image.shape[0])
np.random.shuffle(s)
image=image[s]
labels=labels[s]
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False, 
    samplewise_std_normalization=False,
    rotation_range=60, 
    zoom_range = 0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.1,
    fill_mode = "reflect"
    )
y_class=len(np.unique(labels))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(image,labels,test_size=0.1)
import keras
y_train=keras.utils.to_categorical(y_train,y_class)
y_test=keras.utils.to_categorical(y_test,y_class)
datagen.fit(x_train)
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers import Activation, Convolution2D, Dropout, Conv2D,AveragePooling2D, BatchNormalization,Flatten,GlobalAveragePooling2D
from keras import layers
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
l2_reg=0.001
opt=Adam(lr=0.001)
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=128),epochs= 50,validation_data=(x_test,y_test),steps_per_epoch=50)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])
t_image=[]
for i in glob.glob('/kaggle/input/10-monkey-species/validation/validation/*/*.jpg'):
    img=cv2.imread(i)
    ar=Image.fromarray(img,'RGB')
    r=ar.resize((50,50))
    t_image.append(np.array(r))
len(t_image)
np.save("images",t_image)
t_images=np.load('images.npy')
pred=np.argmax(model.predict(t_images),axis=1)
pred_t=la.fit_transform(pred)
t_img=np.expand_dims(t_images[0],axis=0)
pred1=np.argmax(model.predict(t_img),axis=1)
pred_t1=la.fit_transform(pred1)
print(pred_t1[0])
plt.imshow(t_images[0])