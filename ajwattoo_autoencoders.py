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
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
import glob
train_images_alder = glob.glob('/kaggle/input/trunk-assignment/alder/*.JPG')
train_images_ginkgo_biloba = glob.glob('/kaggle/input/trunk-assignment/ginkgo biloba/*.JPG')
train_images_birch = glob.glob('/kaggle/input/trunk-assignment/birch/*.JPG')
train_images_beech = glob.glob('/kaggle/input/trunk-assignment/beech/*.JPG')
train_images_chestnut = glob.glob('/kaggle/input/trunk-assignment/chestnut/*.JPG')
train_images_hornbeam = glob.glob('/kaggle/input/trunk-assignment/hornbeam/*.JPG')
train_images_horse_chestnut = glob.glob('/kaggle/input/trunk-assignment/horse chestnut/*.JPG')
train_images_linden = glob.glob('/kaggle/input/trunk-assignment/linden/*.JPG')
train_images_oak = glob.glob('/kaggle/input/trunk-assignment/oak/*.JPG')
train_images_pine = glob.glob('/kaggle/input/trunk-assignment/pine/*.JPG')
train_images_spruce = glob.glob('/kaggle/input/trunk-assignment/spruce/*.JPG')
train_images_oriental_plane= glob.glob('/kaggle/input/trunk-assignment/oriental plane/*.JPG')
train_images=[train_images_alder,train_images_ginkgo_biloba,train_images_birch,train_images_beech,train_images_chestnut,train_images_hornbeam ,train_images_horse_chestnut,train_images_linden,train_images_oak,train_images_pine,train_images_spruce,train_images_oriental_plane]
import cv2
y_train=[]
X_train = []
for (label, fnames) in enumerate(train_images):
    for fname in fnames:
            img = cv2.imread(fname)
            img = cv2.resize(img, (150 ,150 ) , interpolation=cv2.INTER_AREA)
            img=  img.astype('float32') / 255.
            y_train.append(label)
            X_train.append(img)
X_train=np.array(X_train)
print(X_train.shape)
categories=['alder','ginkgo_biloba','birch','beech','chestnut','hornbeam','horse_chestnut','linden','oak','pine','spruce','oriental_plane']

import matplotlib.pyplot as plt
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221,title=categories[y_train[i]]), plt.imshow(X_train[i], cmap='gray')
plt.subplot(222,title=categories[y_train[i+34]]), plt.imshow(X_train[i+34], cmap='gray')
plt.subplot(223,title=categories[y_train[i+200]]), plt.imshow(X_train[i+200], cmap='gray')
plt.subplot(224,title=categories[y_train[i+300]]), plt.imshow(X_train[i+300], cmap='gray')
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, y_test = train_test_split(X_train, y_train, test_size=0.1,train_size=0.9,shuffle=True, stratify=y_train)
x, x_val, y, y_val = train_test_split(x_train, Y_train, test_size=0.2,train_size=0.8,shuffle=True, stratify=Y_train)
print('test data',x_test.shape)
print('length of Y_test',len(y_test))
print('training data is',x.shape)
print('length of Y_train',len(y))
print('validation data is',x_val.shape)
print('length of Y_val',len(y_val))
from keras.utils.np_utils import to_categorical

y= to_categorical(y)
y_val=to_categorical(y_val)
y_test=to_categorical(y_test)
print(x.shape)
print(x_val.shape)
print(x_test.shape)
print(y.shape)
print(y_val.shape)
print(y_test.shape)

from keras import  models
'''
model=models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(150,150,3))) #150*150*32
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))  #75*75*32
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) #75*75*64
model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) #38*38*64
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) #38*38*128
model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) #19*19*128

#decoder
model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) #19*19*128
model.add(UpSampling2D((2,2))) #38*38*128
model.add(Conv2D(64, (3, 3), activation='relu',padding='same')) #38*38*64
model.add(UpSampling2D((2,2))) #72*72*128
model.add(Conv2D(32, (3, 3), activation='relu',padding='same' ))  #72*72*64
model.add(UpSampling2D((2,2))) #144*144*64
model.add(Conv2D(3, (3, 3),activation='softmax', padding='same'))  #144*144*3

model.compile(loss='mean_squared_error', optimizer = 'Adam')'''
#model.summary()
input_img = Input(shape=(150,150, 3))  # adapt this if using `channels_first` image data format

new = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
new = MaxPooling2D((2, 2), padding='same')(new)
new = Conv2D(64, (3, 3), activation='relu', padding='same')(new)
new = MaxPooling2D((2, 2), padding='same')(new)
new = Conv2D(128, (3, 3), activation='relu', padding='same')(new)
encoded = MaxPooling2D((2, 2), padding='same')(new)

# at this point the representation is (19, 19, 128)

new = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
new = UpSampling2D((2, 2))(new)
new = Conv2D(64, (3, 3), activation='relu', padding='same')(new)
new = UpSampling2D((2, 2))(new)
new = Conv2D(32, (3, 3), activation='relu',padding='same')(new)
new = UpSampling2D((2, 2))(new)
decoded = Conv2D(3, (3, 3), activation='sigmoid')(new)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='mean_squared_error')
autoencoder.summary()
print(x.shape)
history = autoencoder.fit(x,x,epochs=50, batch_size=128,shuffle=True,validation_data=(x_val,x_val))
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(50)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
decoded_imgs = autoencoder.predict(x_test)
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i, ..., 0])
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(decoded_imgs[i, ..., 0])  
plt.show()
