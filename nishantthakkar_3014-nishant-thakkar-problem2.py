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
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()


Image.open("/kaggle/input/sign-language-mnist/amer_sign2.png")
Image.open("/kaggle/input/sign-language-mnist/amer_sign3.png")
df = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test_df = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

print(">>> Train Data Shape: ",df.shape)
print(df.head()) 

print("="*75)

print(">>> Test Data Shape: ",test_df.shape)
print(test_df.head())
# create labels 
train_labels =  lb.fit_transform(df["label"])
test_labels = lb.transform(test_df["label"])
print("Number of images in train: ",len(train_labels),"\n"
      "nNumber of images in test: ",len(test_labels))

# unique
nunique_labels = df["label"].nunique()

print("nuique labels: ",nunique_labels)
# drop label data from df and test df 
df , test_df = df.drop(['label'],axis=1) , test_df.drop(['label'],axis=1)
# reshape data
train_images , test_images =  df.values.reshape(-1,28,28,1), test_df.values.reshape(-1,28,28,1)
print("train data shape:" , train_images.shape)
print("train labels shape:" , train_labels.shape)
print("="*40)
print("test data shape:" , test_images.shape)
print("test labels shape:" , test_labels.shape)
print("="*40)
plt.imshow(train_images[0].reshape(28,28))
plt.show()
plt.imshow(test_images[0].reshape(28,28)),
plt.show()
# Build Data Imagegenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
# rescale test data
test_images =test_images/255

# Build model

model=Sequential()
# layer 1
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=train_images.shape[1:]))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))

# layer 2 
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))

#layer 3
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))

# flatten image
model.add(Flatten())

# dense layers
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))

# output layer
model.add(Dense(units=nunique_labels,activation='softmax'))

# view model summary
model.summary()

# compile data
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Trial 1
# fit data
model.fit(train_datagen.flow(train_images,train_labels,batch_size=200),
         epochs = 35,
          validation_data=(test_images,test_labels),
          shuffle=1)

print("Trial: 1")

trial1_history = model.history.history

plt.plot(trial1_history["loss"],label ='Traing Loss')
plt.plot(trial1_history["val_loss"],label ='Validation Loss')
plt.legend()
plt.show()

plt.plot(trial1_history["accuracy"],label ='Traing accuracy')
plt.plot(trial1_history["val_accuracy"],label ='Validation accuracy')
plt.legend()
plt.show()
(ls,acc)=model.evaluate(x=test_images,y=test_labels)
print(f'MODEL ACCURACY = {acc*100}')
# Trial 2
# fit data
model.fit(train_datagen.flow(train_images,train_labels,batch_size=200),
         epochs = 35,
          validation_data=(test_images,test_labels),
          shuffle=1)

print("Trial: 2")
trial2_history = model.history.history
plt.plot(trial2_history["loss"],label ='Traing Loss')
plt.plot(trial2_history["val_loss"],label ='Validation Loss')
plt.legend()
plt.show()

plt.plot(trial2_history["accuracy"],label ='Traing accuracy')
plt.plot(trial2_history["val_accuracy"],label ='Validation accuracy')
plt.legend()
plt.show()
(ls,acc)=model.evaluate(x=test_images,y=test_labels)
print(f'MODEL ACCURACY = {acc*100}')