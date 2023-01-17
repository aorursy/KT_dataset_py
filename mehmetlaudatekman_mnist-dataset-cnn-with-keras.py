# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('darkgrid')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.head()
fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(train_df["label"])

plt.show()
train_df.info()
test_df.head()
test_df.info()
XTRAIN = train_df.drop("label",axis=1)

YTRAIN = train_df.label



XTEST = test_df
# Normalizing

XTRAIN = XTRAIN / 255.0

XTEST = XTEST / 255.0

# Label Encoding

from keras.utils.np_utils import to_categorical

YTRAIN = to_categorical(YTRAIN,num_classes=10)

XTRAIN = XTRAIN.values.reshape((-1,28,28,1))

XTEST = XTEST.values.reshape((-1,28,28,1))



print("Shape of XTRAIN",XTRAIN.shape)

print("Shape of XTEST",XTEST.shape)
from sklearn.model_selection import train_test_split



x_train,x_val,y_train,y_val = train_test_split(XTRAIN,YTRAIN,test_size=0.1,random_state=1)

print("Shape of x_train",x_train.shape)

print("Shape of x_val",x_val.shape)

print("Shape of y_train",y_train.shape)

print("Shape of y_val",y_val.shape)
from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D # Layer types

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau
# Model Object Creating

model = Sequential()

#Convolution Operator

model.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation="relu",input_shape=(28,28,1)))

model.add(Conv2D(filters=8,kernel_size=(5,5),padding="Same",activation="relu"))

# Max Pool

model.add(MaxPool2D(pool_size=(2,2)))

#Dropout

model.add(Dropout(0.25))

#Convolution Operator

model.add(Conv2D(filters=16,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu"))

#Max Pool

model.add(MaxPool2D(pool_size=(2,2)))

#Dropout

model.add(Dropout(0.25))

#Convolution Operator

model.add(Conv2D(filters=16,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu"))

#Max Pool

model.add(MaxPool2D(pool_size=(2,2)))

#Dropout

model.add(Dropout(0.25))

# Fully Connected (Flatten)

model.add(Flatten())

# Artifical Neural Network Layer 1 

model.add(Dense(256,activation="relu"))

#Dropout

model.add(Dropout(0.25))

#Artifical Neural Network Layer 2

model.add(Dense(256,activation="relu"))

#Dropout

model.add(Dropout(0.25))

#Output

model.add(Dense(10,activation="softmax"))
#Adam Optimizer

optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999) 
model.compile(optimizer = optimizer, loss = "categorical_crossentropy",metrics=["accuracy"])
batch_size = 300

epochs = 30

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.5,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.4, # Randomly zoom image 4%

        width_shift_range=0.8,  # randomly shift images horizontally 8%

        height_shift_range=0.8,  # randomly shift images vertically 8%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val),

                              steps_per_epoch=x_train.shape[0] // batch_size)



#model.save("path")

#from keras.models import load_model

# model = model_load("path")
pred = model.predict_classes(XTEST)
pred[:5]
pred.shape
result = pd.DataFrame({"ImageId":range(1,28001),"Label":pred})

result.head()
result.to_csv('result.csv',index=False)