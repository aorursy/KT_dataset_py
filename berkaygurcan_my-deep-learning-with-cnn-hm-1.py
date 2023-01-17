# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#read the train set
train = pd.read_csv('../input/train.csv')
print(train.shape)
train.head()
train.info()
train.describe()
# read the test set
test= pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
Y_train=train["label"]
X_train=train.drop(["label"],axis=1)
# visualize number of digits classes
plt.figure(figsize=(15,7))
g = sns.countplot(Y_train, palette="icefire")
plt.title("Number of digit classes")
Y_train.value_counts()
X_train=X_train/255.0
test=test/255.0
print("X_train shape :" ,X_train.shape)
print("Test shape :",test.shape)
print(type(X_train))
#reshape 
X_train=X_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
print("x_train shape: ",X_train.shape)
print("test shape: ",test.shape)
#label encoding
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
Y_train=to_categorical(Y_train,num_classes=10)
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val=train_test_split(X_train,Y_train,test_size=0.2,random_state=2)
print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)
# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

model.add(Conv2D(64, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(32,kernel_size=(3,3),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(32,kernel_size=(5,5),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Dropout(0.25))

model.add(Conv2D(16,kernel_size=(2,2),padding="same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128,activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()
# Define the optimizer
optimizer=Adam(lr=0.003,beta_1=0.9,beta_2=0.999)
#compile model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# data augmentation I dont use it now

datagen = ImageDataGenerator(
    """
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
"""
    )
datagen.fit(X_train)

#Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=256),
                              epochs = 20, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0]//256)

test_x = pd.read_csv('../input/test.csv')
test_x = test_x.values.reshape(-1,28,28,1)
test_x = test_x / 255.0
predictions=model.predict(test_x)
predictions[100]
predictions= np.argmax(predictions, axis=1)
submission=pd.DataFrame({'ImageId':range(1,len(test_x)+1),'Label':predictions})
submission.to_csv("cnn_results3.csv",index=False)
submission.head(7)
