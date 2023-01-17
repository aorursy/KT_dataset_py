# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
# load data set
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
# splitting tool for the validation set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_l,Y_l,test_size=0.2,random_state=42)
#reshape
x_train = x_train.reshape(-1,64,64,1)
x_test = x_test.reshape(-1,64,64,1)
print("x train shape :",x_train.shape)
print("y train shape :",y_train.shape)
print("x test shape :",x_test.shape)
print("y test shape :",y_test.shape)
# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),activation="relu",padding="same",input_shape=(64,64,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#
model.add(Conv2D(filters=8,kernel_size=(4,4),activation="relu",padding="same"))
model.add(Conv2D(filters=8,kernel_size=(4,4),activation="relu",padding="same"))
model.add(MaxPool2D(pool_size=(2,2)))#!!!dersi tekrar izle bu kısmını
model.add(Dropout(0.25))   
model.add(Flatten())
          
#full connected layer
#hidden layers
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
#output layers
model.add(Dense(10,activation="softmax"))


model.summary()
#compile model
model.compile(optimizer = Adam(lr=0.0003) , loss = "categorical_crossentropy", metrics=["accuracy"])
#Train
history = model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
plt.figure(figsize=(24,8))
plt.subplot(1,2,1)
plt.plot(history.history["val_acc"],label="validation_accuracy",c="blue",linewidth=4)
plt.plot(history.history["acc"],label="training_accuracy",c="red",linewidth=4)
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["val_loss"],label="validation_loss",c="red",linewidth=4)
plt.plot(history.history["loss"],label="training_loss",c="green",linewidth=4)
plt.legend()
plt.grid(True)

plt.suptitle("ACC / LOSS",fontsize=18)

plt.show()