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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import seaborn as sns

np.random.seed(2)
num_classes=10

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import itertools
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")
train_images=[]
for i in range(0,len(train)):
    x=np.array(train.iloc[i][1:].values)
    x=x.reshape(28,28)
    train_images.append(x)
train_images = np.asarray(train_images) 


test_images = []
for i in range(0,len(test)):
    x = np.array(test.iloc[i].values)
    x = x.reshape(28,28)
    test_images.append(x)
test_images = np.asarray(test_images)

plt.imshow(train_images[7],cmap='gray')
plt.show()
print(train.shape)
print(test.shape)
X_train = train.drop(labels=["label"],axis=1)
Y_train = train['label']
# normalising the data values
X_train = X_train / 255.0
test = test / 255.0
#images converted to grayscale 
# reshape which will convert to a 28x28x1 array. height = width = 28. Gray scale so 1. RGB => 3
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print(X_train.shape)
print(test.shape)
# onehot encoding for label 
Y_train_cat = to_categorical(Y_train, num_classes = 10)
model = Sequential()
model.add(Conv2D(64,(5,5),padding="valid",strides=1,name="Conv1",activation="relu",input_shape=(28,28,1)))
#practice model.add(Conv2D(8,(2,2),padding="valid",strides=1,name="Conv2"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(8,(5,5),padding="valid",strides=1,name="Conv2",activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64,activation ="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(10,activation ="softmax"))
model.summary()
model.compile(loss = keras.losses.categorical_crossentropy,optimizer = RMSprop(),metrics =['accuracy'])
History=model.fit(X_train,Y_train_cat,batch_size = 32,epochs =5,verbose= 1,validation_split = 0.2)
model.evaluate(X_train,Y_train_cat)
predict=model.predict_classes(test)
results = pd.Series(predict,name="Label")
submission = pd.concat([pd.Series(range(1,len(predict)+1),name = "ImageId"),results],axis = 1)
submission.to_csv("submission1.csv", index = False)
