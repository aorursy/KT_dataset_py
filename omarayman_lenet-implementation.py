# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()

train.describe()
train.info()
train.isnull().sum()
ytrain = train['label']
xtrain=train.drop('label',axis=1)
xtrain.head()
ytrain.head()
sns.countplot(ytrain)
xtrain = xtrain/255.0
test = test/255.0
xtrain = xtrain.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
ytrain = to_categorical(ytrain,num_classes=10)
from sklearn.model_selection import train_test_split
xtrain,xval,ytrain,yval = train_test_split(xtrain,ytrain,test_size=0.2,random_state=42)
import random 
import matplotlib.pyplot as plt
index = random.randint(0,len(xtrain))
image = xtrain[index].squeeze()
plt.figure(figsize=(1,1))
plt.imshow(image,cmap="gray")
print(ytrain[index])

from sklearn.utils import shuffle
xtrain,ytrain = shuffle(xtrain,ytrain)
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD

    
    

model = Sequential()
model.add(Convolution2D(20, 5, 5, border_mode="same",
input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
 
    # softmax classifier
model.add(Dense(10))
model.add(Activation("softmax"))
    
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01),
metrics=["accuracy"])
model.fit(xtrain,ytrain,batch_size=128,epochs=50,verbose=1,validation_data=(xval,yval))
predicted_labels = model.predict(test)
predicted_labels = np.argmax(predicted_labels,axis=1)
predicted_labels=pd.Series(predicted_labels,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predicted_labels],axis = 1)
submission.to_csv("LeNet.csv",index=False)
