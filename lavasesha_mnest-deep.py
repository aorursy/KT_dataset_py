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
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from keras.datasets import mnist
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
ytest=pd.read_csv("../input/sample_submission.csv")
train['label'].unique()
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(train.drop('label',axis=1),train['label'],test_size = 0.25, random_state=101)
y_train.shape
y_val.shape
#reshape the X train and X test
X_val=X_val.values.reshape(-1,28,28,1)
X_train=X_train.values.reshape(-1,28,28,1)

X_val=X_val/X_val[0].max()
X_train=X_train/X_train[0].max()
from keras.utils.np_utils import to_categorical
y_train_cat=to_categorical(y_train)
y_val_cat=to_categorical(y_val)
y_val_cat[0]
scaledimage=X_train[0]
scaledimage.shape

g = plt.imshow(X_train[1][:,:,0])
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
model=Sequential()

#Convolutional layer
#model.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu',input_shape(28,28,1)))

model.add(Conv2D(filters = 256, kernel_size = (4,4),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 256, kernel_size = (4,4),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
#Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu', input_shape = (14,14,1)))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu', input_shape = (14,14,1)))
#Pooling layer
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', activation ='relu', input_shape = (7,7,1)))
model.add(Conv2D(filters = 64, kernel_size = (2,2),padding = 'Same', activation ='relu', input_shape = (7,7,1)))
model.add(Flatten())

#Dense layer
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
#Adam=Keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
#compile model
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999),metrics=['accuracy'])

model.summary()
model.fit(X_train,y_train_cat,epochs=5)
model.evaluate(X_val,y_val_cat)
X_test=test.values.reshape(-1,28,28,1)
predictions=model.predict_classes(X_test)
df=pd.DataFrame(predictions,columns=['Label'])
df['ImageId']=range(1,1+len(df))
#df['New_ID'] = range(880, 880+len(df))

df.head()
df1 = df[['ImageId','Label']] 
df1.set_index("ImageId",inplace=True)
df1.to_csv("submission_mnist.csv")
