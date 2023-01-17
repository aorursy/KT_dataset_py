# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

Dig_MNIST = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
print(train_data.shape)

print(test_data.shape)

print(Dig_MNIST.shape)
#训练集数据

train=train_data.iloc[:,1:].values

train_label=train_data.iloc[:,0].values

print("训练数据shape：",train.shape)

print("标签数据shape",train_label.shape)
#对训练数据进行预处理

train=train.reshape(train.shape[0],28,28,1)

print("训练数据shape:",train.shape)
import tensorflow as tf

from sklearn.model_selection import train_test_split

import keras

train_label=keras.utils.to_categorical(train_label,10)

print("标签shape:",train_label.shape)
#测试数据

test=test_data.drop('id',axis=1).iloc[:,:].values

test=test.reshape(test.shape[0],28,28,1)

print("测试数据shape:",test.shape)
train=train/255

test=test/255
train.shape[0]
#验证数据

valid=Dig_MNIST.drop('label',axis=1).iloc[:,:].values

print("验证数据shape",valid.shape)

valid = valid.reshape(valid.shape[0], 28, 28,1)

print("reshape后的shape:",valid.shape)
#valid标签

valid_label=Dig_MNIST.label

print("标签数据shape:",valid_label.shape)
valid=valid/255
X_train,X_test,y_train,y_test=train_test_split(train,train_label,random_state=42,test_size=0.15)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
train.shape
valid.shape
train_label.shape
valid_label=keras.utils.to_categorical(valid_label,10)
valid_label.shape
import matplotlib.pyplot as plt

plt.imshow(X_train[0][:,:,0])
from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization
#模型的构建

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=0.15))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(BatchNormalization(momentum=.15))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))
model.summary()
NUM_CLASSES = 10

EPOCHS = 50

BATCH_SIZE = 128
#模型训练

optmize = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)



model.compile(loss = keras.losses.categorical_crossentropy, optimizer = optmize,  metrics=['accuracy'])

History = model.fit(train, train_label,

          batch_size = BATCH_SIZE,

          epochs = EPOCHS,

          verbose = 1,

          validation_data=(valid, valid_label))
model.evaluate(train,train_label)
model.evaluate(valid,valid_label)
my_predicted=model.predict(test)
my_predicted
y_pre=my_predicted.astype(int)

y_pre=np.argmax(y_pre,axis=-1)

y_pre
sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sub['label']=y_pre
sub.to_csv('submission.csv',index=False)
sub.head(20)