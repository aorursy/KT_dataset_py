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
import tensorflow as tf

import csv

train_x,train_y=[],[]

colum=None

with open('/kaggle/input/mnist-in-csv/mnist_train.csv','r') as file:

    reader=csv.reader(file,delimiter=',')

    next(reader)

    for row in reader:

        train_x.append(row[1:])

        train_y.append(int(row[0]))
import tensorflow as tf

tf.__version__
import numpy as np

ntrain=len(train_x)

dim=int(np.sqrt(len(train_x[0])))
train_input=np.array(train_x).reshape(ntrain,dim,dim)



input_dim=train_input.shape
train_label=tf.keras.utils.to_categorical(np.array(train_y),10)

train_input=train_input.reshape(ntrain,dim,dim,1)

print(train_label.shape)

print(train_input.shape)
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Flatten

from tensorflow.keras.layers import BatchNormalization,Dropout



model=Sequential([

    Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),

    MaxPool2D((2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Conv2D(32,kernel_size=(3,3),activation='relu'),

    MaxPool2D((2,2)),

    BatchNormalization(),

    Dropout(0.2),

    Flatten(),

    Dense(64,activation='relu'),

    Dense(10,activation='softmax')

])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen=ImageDataGenerator(rescale=1/255.0)

train_data_gen=train_gen.flow(train_input,train_label,batch_size=32)
test_x,test_y=[],[]

with open('/kaggle/input/mnist-in-csv/mnist_test.csv','r') as file:

    reader=csv.reader(file,delimiter=',')

    next(reader)

    for row in reader:

        test_x.append(row[1:])

        test_y.append(int(row[0]))

ntest=len(test_y)

dim_t=int(np.sqrt(len(test_x[0])))

test_input=np.array(test_x)

test_input=test_input.reshape(ntest,dim_t,dim_t,1)

test_output=tf.keras.utils.to_categorical(np.array(test_y),10)
test_gen=ImageDataGenerator(rescale=1/255.0)

test_data_gen=test_gen.flow(test_input,test_output,batch_size=32)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit_generator(train_data_gen,epochs=10,validation_data=test_data_gen)
test_data_gen.reset()

pred=model.predict_generator(test_data_gen,steps=313)

pred=np.argmax(pred,axis=1)

print(len(pred))
from sklearn.metrics import confusion_matrix



x_gen,y_gen=test_data_gen.next()

print(x_gen.shape)

print(y_gen.shape)

y_gen=np.argmax(y_gen,axis=1)

print(y_gen.shape)
cm=confusion_matrix(test_y,pred)

precision=cm[0,0]/(cm[0,0]+cm[0,1])

recall=cm[0,0]/(cm[0,0]+cm[1,0])

fscore=2*(precision*recall)/(precision+recall)
print(precision)

print(recall)

print(fscore)