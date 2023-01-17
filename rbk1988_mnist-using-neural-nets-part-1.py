# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.optimizers import Adam,RMSprop
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#setting random seed
np.random.seed(123)
train=pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
test=pd.read_csv("../input/test.csv")
print(test.shape)
test.head()
X_train=(train.iloc[:,1:].values).astype('float32')
y_train=(train.iloc[:,0].values).astype('int32')
X_test=(test.values).astype('float32')
X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_train.shape
X_test=X_test.reshape(X_test.shape[0],28,28,1)
X_test.shape
X_train=X_train/255.0
X_test=X_test/255.0
y_train=to_categorical(y_train)
#check the number of categories in the target
y_train.shape[1]
X=X_train
y=y_train
#random seed is set at the top of the file
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.10)
print(X_train.shape)
print(X_val.shape)
from keras.preprocessing import image
generator=image.ImageDataGenerator()
train_batches=generator.flow(X_train,y_train,batch_size=64)
validation_batches=generator.flow(X_val,y_val,batch_size=64)
shallow_16=Sequential()
shallow_16.add(Flatten(input_shape=(28,28,1)))
shallow_16.add(Dense(16,activation="relu"))
shallow_16.add(Dense(10,activation="softmax"))
shallow_16.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
shallow_16_training=shallow_16.fit_generator(generator=train_batches,
                                             steps_per_epoch=train_batches.
                                             n,epochs=1,
                                             validation_data=validation_batches,
                                             validation_steps=validation_batches.n)
shallow_16.summary()
import matplotlib.pyplot as plt
%matplotlib inline
epochs=[1,2,3]
train_accuracy=[0.9643,0.9846,0.9915]
val_accuracy=[0.9814,0.9893,0.9962]
plt.plot(epochs,train_accuracy,label="Training accuracy")
plt.plot(epochs,val_accuracy,label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='center right')
plt.title("Accuracy plot")
shallow_32=Sequential()
shallow_32.add(Flatten(input_shape=(28,28,1)))
shallow_32.add(Dense(32,activation="relu"))
shallow_32.add(Dense(10,activation="softmax"))
shallow_32.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
shallow_32_training=shallow_32.fit_generator(generator=train_batches,
                                             steps_per_epoch=train_batches.n,
                                             epochs=1,
                                             validation_data=validation_batches,
                                             validation_steps=validation_batches.n)
shallow_32.summary()
import matplotlib.pyplot as plt
%matplotlib inline
epochs=[1,2,3]
train_accuracy=[0.9868,0.9995,0.9997]#training_history1.history["acc"]
val_accuracy=[0.9998,0.9998,0.9998]#training_history1.history["val_acc"]
plt.plot(epochs,train_accuracy,label="Training accuracy")
plt.plot(epochs,val_accuracy,label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='center right')
plt.title("Accuracy plot")
shallow_64=Sequential()
shallow_64.add(Flatten(input_shape=(28,28,1)))
shallow_64.add(Dense(64,activation="relu"))
shallow_64.add(Dense(10,activation="softmax"))
shallow_64.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
shallow_64_training=shallow_64.fit_generator(generator=train_batches,
                                             steps_per_epoch=train_batches.n,
                                             epochs=1,
                                             validation_data=validation_batches,
                                             validation_steps=validation_batches.n)
shallow_64.summary()
import matplotlib.pyplot as plt
%matplotlib inline
epochs=[1,2,3]
train_accuracy=[0.9922,0.9997,0.9998]
val_accuracy=[1,1,1]
plt.plot(epochs,train_accuracy,label="Training accuracy")
plt.plot(epochs,val_accuracy,label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='center right')
plt.title("Accuracy plot")
deep_3layer=Sequential()
deep_3layer.add(Flatten(input_shape=(28,28,1)))
deep_3layer.add(Dense(16,activation="relu"))
deep_3layer.add(Dense(16,activation="relu"))
deep_3layer.add(Dense(10,activation="softmax"))
deep_3layer.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
deep_3layer_training=deep_3layer.fit_generator(generator=train_batches,
                                               steps_per_epoch=train_batches.n,
                                               epochs=1,
                                               validation_data=validation_batches,
                                               validation_steps=validation_batches.n)
deep_3layer.summary()
train_accuracy=[0.9691,0.9896,0.9949]
val_accuracy=[0.9905,0.9967,0.9979]
plt.plot(epochs,train_accuracy,label="Training accuracy")
plt.plot(epochs,val_accuracy,label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='center right')
plt.title("Accuracy plot")
deep_4layer=Sequential()
deep_4layer.add(Flatten(input_shape=(28,28,1)))
deep_4layer.add(Dense(16,activation="relu"))
deep_4layer.add(Dense(16,activation="relu"))
deep_4layer.add(Dense(16,activation="relu"))
deep_4layer.add(Dense(16,activation="relu"))
deep_4layer.add(Dense(10,activation="softmax"))
deep_4layer.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=["accuracy"])
deep_4layer_training=deep_4layer.fit_generator(generator=train_batches,
                                               steps_per_epoch=train_batches.n,
                                               epochs=1,
                                               validation_data=validation_batches,
                                               validation_steps=validation_batches.n)
deep_4layer.summary()
train_accuracy=[0.9688,0.9891,0.9931]
val_accuracy=[0.9886,0.9838,0.9947]
plt.plot(epochs,train_accuracy,label="Training accuracy")
plt.plot(epochs,val_accuracy,label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='center right')
plt.title("Accuracy plot")
#let us plot the number of trainable parameters in each model type
num_neurons=[26,42,74]
val_parameters_shallow=[12730,25450,50890]
val_parameters_deep=[12730,13002,13546]
plt.plot(num_neurons,val_parameters_shallow,label="shallow")
plt.plot(num_neurons,val_parameters_deep,label="deep")
plt.xlabel("Number of neurons")
plt.ylabel("Number of trainable parameters")
plt.legend(loc='upper left')