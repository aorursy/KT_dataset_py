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
os.chdir('../input')

train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
train.head()
len(train['label'].unique())

train['label'].value_counts().sort_index()
import seaborn as sns
g = sns.countplot(train['label'])
train.describe()
train.isnull().values.any()
X_train = train.drop(labels = ['label'],axis=1)
X_train.head()
Y_train = train['label']
X_train = X_train/255

test = test/255
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
np.unique(Y_train,axis=0)
X_train_matrix = X_train.values.reshape(-1,28,28,1)

test_matrix = test.values.reshape(-1,28,28,1)
X_train_matrix.shape
test_matrix.shape
from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(X_train_matrix,Y_train,test_size  = 0.1,random_state = 2)
from keras.models import Sequential

from keras.layers.core import Flatten

from keras.layers import Conv2D,Dense,Dropout,MaxPool2D
model = Sequential()

model.add(Conv2D(filters=32,padding='Same',kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,padding='Same',kernel_size=(5,5),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(2,2),padding='Same',activation='relu'))

model.add(Conv2D(filters=64,kernel_size=(2,2),padding='Same',activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(2,2),activation='relu',padding='Same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(10,activation='softmax'))
model.summary()
from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator
optimizer = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 5,verbose = 1,factor = 0.3,min_lr = 0.00001)
epoch = 22

batch_size = 50
datagen = ImageDataGenerator(featurewise_center=False,

                            featurewise_std_normalization=False,

                            samplewise_center=False,

                            samplewise_std_normalization=False,

                            zca_whitening=False,

                            rotation_range=30,

                            zoom_range=0.1,

                            horizontal_flip=False,

                            vertical_flip=False,

                            width_shift_range=0.1,

                            height_shift_range=0.1)

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),

                             epochs=epoch,verbose=1,steps_per_epoch=X_train.shape[0]//batch_size,

                             validation_data = (X_val,Y_val),callbacks=[learning_rate_reduction])
import matplotlib.pyplot as plt
f,ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'],color='b',label='Training Loss')

ax[0].plot(history.history['val_loss'],color='r',label='Validation Loss',axes=ax[0])

legend = ax[0].legend(loc='best',shadow=True)

ax[1].plot(history.history['acc'],color='b',label='Accuracy')

ax[1].plot(history.history['val_acc'],color='r',label='Valoidation Accuracy')

legen = ax[1].legend(loc='best',shadow=True)
from sklearn.metrics import confusion_matrix

Y_pred = model.predict(X_val)

Y_pred = np.argmax(Y_pred,axis=1)

Y_true = np.argmax(Y_val,axis=1)

cm = confusion_matrix(Y_pred,Y_true)

plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)

plt.title('Confusion_matrix')

results = model.predict(test_matrix)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



os.chdir('../working')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)