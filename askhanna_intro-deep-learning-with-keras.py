# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau





sns.set(style='white', context='notebook', palette='deep')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
y_train = train_data['label']

X_train = train_data.drop(columns=['label'],axis=1)



del train_data
print(X_train.shape)

print(y_train.shape)

print(y_train.value_counts())
sns.countplot(y_train)
X_train.isnull().any().describe()
test_data.isnull().any().describe()
X_train = X_train/255.

test_data = test_data/255.
y_train = to_categorical(y_train, num_classes = 10)
X_train = X_train.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
print(X_train.shape)

#print(X_train[0])
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,random_state = 42,test_size=0.1)
y_train[0]
g = plt.imshow(X_train[0][:,:,0])

network = Sequential()

network.add(Dense(512,activation='relu',input_shape = (28,28,1)))

network.add(Flatten())

#network.add(Dense(512, activation='relu'))

network.add(Dense(10, activation='softmax'))

print("input shape ",network.input_shape)

print("output shape ",network.output_shape)



network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
network.fit(X_train, y_train, epochs=5, batch_size=128)
history = network.fit(X_train,y_train,epochs=20,batch_size=512,validation_data=(X_val,y_val))
history_dict = history.history

history_dict.keys()
loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



epochs = range(1,len(acc_values)+1)



plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()