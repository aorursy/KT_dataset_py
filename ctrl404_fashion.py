n_feature = 784

n_class = 10

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

n_train = 60000

n_test = 10000
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# import os

# os.listdir('../input')

df = pd.read_csv('../input/fashion-mnist_train.csv')

y_train, x_train = df.iloc[:,0].values, df.iloc[:,1:].values.reshape(n_train,28,28)

df = pd.read_csv('../input/fashion-mnist_test.csv')

y_test, x_test = df.iloc[:,0].values, df.iloc[:,1:].values.reshape(n_test,28,28)
print(x_train.min(),x_train.max())

print(y_train.min(),y_train.max())

print(x_train.shape,y_train.shape)

print(x_test.shape,y_test.shape)
ind_plot = [1234,2345,3456,4567]

fig,axes = plt.subplots(1,4,sharey=True,figsize=(12,3))

for i,k in enumerate(ind_plot):    

    axes[i].imshow(x_train[k],cmap='gray')

    axes[i].set_title(class_names[y_train[k]])
from keras.utils import to_categorical
def process(X,Y):

    X = X[:,:,:,None].astype('float32')/255

    Y = to_categorical(Y) # one-hot

    return X,Y



x_train,y_train = process(x_train,y_train)

x_test,y_test = process(x_test,y_test)
# x_train.shape,x_train.dtype

# y_train.shape,y_train.dtype
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
model = Sequential()

model.add(Conv2D(32, 3, activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(2))

model.add(Conv2D(64, 3, activation='relu'))

model.add(MaxPooling2D(2))

model.add(Conv2D(64, 3, activation='relu'))

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(n_class,activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
hist = model.fit(x_train,y_train,epochs=20,batch_size=64)