import pandas as pd

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pickle

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',

               'dog', 'frog', 'horse', 'ship', 'truck']

data_dir='/kaggle/input/cifar10-python/cifar-10-batches-py/'

images_train=[]

labels_train=[]

def Load_Data():

    for i in range (5):

        filepath=os.path.join(data_dir,'data_batch_%d' %(i+1))

        print('loading data',filepath)

        with open (filepath,'rb') as f:

            data_dict=pickle.load(f, encoding='latin1')

            images_batch = data_dict['data']

            labels_batch = data_dict['labels']

            images_batch = images_batch.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

            labels_batch = np.array(labels_batch)

        images_train.append(images_batch)

        labels_train.append(labels_batch)

        X_train=np.concatenate(images_train)

        Y_train=np.concatenate(labels_train)

    filepath=os.path.join(data_dir,'test_batch')

    images_test=[]

    labels_test=[]

    with open (filepath,'rb') as f:

        print('loading data',filepath)

        data_dict=pickle.load(f, encoding='latin1')

        images_batch = data_dict['data']

        labels_batch = data_dict['labels']

        images_batch = images_batch.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

        labels_batch = np.array(labels_batch)

        images_test.append(images_batch)

        labels_test.append(labels_batch)

        X_test=np.concatenate(images_test)

        Y_test=np.concatenate(labels_test)

        print('finished loading ')

        return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test=Load_Data()

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
X_train_normalize=X_train.astype('float32')/255.0

X_test_normalize=X_test.astype('float32')/255.0
import matplotlib.pyplot as plt

def plot_image_labels_prediction_1(image,labels,idx,num=10):

    fig=plt.gcf()

    fig.set_size_inches(12,14)

    if num>25:num=25

    for i in range(0,num):

        ax=plt.subplot(5,5,i+1)

        ax.imshow(image[idx],cmap='binary')

        title=str(i)+','+class_names[labels[i]]

        ax.set_title(title,fontsize=10)

        ax.set_xticks([]);ax.set_yticks([])

        idx+=1

    plt.show()

plot_image_labels_prediction_1(X_test_normalize,Y_test,0,10)
from keras.utils import np_utils

Y_train=np_utils.to_categorical(Y_train)

Y_test=np_utils.to_categorical(Y_test)
import keras

from keras import layers

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.constraints import maxnorm

epochs=25

model=keras.Sequential()

model.add(layers.Conv2D(32,(3,3),input_shape=(32,32,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))

model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_constraint=maxnorm(3)))

model.add(layers.convolutional.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu',kernel_constraint=maxnorm(3)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(10,activation='softmax'))

lrate=0.01

decay=lrate/epochs

sgd=keras.optimizers.SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(x=X_train_normalize,y=Y_train,epochs=epochs,batch_size=32)
print(model.evaluate(X_test_normalize, Y_test, verbose=0))