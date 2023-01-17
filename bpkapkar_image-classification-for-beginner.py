import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from keras.datasets import cifar10



(X_train, y_train), (X_test, y_test) = cifar10.load_data() #使用keras提供的api读取数据



num_classes = 10

#print(X_train.shape)

#print(y_train.shape)

#print(X_test.shape)

#print(y_test.shape)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
i=1001

plt.imshow(X_train[i])

print(y_train[i])

plt.ioff()
W_grid=15

L_grid=15

fig,axes = plt.subplots(L_grid,W_grid,figsize=(25,25))

axes=axes.ravel()

n_training=len(X_train)

for i in np.arange(0,L_grid * W_grid):

    index=np.random.randint(0,n_training) #Pick a random number 

    axes[i].imshow(X_train[index])

    axes[i].set_title(y_train[index]) #Prints labels on top of the picture

    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)
X_train=X_train.astype('float32')

X_test=X_test.astype('float32')
number_cat=10
y_train
import keras

y_train=keras.utils.to_categorical(y_train,number_cat)
y_train
y_test=keras.utils.to_categorical(y_test,number_cat)
y_test
X_train=X_train/255

X_test=X_test/255
X_train
X_train.shape
X_train
X_train.shape
Input_shape = X_train.shape[1:]
Input_shape
from keras.models import Sequential 

from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

cnn_model=Sequential()

cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',input_shape=Input_shape))

cnn_model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Dropout(0.4))



cnn_model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))

cnn_model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))

cnn_model.add(MaxPooling2D(2,2))

cnn_model.add(Dropout(0.4))



cnn_model.add(Flatten())



cnn_model.add(Dense(units=1024,activation='relu'))



cnn_model.add(Dense(units=1024,activation='relu'))



cnn_model.add(Dense(units=10,activation='softmax'))
cnn_model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.rmsprop(lr=0.001),metrics=['accuracy'])
history=cnn_model.fit(X_train,y_train,batch_size=32,epochs=2,shuffle=True)
evaluation=cnn_model.evaluate(X_test,y_test)

print('Test Accuracy: {}'.format(evaluation[1]))
predicted_classes=cnn_model.predict_classes(X_test)

predicted_classes
y_test=y_test.argmax(1)
y_test
L=7

W=7

fig,axes=plt.subplots(L,W,figsize=(12,12))

axes=axes.ravel()



for i in np.arange(0,L*W):

    axes[i].imshow(X_test[i])

    axes[i].set_title('Prediction= {}\nTrue={}'.format(predicted_classes[i],y_test[i]))

    axes[i].axis('off')

    plt.subplots_adjust(wspace=1)
from sklearn.metrics import confusion_matrix

import seaborn as sns

cm= confusion_matrix(y_test,predicted_classes)

cm

plt.figure(figsize=(10,10))

sns.heatmap(cm,annot=True)

plt.ioff()
#import os 

#directory=os.path.join(os.getcwd(),'saved_models')



#if not os.path.isdir(directory):

#    os.makedirs(directory)

#model_path=os.path.join(directory,'keras_cifar10_trained_model.h5')

#cnn_model.save(model_path)