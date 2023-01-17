import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random

import h5py



%matplotlib inline

# http://setosa.io/ev/image-kernels/
t = h5py.File('../input/train_happy.h5')

for key in t.keys():

    print(key)
s = h5py.File('../input/test_happy.h5')

for key in s.keys():

    print(key)
happy_trainging = h5py.File('../input/train_happy.h5')

happy_testing = h5py.File('../input/test_happy.h5')
X_train = np.array(happy_trainging['train_set_x'][:])

y_train = np.array(happy_trainging['train_set_y'][:])



X_test = np.array(happy_testing['test_set_x'][:])

y_test = np.array(happy_testing['test_set_y'][:])
print('X_train.shape = ',X_train.shape)

print('X_test.shape = ',X_test.shape)
y_train # target class
y_test.shape
y_test
index = random.randint(1,600)

plt.imshow(X_train[index])

print('index = ', index,', result = ', y_train[index])
W_grid = 5

Len_grid = 5



n_training =len(X_train)



fig,axes=plt.subplots(Len_grid,W_grid,figsize = (25,25))

axes = axes.ravel()



for i in np.arange(0, W_grid * Len_grid ):

    index = np.random.randint(0,n_training)

    axes[i].imshow(X_train[index])

    axes[i].set_title(y_train[index],fontsize = 25)

    axes[i].axis('off')

    
#Normalization

X_train = X_train / 255

X_test = X_test / 255
plt.imshow(X_train[1])
from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

cnn_model = Sequential()

cnn_model.add(Conv2D(64, 6, 6,input_shape = (64, 64, 3),activation = 'relu' ))

cnn_model.add(MaxPool2D(pool_size =(2, 2)))



cnn_model.add(Dropout(0.2))



cnn_model.add(Conv2D(64, 5, 5, activation = 'relu'))

cnn_model.add(MaxPool2D(pool_size =(2, 2)))





cnn_model.add(Flatten())



cnn_model.add(Dense(output_dim = 128, activation = 'relu'))



cnn_model.add(Dense(output_dim = 1, activation = 'sigmoid'))



cnn_model.compile(loss = 'binary_crossentropy',optimizer = Adam(lr = 0.001), metrics = ['accuracy'])
epochs = 50



history =cnn_model.fit(X_train, y_train, batch_size = 30, nb_epoch = epochs, verbose = 1)
predict_classes = cnn_model.predict_classes(X_test)
predict_classes
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test ,predict_classes)

sns.heatmap(cm, annot = True)
from sklearn.metrics import classification_report

print(classification_report(y_test,predict_classes ))
L = 10

W = 15

fig, axes = plt.subplots(L, W, figsize=(12, 12))

axes = axes.ravel()



for i in np.arange(0, L * W):

    if (predict_classes[i] != y_test[i]):

        axes[i].imshow(X_test[i])

        axes[i].set_title('prediction class ={}\n True Class = {}'.format(predict_classes[i],y_test[i]))

        axes[i].axis('off')

    

plt.subplots_adjust(wspace = 0.5)