# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
fashion_train_df=pd.read_csv('../input/fashion-mnist_train.csv',sep = ',')



fashion_test_df=pd.read_csv('../input/fashion-mnist_test.csv',sep = ',')
fashion_train_df.head()
fashion_test_df.tail()
print(fashion_test_df.shape)

print(fashion_train_df.shape)
training =np.array(fashion_train_df,dtype ='float32')
testing=np.array(fashion_test_df,dtype = 'float32')
import random

i=random.randint(1,60000)

plt.imshow(training[i,1:].reshape(28,28))

label=training[i,0]

label
#view of images in grid format

# Define the dimensions of the plot grid 



w_grid=15

l_grid=15



# fig,axes = plt.subplot(l_grid,w_grid)

# subplot return the figure object and axes object

# we can use the axes object to plot specific figures at various locations



fig,axes=plt.subplots(l_grid,w_grid,figsize=(17,17))



axes = axes.ravel() # flatten thr 15 X 15 matrix into 225 array 



n_training = len(training) # get the length of the training dataset



#select a random number from 0 t n_training



for i in np.arange(0,w_grid*l_grid): #create evenly spaces variables

    #select a random number

    

    index = np.random.randint(0,n_training)

    # read and disply and images with the selectd index

    axes[i].imshow(training[index,1:].reshape((28,28)))

    axes[i].set_title(training[index,0],fontsize = 8)

    axes[i].axis('off')

    

plt.subplots_adjust(hspace=0.4)
x_train=training[:,1:]/255

y_train= training[:,0]
x_test= testing[:,1:]/255

y_test=testing[:,0]
from sklearn.model_selection import train_test_split

X_train,X_vali,y_train,y_vali=train_test_split(x_train,y_train,test_size = 0.2, random_state = 12345)
X_train =X_train.reshape(X_train.shape[0],*(28,28,1))

x_test = x_test.reshape(x_test.shape[0],*(28,28,1))

X_vali = X_vali.reshape(X_vali.shape[0],*(28,28,1))
X_train.shape
import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard
clf= Sequential()

clf.add(Conv2D(32,3,3,input_shape= (28,28,1),activation='relu'))

clf.add(MaxPooling2D(pool_size =(2,2)))

clf.add(Flatten())

clf.add(Dense(output_dim= 32,activation='relu'))

clf.add(Dense(output_dim= 10,activation='sigmoid'))
clf.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics =['accuracy'])
clf.fit(X_train,

       y_train,

       batch_size= 512,

       nb_epoch=50,

        verbose=1,

        validation_data=(X_vali,y_vali))
evaluation=clf.evaluate(x_test,y_test)

print('Test Accuracy:{:.3f}'.format(evaluation[1]))
pred=clf.predict_classes(x_test)
L=5

W=5



fig,axes=plt.subplots(L,W,figsize=(12,12))

axes=axes.ravel()



for i in np.arange(0,L*W):

    axes[i].imshow(x_test[i].reshape(28,28))

    axes[i].set_title('Prediction Class = {:0.1f}\n True class = {:0.1f}'.format(pred[i],y_test[i]))

    axes[i].axis('off')

    

plt.subplots_adjust(wspace=0.5)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,pred)

plt.figure(figsize=(14,10))

sns.heatmap(cm,annot=True)
from sklearn.metrics import classification_report



num_classes=10

target_names=["class {}".format(i)for i in range (num_classes)]



print(classification_report(y_test,pred,target_names=target_names))
