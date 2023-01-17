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
## Load Data into Dataframes



import pandas as pd

train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

xtest=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head()
## Split train data into train and validation with stratified sampling



from sklearn.model_selection import train_test_split

Y=train['label']

xtrain, xcv, ytrain, ycv = train_test_split(train, Y, test_size=0.33, stratify=Y)
xtrain=xtrain[xtrain.columns[1:785]]

xcv=xcv[xcv.columns[1:785]]

print(xtrain.shape, ytrain.shape)

print(xcv.shape, ycv.shape)

from keras.utils import np_utils

import seaborn as sns

from keras.initializers import RandomNormal, he_normal
%matplotlib inline

import matplotlib.pyplot as plot

import numpy as np

import time



def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label='Validation Loss')

    ax.plot(x, ty, 'r', label='Train Loss')

    plt.legend()

    plt.grid()

    fig.canvas.draw()
xtrain=xtrain.to_numpy()

xcv=xcv.to_numpy()

ytrain=ytrain.to_numpy()

ycv=ycv.to_numpy()
xtest=xtest.to_numpy()
## normalize data (min-max normalization) - so that values will be in 0 - 1



xtrain=xtrain/255

xcv=xcv/255
print(ytrain[1])
## convert class labels into vectors



ytrain=np_utils.to_categorical(ytrain,10)

ycv=np_utils.to_categorical(ycv, 10)
print(ytrain[1])
from keras.models import Sequential

from keras.layers import Dense, Activation

## modelparams

outputdim=10

inputdim=xtrain.shape[1]

batchsize=128

nbepoch=20
## Multilayer Perceptron 

# MLP + BatchNorm +Dropout + RELU + ADAM   -   4 layers  -  (1024, 256, 128, 64)



from keras.layers.normalization import BatchNormalization

from keras.layers import Dropout





model1=Sequential()

'''

model1.add(Dense(1024,activation='relu',input_shape=(inputdim,),

                 kernel_initializer=he_normal(seed=None)))

model1.add(BatchNormalization())

model1.add(Dropout(0.5))

'''           

model1.add(Dense(256,activation='relu',input_shape=(inputdim,),

                 kernel_initializer=he_normal(seed=None)))

model1.add(BatchNormalization())

model1.add(Dropout(0.5))



model1.add(Dense(128,activation='relu',input_shape=(inputdim,),

                 kernel_initializer=he_normal(seed=None)))

model1.add(BatchNormalization())

model1.add(Dropout(0.5))

           

model1.add(Dense(64,activation='relu',input_shape=(inputdim,),

                 kernel_initializer=he_normal(seed=None)))

model1.add(BatchNormalization())

model1.add(Dropout(0.5))

           

model1.add(Dense(outputdim, activation='softmax'))

model1.summary()
model1.compile(optimizer='adam', loss='categorical_crossentropy', 

               metrics=['accuracy'])

history=model1.fit(xtrain, ytrain, batch_size=batchsize, 

                   epochs=nbepoch, verbose=1, validation_data=(xcv, ycv))
%matplotlib inline

import matplotlib.pyplot as plt

score=model1.evaluate(xcv, ycv, verbose=0)

print('CV Loss=', score[0])

print('CV Accuracy=', score[1])

fig, ax=plt.subplots(1,1)

ax.set_xlabel('epochs')

ax.set_ylabel('Categorical_CrossEntropy Loss')

x=list(range(1, nbepoch+1))

vy=history.history['val_loss']

ty=history.history['loss']

plt_dynamic(x, vy, ty, ax)
## Test class label predictions



ytest=model1.predict_classes(xtest)
ids=np.arange(1,28001)

print(ids)

print(ytest)

results_mlp=pd.DataFrame({'ImageId':ids, 'Label':ytest})

results_mlp.head()
filename='Digit_Recognizer_Predictions_MLP.csv'

results_mlp.to_csv(filename, index=False)

print('Saved File ', filename)