# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.datasets import make_blobs,make_circles
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Helper Functions
def plot_data(pl,X,y):
    pl.plot(X[y==0,0],X[y==0,1], 'ob',alpha=0.5)
    pl.plot(X[y==1,0],X[y==1,1], 'xr',alpha=0.5)
    pl.legend(['0','1'])
    return pl

def plot_decision_boundary(model,X,y):
    amin,bmin=X.min(axis=0)-0.1
    amax,bmax=X.max(axis=0)+0.1
    hticks = np.linspace(amin,amax,101)
    vticks = np.linspace(bmin,bmax,101)
    
    aa,bb = np.meshgrid(hticks,vticks)
    ab = np.c_[aa.ravel(),bb.ravel()]
    
    c=model.predict(ab)
    Z=c.reshape(aa.shape)
    
    plt.figure(figsize=(12,18))
    plt.contourf(aa,bb,Z,cmap='bwr',alpha=0.2)
    plot_data(plt,X,y)
    return plt
#X,y =make_blobs(n_samples=1000, centers=2,random_state=42)
X,y =make_circles(n_samples=1000, factor=.6,noise=.1,random_state=42)

pl = plot_data(plt,X,y)
pl.show()

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
#Importing Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input

#Sequential Model
'''
model = Sequential()

model.add(Dense(4,input_shape=(2,),activation='tanh', name='Hidden-1'))
model.add(Dense(4,activation='tanh', name='Hidden-2'))
model.add(Dense(1,activation='sigmoid',name='Output_Layer'))
'''
#Implement as Functional API

#Input
inputs = Input(shape=(2,))

#Hidden Layer
x = Dense(4,activation='tanh', name='Hidden-1')(inputs)
x = Dense(4,activation='tanh', name='Hidden-2')(x)

#Output Layer
o = Dense(1,activation='sigmoid',name='Output_Layer')(x)

model = Model(inputs=inputs, outputs=o)

model.summary()

model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
#Ploting model for documentation and understanding purpose
from keras.utils import plot_model
plot_model(model,show_shapes=True,show_layer_names=True,to_file='model.png')
#plot_model()
#Adding Callback
from keras.callbacks import EarlyStopping
my_callback=[EarlyStopping(monitor='val_acc',patience=5,mode=max)]

#Training Model using fit() method
model.fit(X_train,y_train,epochs=100, verbose=1,callbacks=my_callback, validation_data=(X_test,y_test))

eval_result = model.evaluate(X_test,y_test)
print('\nTest loss:', eval_result[0], '\nTest accuracy: ',eval_result[1])

plot_decision_boundary(model,X,y).show()
#save model
model.save('../output')
