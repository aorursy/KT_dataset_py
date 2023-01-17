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
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X,y = make_moons(n_samples = 1000,noise = 0.1,random_state = 0)
plt.plot(X[y==0 , 0], X[y==0,1], 'ob', alpha = 0.5)
plt.plot(X[y==1 , 0], X[y==1,1], 'ob', alpha = 0.5)
plt.legend(['0','1'])
X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
model = Sequential()
model.add(Dense(1,input_shape = (2,), activation = 'sigmoid'))
model.compile(Adam(lr = 0.05), 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train , epochs = 200, verbose =0)
results = model.evaluate(X_test,y_test)
print("the accuracy score on the train set is:\t{:0.3f}".format(results[1]))
def plot_decision_boundary(model,X,y):
    amin,bmin = X.min(axis = 0) - 0.1
    amax,bmax = X.max(axis = 0) + 0.1
    hticks = np.linspace(amin,amax,101)
    vticks = np.linspace(bmin,bmax,101)
    
    aa,bb = np.meshgrid(hticks,vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    
    c  = model.predict(ab)
    cc = c.reshape(aa.shape)
    
    plt.figure(figsize = (12,8))
    plt.contourf(aa,bb,cc,cmap = 'bwr', alpha = 0.2)
    plt.plot(X[y==0 , 0], X[y==0,1], 'ob', alpha = 0.5)
    plt.plot(X[y==1 , 0], X[y==1,1], 'xr', alpha = 0.5)
    plt.legend(['0','1'])
plot_decision_boundary(model,X,y)

model = Sequential()
model.add(Dense(4,input_shape = (2,),activation = 'tanh'))
model.add(Dense(2,activation='tanh'))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(Adam(lr = 0.05), 'binary_crossentropy' , metrics= ['accuracy'])

model.fit(X_train,y_train,epochs = 100,verbose = 0)

model.evaluate(X_test,y_test)
from sklearn.metrics import accuracy_score,confusion_matrix
y_train_pred = model.predict_classes(X_train)
y_test_pred = model.predict_classes(X_test)


#print("the accuracy score on the train set is:\t{:0.3f}".format(accuracy_score))
#print("the accuracy score on the test set is:\t{:0.3f}".format(accuracy_score))

plot_decision_boundary(model,X,y)




