# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras import initializers
from keras.layers import Dense
from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/creditcard.csv')


X=df.iloc[:,1:30]
Y=df.iloc[:,30] 
Xtrain=X.iloc[0:280000]
Ytrain=Y.iloc[0:280000]
Xval=X.iloc[280000:284807,:]
Yval=Y.iloc[280000:284807]
len(Xval)
model = Sequential()
model.add(Dense(4, input_dim=29, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, Ytrain,epochs=1, batch_size=10)
scores = model.evaluate(Xval, Yval)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
def NNfun( hlayers ):
 model1=Sequential()
 model1.add(Dense(4, input_dim=29, activation='relu'))
 for x in range(hlayers-1):
     model1.add(Dense(4, activation='relu'))
 model1.add(Dense(1,activation='sigmoid'))
 model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model1.fit(Xtrain, Ytrain,epochs=1, batch_size=10)
 scores1 = model1.evaluate(Xval,Yval)
 return scores1;

def NNfunw(  ):
 model1=Sequential()
 myinit=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
 model1.add(Dense(4, input_dim=29, kernel_initializer=myinit,
                bias_initializer='zeros', activation='relu'))
 model1.add(Dense(1,activation='sigmoid'))
 model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model1.fit(Xtrain, Ytrain,epochs=1, batch_size=10)
 scores1 = model1.evaluate(Xval, Yval)
 return scores1;
Hypl=[]
for x in range(4,10):
 scr1 = NNfun(x)
 Hypl=Hypl + scr1
import matplotlib.pyplot as plt
Hypl1=[Hypl[1],Hypl[3],Hypl[5],Hypl[7],Hypl[9],Hypl[11]]
plt.plot([4,5,6,7,8,9],Hypl1)
plt.show()

Hypl2=[Hypl[0],Hypl[2],Hypl[4],Hypl[6],Hypl[8],Hypl[10]]
plt.plot([4,5,6,7,8,9],Hypl2)
plt.show()
Hypr=[]
for x in range(1,5):
 scr2 = NNfunw()
 Hypr=Hypr + scr1
Hypr1=[Hypr[0],Hypr[2],Hypr[4],Hypr[6]]
plt.plot([1,2,3,4],Hypr1)
plt.show()
Hypr2=[Hypr[1],Hypr[3],Hypr[5],Hypr[7]]
plt.plot([1,2,3,4],Hypr2)
plt.show()