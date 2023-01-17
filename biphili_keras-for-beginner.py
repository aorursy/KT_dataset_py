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
import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_circles
X,y=make_circles(n_samples=1000,noise=0.1,factor=0.2,random_state=0)
X

X.shape
plt.figure(figsize=(5,5))

plt.plot(X[y==0,0],X[y==0,1],'ob',alpha=0.5)

plt.plot(X[y==1,0],X[y==1,1],'xr',alpha=0.5)

plt.xlim(-1.5,1.5)

plt.ylim(-1.5,1.5)

plt.legend(['0','1'])

plt.title('Blue circles and Red crosses')
from keras.models import Sequential

from keras.layers import Dense 

from keras.optimizers import SGD
model=Sequential()
model.add(Dense(4,input_shape=(2,),activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.compile(SGD(lr=0.5),'binary_crossentropy',metrics=['accuracy'])
model.fit(X,y,epochs=20)
hticks=np.linspace(-1.5,1.5,101)

vticks=np.linspace(-1.5,1.5,101)

aa,bb=np.meshgrid(hticks,vticks)

ab=np.c_[aa.ravel(),bb.ravel()]

c=model.predict(ab)

cc=c.reshape(aa.shape)
plt.figure(figsize=(5,5))

plt.contourf(aa,bb,cc,cmap='bwr',alpha=0.2)

plt.plot(X[y==0,0],X[y==0,1],'ob',alpha=0.5)

plt.plot(X[y==1,0],X[y==1,1],'xr',alpha=0.5)

plt.xlim(-1.5,1.5)

plt.ylim(-1.5,1.5)

plt.legend(['0','1'])

plt.title('Blue circles and Red crosses')