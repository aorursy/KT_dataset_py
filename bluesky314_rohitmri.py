# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%matplotlib inline
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import keras 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
# %matplotlib
from matplotlib.animation import FuncAnimation


import os
print(os.listdir("../input"))
import keras
import tensorflow as tf

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
import numpy as np

# Any results you write to the current directory are saved as output.
import urllib

urllib.urlopen('http://www.gunnerkrigg.com//comics/00000001.jpg').read()
x_range=[.5,np.pi]
x_range
x=np.linspace(x_range[0],x_range[1],256)
y2=np.cos(1.8*x)-0.2

feats1=zip(x,y2)
 #np_utils.to_categorical(labels1, nb_classes=1)

#def create_data(x_range=x_range,n=256): # creates data in range of x_range, using y1,y2 functions for 2 classes 
                                        # y1,y2 can be altered to simulate any decision boundry
                                        #outputs y_range(for plotting), feats(inputs), labels(targets)
    
    
x=np.linspace(x_range[0],x_range[1],256)

y1=np.cos(1.8*x)+0.8
y2=np.cos(1.8*x)-0.2

feats1=zip(x,y1)
feats2=zip(x,y2)

labels1=np.zeros((256,)) # create classes
labels2=np.ones((256,))

feats=np.concatenate([feats1,feats2]) #join data
labels=np.concatenate([labels1,labels2])

labels=np_utils.to_categorical(labels, num_classes=2) # one hot encode targets(keras API intake)

y_range=[np.min(np.concatenate([y1,y2])),np.max(np.concatenate([y1,y2]))]

    #return(y_range,feats,labels)




y_range,feats,labels=create_data()

plt.scatter(feats[:,0],feats[:,1]) 


plt.show()




act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
act=Activation('relu')

model = Sequential()
model.add(Dense(4, input_shape=(2,)))
model.add(act)



model.add(Dense(2))
model.add(act)
# model.add(Dense(8))
# model.add(act)



model.add(Dense(2)) 


sgd = keras.optimizers.SGD(lr=0.00001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
history=model.fit(feats, labels, epochs=1200, batch_size=3)







def fun1(W):
    x=np.linspace(-W,W,1000)
    return(x,(1+(np.abs(x/W))**3))
for i in range(0,25,5):
    w=25+i
    a,b=fun1(w)
    plt.plot(a,b,label=str(w))
    plt.grid()
plt.show()
def Q(w):return (2.5*w)
for i in range(0,25,5):
    print(Q(25+i))
def ratio(W):
    B=2*np.pi*63.87
#     print(B)
    v=1.6*(np.sin(B*W)/(B*W))
    print(v)
    g=(2.4)/(B*W)**2
#     print(g)
    k=g*(np.cos(B*W)-(2*np.sin(B*W))/(B*W) + (np.sin(B*W/2)**2)/((B*W/2)**2))
    print(k)
    return(v+k)
x=[]
y=[]
for i in range(0,25,5):
    w=25+i
    a=ratio(w)
    x.append(w); y.append(a)
plt.plot(x,y,label=str(w))
plt.grid()
plt.show()
2*np.pi*63.87*10e6
def ge(beta,b,s,t):
    f=0.5*(1+(np.cosh(beta*(b-s-t)))/np.cosh(beta*(b-s)) )
    return(f)
def cap(beta,b,s,t,w):
    c=np.pi*8.85*10e-12*Q(w)**2
    c=1/c
    
    
def cf0(b,s):
    v=(b/(s*np.pi))*(np.log(1/(1-s/b) + ( (s/b)/(1-s/b) )*np.log(b/s)  ))
    return(v)
def z0o(b,s,t,w,ep):
    c=60*np.pi/np.sqrt(ep)
    
    denom=2*( (1+(t/s)*np.log(1+t/s) - (t/s)*np.log(t/s)))
    
    d=(   (w/(b-s) + cf0(b=b,s=s) + 2*denom/np.pi ))
       
    return(c/d)
def z0e(b,s,t,w,ep):
    c=60*np.pi/np.sqrt(ep)
    
    denom=np.log((b+2*t)/(b-s)) + ((s+2*t)/(b-s))*np.log((b+2*t) /(s+2*t)) 
    
    d=(   (w/(b-s) +0.443+denom/np.pi ))
    
    return(c/d)
    
    
    
def z0(b,s,t,w,ep):
    return(np.sqrt(z0o(b=b,s=s,t=t,w=w,ep=ep)*z0e(b=b,s=s,t=t,w=w,ep=ep)))
z0(b=150,s=2,t=2,w=50,ep=2.3)
