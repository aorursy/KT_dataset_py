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
from keras.models import *

from keras.layers import *

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv('../input/train.csv').values
data.shape
X=data[:,1:].reshape(-1,28,28,1)

Y=data[:,0]
X.shape
plt.imshow(X[4].reshape(28,28))


ip_layers=Input(shape=(28,28,1))

d1=Conv2D(32,(3,3))(ip_layers)

d2=Activation(activation='relu')(d1)

d3=MaxPool2D()(d2)

d1=Conv2D(64,(3,3))(d3)

d2=Activation(activation='relu')(d1)

d3=MaxPool2D()(d2)

d1=Conv2D(64,(3,3))(d3)

d2=Activation(activation='relu')(d1)

d3=MaxPool2D()(d2)

fl=Flatten()(d3)

d4=Dense(50,activation='relu')(fl)

d5=Dense(25,activation='relu')(d4)

out=Dense(10,activation='softmax')(d5)



model=Model(ip_layers,out)

model.summary()
ohe=OneHotEncoder(categories='auto')

Y_ohe=ohe.fit_transform(Y.reshape(-1,1)).todense()
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X,Y_ohe,epochs=10,batch_size=100,validation_split=0.2)
model.evaluate(X[600:1000],Y_ohe[600:1000])
test_data=pd.read_csv('../input/test.csv').values

output=pd.read_csv('../input/sample_submission.csv')
y_pred=np.argmax(model.predict(test_data.reshape(-1,28,28,1)),axis=1)
y_pred
np.argmax(model.predict(X[5].reshape(-1,28,28,1)),axis=1)
Y[5]
output['Label']=y_pred
output.head()
output.to_csv('submission.csv',index=None)