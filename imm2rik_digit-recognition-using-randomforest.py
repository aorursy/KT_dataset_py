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
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data=pd.read_csv('../input/train.csv')
data.head()
d2=pd.read_csv('../input/test.csv')
d2.head()

import matplotlib.pyplot as plt
I=data.iloc[7,1:].values
I=I.reshape(28,28).astype('uint8')
plt.imshow(I)


d_x=data.iloc[:,1:]
d_y=data.iloc[:,0]
d2_y=data.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(d_x,d_y,test_size=0.1,random_state=4)
x_train.head()

d2_y.head()
y_train.head()
rf=RandomForestClassifier(n_estimators=100,criterion='gini')
rf.fit(x_train,y_train)

pred=rf.predict(x_test)
print (pred)

