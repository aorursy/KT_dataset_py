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
df=pd.read_csv('/kaggle/input/zaloni-techniche-datathon-2019/train.csv')

df.head(10)
df=df.dropna(subset=['last_name','first_name'],how='all')


from sklearn.model_selection import train_test_split
X1=np.asanyarray(df[['last_name']])

X2=np.asanyarray(df[['first_name']])

Y1=np.asanyarray(df[['gender']])

Y2=np.asanyarray(df[['race']])
X=np.c_[X1,X2]

for f in range(85265):

    s=type(X[f][0])

    if s== float:

        X[f][0]=X[f][1] 

for f in range(85265):

    s=type(X[f][1])

    if s==float:

        X[f][1]=X[f][0]
x1=[]

x2=[]

import math

def convertToNumber(s):

        return int.from_bytes(s.encode(), 'little')

for f in range(85265):

    s=X[f][0]

    i=convertToNumber(s)

    x1.append(i)

    

for f in range(85265):

    s=X[f][1]

    i=convertToNumber(s)

    x2.append(i)

x=np.c_[x1,x2]

y1=[]

y2=[]
for f in Y1:

    if f=='m':

        y1.append(1)

    if f=='f':

        y1.append(0)

y1=np.array(y1)   
train_x,test_x,train_y,test_y=train_test_split(x,y1,test_size=.3,random_state=5)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5).fit(train_x,train_y)
knn.score(train_x,train_y)
knn.score(test_x,test_y)
df1=pd.read_csv('/kaggle/input/zaloni-techniche-datathon-2019/test.csv')

df1.head(10)

df1=df1.dropna(subset=['last_name','first_name'],how='all')

df1.isnull().sum()

df1.shape
X3=np.asanyarray(df[['last_name']])

X4=np.asanyarray(df[['first_name']])
X5=np.c_[X3,X4]

for f in range(12185):

    s=type(X5[f][0])

    if s== float:

        X5[f][0]=X5[f][1] 

for f in range(12185):

    s=type(X5[f][1])

    if s==float:

        X5[f][1]=X5[f][0]        
x3=[]

x4=[]

import math

def convertToNumber(s):

        return int.from_bytes(s.encode(), 'little')

for f in range(12185):

    s=X5[f][0]

    i=convertToNumber(s)

    x3.append(i)

    

for f in range(12185):

    s=X5[f][1]

    i=convertToNumber(s)

    x4.append(i)

x5=np.c_[x3,x4]
predict=knn.predict(x5)

print(predict)

x6=np.asanyarray(predict)