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
df=pd.read_csv("../input/insurance.csv",sep=",")

df.head()
df.sex=[1 if each=="female" else 0 for each in df.sex]

df.head()
df.smoker=[1 if each=="no" else 0 for each in df.smoker]

df.head()
df.region[df.region=="southwest"]=1

df.region[df.region=="southeast"]=2

df.region[df.region=="northwest"]=3

df.region[df.region=="northeast"]=4

df.head()
x=df.iloc[:,:-1].values

y=df.charges.values.reshape(-1,1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

multiple_linear_reg=LinearRegression()

multiple_linear_reg.fit(x_train,y_train)

y_head=multiple_linear_reg.predict(x_test)

print("y_head",y_head)
from sklearn.metrics import r2_score

score=r2_score(y_test,y_head)

print("r2 score",score)