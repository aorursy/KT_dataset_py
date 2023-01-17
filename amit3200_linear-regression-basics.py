# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input")[0])

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
df=pd.read_csv("..//input//train.csv")



#print(df)#print(d 

print(df.isnull().sum())

df.fillna(" ")

x=df["x"].tolist()

y=df["y"].tolist()

plt.scatter(x,y)
from sklearn.model_selection import train_test_split 



X_train,X_test,Y_train,Y_test = train_test_split(df[['x']],df[['y']],test_size=0.3)



from sklearn.linear_model import LinearRegression
model=LinearRegression()

X_train=X_train.fillna(X_train.mean())

Y_train=Y_train.fillna(Y_train.mean())
model.fit(X_train,Y_train)
y_predicts = model.predict(X_test)

model.coef_