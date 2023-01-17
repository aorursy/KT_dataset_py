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
df=pd.read_csv("../input/master.csv")

df.shape
df.head()
df.describe()
df.corr()
from sklearn.model_selection import train_test_split

x=df[["population"]]

y=df["suicides_no"]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.30)

from sklearn import neighbors

from sklearn.metrics import mean_squared_error 

import math

import matplotlib.pyplot as plt

%matplotlib inline
rmse_values = []

for k in range(150):

    k = k+1

    model = neighbors.KNeighborsRegressor(n_neighbors = k)



    model.fit(x_train, y_train)  #fit the model

    pred=model.predict(x_test) #make prediction on test set

    error =math.sqrt(mean_squared_error(y_test,pred)) #calculate rmse

    rmse_values.append(error) #store rmse values

    print('RMSE value for k= ' , k , 'is:', error)
min(rmse_values)
plt.plot(range(150),rmse_values)

plt.xlabel("K value")

plt.ylabel("RMSE")
predict=model.predict(x_test)
for i in range(len(predict)):

    print("For Population:",x_test.values[i])

    print("The expected number of suicides is :",predict[i])
K=[]

for x in range(1,151):

    j=1/x

    K.append(j)
plt.figure(figsize=(10,12))

plt.plot(K,rmse_values)

plt.xlabel("1/K")

plt.ylabel("RMSE Values")