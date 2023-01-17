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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
bike_data = pd.read_csv("../input/bike_share.csv")
bike_data.info()
bike_data.head()
bike_data[bike_data.duplicated()]
bike_data.drop_duplicates(inplace = True)
bike_data[bike_data.duplicated()]
bike_data.isna().sum()
bike_data.corr()
X = bike_data[["season","workingday","temp","atemp","windspeed"]].values

X[0:3]
Y = bike_data[["count"]].values

Y[0:3]
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split (X,Y,test_size=0.3, random_state=42)
X_train.shape
X_test.shape
from sklearn.neighbors import KNeighborsRegressor

from math import sqrt

from sklearn.metrics import mean_squared_error
length = round(sqrt(bike_data.shape[0]))
length
rmse_dict = {}

rmse_list = []

for k in range(1,length+1):

    model = KNeighborsRegressor(n_neighbors = k).fit(X_train,Y_train)

    Y_predict = model.predict(X_test)

    rmse = sqrt(mean_squared_error(Y_test,Y_predict))

    rmse_dict.update({k:rmse})

    rmse_list.append(rmse)

    print("Rmse for k = {} is {}" .format(k,rmse))
key_min = min(rmse_dict.keys(), key=(lambda k: rmse_dict[k]))



print( "The miminum RMSE value is ",rmse_dict[key_min], "with k= ", key_min) 
elbow_curve = pd.DataFrame(rmse_list,columns = ['RMSE'])
elbow_curve.head()
elbow_curve.plot()