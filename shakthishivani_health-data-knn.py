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

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
health_data=pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")
health_data.info()
health_data.head()
health_data = health_data.drop(columns=["BCHC Requested Methodology","Source","Methods","Notes"])
health_data[health_data.duplicated()]
health_data.drop_duplicates(inplace= True)
health_data[health_data.duplicated()]
health_data.isna().sum()
health_data.dropna(inplace = True)
health_data.isna().sum()
health_data["City"] = health_data["Place"].apply(lambda x:x.split('(')).str[0]
health_data.head(3)
categorical_cols = health_data.select_dtypes(exclude =np.number).columns
encoded_cols = pd.get_dummies(health_data[categorical_cols])
final_data = pd.concat([encoded_cols,health_data["Value"]],axis=1)
final_data.columns
Y = final_data[['Value']].values

Y[0:3]
X = final_data.drop(columns = ["Value"])

X[0:3]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split (X,Y,test_size=0.3, random_state=42)
X_train.shape
X_test.shape
from sklearn.neighbors import KNeighborsRegressor

from math import sqrt

from sklearn.metrics import mean_squared_error
length = round(sqrt(final_data.shape[0]))
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