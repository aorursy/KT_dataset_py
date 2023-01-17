# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from math import sqrt

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Big_Cities_Health_Data_Inventory.csv")

df.head()
df.rename(columns={"Indicator Category":"Indicator_Category"},inplace=True)

df.Indicator_Category.unique()
df=df.drop(["Source","Methods","Notes","BCHC Requested Methodology","Indicator"],axis=1)

df.head(30)
df.rename(columns={"Race/ Ethnicity":"Race"},inplace=True)

df.Race.unique()
df["Place_Info"] = df.Place.apply(lambda x : x[-2:])
df=df.drop(["Place"],axis=1)

df.head(30)
df.isna().sum()
df[df.isna().any(axis=1)]
df=df.dropna(subset=['Value'])

df_column_numeric = df.select_dtypes(include=np.number).columns

df_column_category = df.select_dtypes(exclude=np.number).columns

df_category_onehot = pd.get_dummies(df[df_column_category])

df_final = pd.concat([df_category_onehot,df[df_column_numeric]], axis = 1)

df_final.head()
x= df_final.drop(["Value"],axis=1)

x[0:5]





y=df_final["Value"]

y[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
from sklearn.neighbors import KNeighborsRegressor



### Training



#Lets start the algorithm with k=4 for now:



k = 5

#Train Model and Predict  

neigh = KNeighborsRegressor(n_neighbors = k).fit(X_train,y_train)

neigh
### Predicting

#we can use the model to predict the test set:



yhat = neigh.predict(X_test)

yhat[0:5]
rmse_val_test = [] #to store rmse values for different k

rmse_val_train=[]

for K in range(25):

    K_value = K+1

    neigh = KNeighborsRegressor(n_neighbors = K_value)

    neigh.fit(X_train, y_train) 

    train_y_pred = neigh.predict(X_train)

    error = sqrt(mean_squared_error(y_train,train_y_pred)) #calculate rmse

    rmse_val_train.append(error) #store rmse values

    test_y_pred = neigh.predict(X_test)

    error = sqrt(mean_squared_error(y_test,test_y_pred)) #calculate rmse

    rmse_val_test.append(error) #store rmse values

       

#plotting the rmse values against k values

curve = pd.DataFrame(rmse_val_train) #elbow curve 

curve.plot()
#plotting the rmse values against k values

curve = pd.DataFrame(rmse_val_test) #elbow curve 

curve.plot()
from sklearn.model_selection import GridSearchCV

k_range = list(range(1, 25))

param_grid = dict(n_neighbors=k_range)

knn = KNeighborsRegressor(param_grid)

model = GridSearchCV(knn, param_grid, cv=5)

model.fit(X_train,y_train)

model.best_params_
k_range = list(range(25))

elbow_curve_train = pd.Series(rmse_val_train,k_range)

elbow_curve_test = pd.Series(rmse_val_test,k_range)

ax=elbow_curve_train.plot(title="RMSE of train VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("RMSE of train")
ax=elbow_curve_test.plot(title="RMSE of test VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("RMSE of test")
