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
Health_df = pd.read_csv('../input/Big_Cities_Health_Data_Inventory.csv')
Health_df.info()
Health_df.head()
list(Health_df['Indicator'].unique())
Health_df.describe()
Health_df.apply(lambda x: len(x.unique()))
Health_df.isna().sum()
Health_df.dropna(subset=['Value'],inplace=True)
Health_df['Source'].value_counts()
list(Health_df['Source'].unique())
Health_df[Health_df['Value']==80977]
Health_df['Indicator Category'].unique()
Health_df[Health_df['Indicator Category']=='Demographics']
Health_df['BCHC Requested Methodology'].fillna(Health_df['BCHC Requested Methodology'].mode()[0],inplace=True)
Health_df['Source'].fillna(Health_df['Source'].mode()[0],inplace=True)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

cat_col = Health_df.select_dtypes(exclude=np.number).drop(['Notes','Methods'],axis = 1)

num_col = Health_df.select_dtypes(include=np.number)
cat_col = cat_col.apply(LabelEncoder().fit_transform)
final_df = pd.concat([cat_col,num_col],axis=1)
final_df.head()
final_df.corr()
from sklearn.linear_model import LinearRegression  

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
modelinput = final_df.drop(columns=['Value'],axis=1)

modeloutput = final_df['Value']
from sklearn import preprocessing

modelinput = preprocessing.StandardScaler().fit(modelinput).transform(modelinput.astype(float))
X_train,X_test,Y_train, Y_test = train_test_split(modelinput,modeloutput,test_size=0.3,random_state=123)
lm = LinearRegression()
lm.fit(X_train,Y_train)
print("Intercept value:", lm.intercept_)

print("Coefficient values:", lm.coef_)
Y_train_predict = lm.predict(X_train)

Y_test_predict = lm.predict(X_test)
print("MSE Train:",mean_squared_error(Y_train, Y_train_predict))

print("MSE Test:",mean_squared_error(Y_test, Y_test_predict))

print("RMSE Train:",np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("RMSE Test:",np.sqrt(mean_squared_error(Y_test, Y_test_predict)))

print('MAE Train', mean_absolute_error(Y_train, Y_train_predict))

print('MAE Test', mean_absolute_error(Y_test, Y_test_predict))

print('R2 Train',r2_score(Y_train, Y_train_predict))

print('R2 Test',r2_score(Y_test, Y_test_predict))
from sklearn.preprocessing import Imputer

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score

for K in range(50):

    K_value = K + 1

    neigh = KNeighborsRegressor(n_neighbors=K_value,weights='uniform',algorithm='auto')

    neigh.fit(X_train, Y_train)

    y_pred=neigh.predict(X_test)

    print("Accuracy is",r2_score(Y_test, y_pred)*100,"% for K-Value",K_value)