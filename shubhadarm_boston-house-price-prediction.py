# Import required libraries

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.datasets import load_boston

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

%matplotlib inline
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
#Storing boston house data 

bos_data = load_boston()
#creating dataframe

bos_df = pd.DataFrame(data = bos_data.data)



print("Shape of data before adding target : ", bos_df.shape)
#adding feature names

bos_df.columns = bos_data.feature_names



#adding price fetaure from target

bos_df['PRICE'] = bos_data.target



print("Shape of data adding adding target : ", bos_df.shape)
#export data

bos_df.to_csv('Boston_House_Prices.csv', index= False)
#check for null values across column if any column has null values for all rows

bos_df.isnull().all(axis=0).value_counts()
#check for null values across rows if any row has null values for all columns

bos_df.isnull().all(axis=1).value_counts()
#check for null values across rows if any row has null values for any column

bos_df.isnull().any(axis=1).value_counts()
#check for null values across column if any column has null values for any row

bos_df.isnull().any(axis=0).value_counts()
#checking data types of features

bos_df.dtypes
#checking statistical summary

bos_df.describe(include="all")
#check correaltion with heatmap

plt.figure(figsize=(10,10))

sns.heatmap(data=bos_df.corr(), annot=True, cmap='viridis', fmt='.1f')
#checking RM using joint plot

sns.jointplot('RM', 'PRICE', data=bos_df, kind='reg')
#checking LSTAT using joint plot

sns.jointplot('LSTAT', 'PRICE', data=bos_df, kind='reg')
#checking relationships between variables

A = ['INDUS','RM','TAX','PTRATIO','LSTAT','PRICE']

sns.pairplot(bos_df[A])
# Taking input features and target

X = bos_df[['INDUS','RM','TAX','PTRATIO','LSTAT']]

Y = bos_df[['PRICE']]
X = StandardScaler().fit_transform(X)

Y = StandardScaler().fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
many_models = {}



many_models["Linear Regression"]        = {}

many_models["KNN Regressor"]           = {}

many_models["DecisionTree Regressor"]  = {}

many_models["SVR"]           = {}

many_models["XGBoost Regressor"]       = {}

many_models["RandomForest Regressor"]  = {}





many_models["Linear Regression"]["model"]        = LinearRegression()

many_models["KNN Regressor"]["model"]           = KNeighborsRegressor()

many_models["DecisionTree Regressor"]["model"]  = DecisionTreeRegressor()

many_models["SVR"]["model"]           = SVR()

many_models["XGBoost Regressor"]["model"]       = XGBRegressor(objective='reg:squarederror')

many_models["RandomForest Regressor"]["model"]  = RandomForestRegressor()



many_models
for k1, v1 in many_models.items():

    for k2,v2 in v1.items():

          print(v2)
for i in list(many_models):

    print(i)


for model,eval_parm in many_models.items():

    for metric in list(eval_parm.values()):

        #training mean square error

        model_obj = metric

        result = cross_val_score(model_obj, x_train, y_train.ravel(), cv=4, scoring='neg_mean_squared_error')

        many_models[model]["Training_MSE"] = round(result.mean(),3)

        #testing mean square error

        result = cross_val_score(model_obj, x_test, y_test.ravel(), cv=4, scoring='neg_mean_squared_error')

        many_models[model]["Testing_MSE"] = round(result.mean(),3)

        #training r2_score

        result = cross_val_score(model_obj, x_train,y_train.ravel(),cv=4, scoring='r2')

        many_models[model]["Training_r2"] = round(result.mean(),3)

        #testing r2_Score

        result = cross_val_score(model_obj, x_test,y_test.ravel(),cv=4, scoring='r2')

        many_models[model]["Testing_r2"] = round(result.mean(),3)

        

        

many_models
# model evaluation

pd.DataFrame(many_models) 
model = RandomForestRegressor()

model.fit(x_train, y_train.ravel())

y_test_pred = model.predict(x_test)
mean_squared_error(y_test,y_test_pred)
r2_score(y_test,y_test_pred)
# distribution plot of prediction of testing datasets

axs = sns.distplot(y_test, hist=False, color='b', label="Actual Testing values")

sns.distplot(y_test_pred, hist=False, color='r', label="Predicted Testing values", ax=axs)