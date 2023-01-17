## Importing the basic libraries

import os

import re

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
## Importind the data into the system for futher analysis

print(os.listdir("../input"))

col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

housing=pd.read_csv('../input/housing.csv',delim_whitespace=True, names=col)
housing.head()
## Check the spread of the data and identify potential outliers at glance

housing.describe()
# Verify the type and categories of data and is there any missing values present that we need to take care

housing.info()
housing.hist(bins=20,figsize=(15,15))
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in housing.items():

    sns.boxplot(y=k, data=housing, ax=axs[index])

    index += 1
CRIM_data=pd.DataFrame(housing.iloc[:,0])

print(CRIM_data.describe())

CRIM_data.hist()

plt.title("Original CRIM Data")

min_cutoff=3.613524-3*8.601545

print("Minimum Cut off Value",min_cutoff)

max_cutoff=3.613524+3*8.601545

print("Maximum Cut Off Value",max_cutoff)

# Values beyond 29.4 must be dropped becasue they are potential outliers so dropping these 4 values

housing_df = housing.drop(housing[housing.CRIM>29.4].index)

print("Final shape of the dataset",housing_df.shape)
CRIM_data_new=pd.DataFrame(housing_df.iloc[:,0])

CRIM_data_new.hist()

plt.xlabel("Value of Criminal Rates")

plt.title("New CRIM Data")
ZN_data=pd.DataFrame(housing_df.iloc[:,1])

print(ZN_data.describe())

ZN_data.hist()
min_cutoff=11.546185-3*23.464449

print("Minimum Cut off Value",min_cutoff)

max_cutoff=11.546185+3*23.464449

print("Maximum Cut Off Value",max_cutoff)

housing_df_N = housing_df.drop(housing_df[housing.ZN>=82].index)

CRIM_data_new=pd.DataFrame(housing_df_N.iloc[:,1])

CRIM_data_new.hist()
housing_df_N['INDUS'].hist()

print(housing_df_N['INDUS'].describe())
housing_df_N['MEDV'].hist()
housing_df_N['MEDV'].describe()
lower_limit=22.370248-3*8.802114

upper_limit=22.370248+3*8.802114

print(upper_limit,lower_limit) ## No action been been taken due to small data size

housing_df_M = housing_df_N.drop(housing_df_N[housing_df_N.MEDV>48.77659].index)
housing_df_M['B'].hist()
housing_df_M['B'].describe()
upper_limit=358.160764+3*88.205368

lower_limit=358.160764-3*88.205368

print("Upper cut off value",upper_limit)

print("Lower Cut off Value",lower_limit) ## I will come back later on this based on model under fitting and over fitting
housing_df_O = housing_df_M.drop(housing_df_M[housing_df_M.B<92].index)
housing_df_O['B'].hist()
housing_df_O['NOX'].hist()
housing_df_O['NOX'].describe()
housing_df_O['RM'].describe()
housing_df_O['AGE'].hist()
housing_df_O['AGE'].unique()

housing_df_P = housing_df_O.drop(housing_df_O[housing_df_O.MEDV>48.77659].index)
housing_df_P.shape
housing_df_O['DIS'].describe()
housing_df_P['LSTAT'].describe()
lower_limit=12.522477-3*6.699438

upper_limit=12.522477+3*6.699438

print("Lower Limit",lower_limit)

print("Upper Limit",upper_limit)
housing_df_Q = housing_df_P.drop(housing_df_P[housing_df_P.MEDV>32.620791].index) ## rm the value>32.620791
housing_df_Q.shape
plt.figure(figsize = (15,12))

sns.heatmap(data=housing_df_Q.corr(), annot=True,linewidths=.8,cmap='Blues')
sns.pairplot(housing_df_Q,x_vars=["CRIM","ZN","INDUS"],y_vars =["MEDV"], kind="reg",height=6)
sns.pairplot(housing_df_Q,x_vars=["NOX","RM","AGE"],y_vars =["MEDV"], kind="reg",height=6)
sns.pairplot(housing_df_Q,x_vars=["DIS","RAD","TAX"],y_vars =["MEDV"], kind="reg",height=6)
sns.pairplot(housing_df_Q,x_vars=["PTRATIO","B","LSTAT"],y_vars =["MEDV"], kind="reg",height=6)
plt.figure(figsize = (9,9))

sns.heatmap(data=housing_df_Q.corr(), annot=True,linewidths=.8,cmap='Blues')
housing_df_Q.head()
import numpy as np

def split_train_test(data,test_ratio):

    shuffled_indicies=np.random.permutation(len(data))

    test_set_size=int(len(data)*test_ratio)

    test_indicies=shuffled_indicies[:test_set_size]

    train_indicies=shuffled_indicies[test_set_size:]

    return data.iloc[test_indicies],data.iloc[train_indicies]

test_set,train_set=split_train_test(housing_df_Q,0.2)
def training_and_testing_set(test_set,train_set):

    test_set_x=test_set.iloc[:,:-1]

    test_set_y=test_set.iloc[:,-1]

    train_set_x=train_set.iloc[:,:-1]

    train_set_y=train_set.iloc[:,-1]

    return test_set_x,test_set_y,train_set_x,train_set_y

test_set_x,test_set_y,train_set_x,train_set_y=training_and_testing_set(test_set,train_set)

print(test_set_x.shape,test_set_y.shape,train_set_x.shape,train_set_y.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

lin_reg=LinearRegression()

lin_reg.fit(train_set_x,train_set_y)

y_pred=lin_reg.predict(test_set_x)

mserr=mean_squared_error(test_set_y,y_pred)

print("Mean Squared Error",mserr)

root_mean_squared_error=np.sqrt(mserr)

print("Root Mean Squared Error",root_mean_squared_error)
import statsmodels.formula.api as sm
X=housing_df_Q.iloc[:,:-1].values

y=housing_df_Q.iloc[:,-1].values
X.shape
X1=np.append(arr=np.ones((403, 1),int).astype(int),values = X,axis =1)
# soring the optimal dataset in X_opt

X_opt = X1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X1[:,[0,1,2,5,6,8,9,10,11,13]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_opt = X1[:,[0,1,5,6,8,9,10,11,13]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()
X_train, X_test, y_train, y_test = train_test_split(X_opt,y,test_size = 0.3,random_state = 1)

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)



print("RMSE: %.2f"% np.sqrt(((y_pred - y_test) ** 2).mean()))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200,random_state=0)
rf.fit(train_set_x,train_set_y)
y_pred1=rf.predict(test_set_x)
mserr=mean_squared_error(test_set_y,y_pred1)

print("Mean Squared Error",mserr)

root_mean_squared_error=np.sqrt(mserr)

print("Root Mean Squared Error",root_mean_squared_error)
from sklearn.metrics import r2_score

r2 = r2_score(test_set_y,y_pred1)

r2