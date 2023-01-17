# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from scipy import stats as stat

from scipy.stats import norm, skew



# Any results you write to the current directory are saved as output.
!pip install regressors
import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

import statsmodels.formula.api as sm

import statsmodels.sandbox.tools.cross_val as cross_val

from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model as lm

from regressors import stats

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn import datasets

from math import sqrt

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut



print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')

df.columns
cols = ['SalePrice','Neighborhood', 'OverallQual','GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt','HouseStyle','TotRmsAbvGrd']

input = ['Neighborhood', 'OverallQual','GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt','HouseStyle','TotRmsAbvGrd']

output = ['SalePrice']

df_train_main = df[cols]

df_train = df[cols]

df_train.head()
sns.pairplot(df_train)
df_train.corr()
print("Check for NaN/null values:\n",df_train.isnull().values.any())

print("Number of NaN/null values:\n",df_train.isnull().sum())
inputDF = df_train_main[['Neighborhood', 'OverallQual','GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt','HouseStyle','TotRmsAbvGrd']]

outputDF = df_train_main[output]



X_train, X_test, y_train, y_test = train_test_split(inputDF, outputDF, test_size=0.2, random_state=0) 
corrmat = df_train_main.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
X_train = pd.get_dummies(X_train)

X_train = X_train.drop(['Neighborhood_Blueste'], axis=1)



model = lm.LinearRegression()

results = model.fit(X_train,y_train)



print("R - Squared value:\n",stats.adj_r2_score(model, X_train, y_train)) 



print(model.intercept_, model.coef_)
X_train.shape
X_test.shape
X_test = pd.get_dummies(X_test)

y_pred = model.predict(X_test) 

#print("Predicted value:\n", y_pred) 

#print("Originial value:\n", y_test) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
sns.distplot(df_train['SalePrice'] , fit=norm);

plt.legend(['Normal dist'],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stat.probplot(df_train['SalePrice'], plot=plt)

plt.show()
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



sns.distplot(df_train['SalePrice'] , fit=norm)



plt.legend(['Normal dist'],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stat.probplot(df_train['SalePrice'], plot=plt)

plt.show()
inputDF1 = df_train.iloc[:,1:]

outputDF1 = df_train.iloc[:,0:1]



X_train1, X_test1, y_train1, y_test1 = train_test_split(inputDF1, outputDF1, test_size=0.2, random_state=0) 



X_train1 = pd.get_dummies(X_train1)

X_train1 = X_train1.drop(['Neighborhood_Blueste'], axis=1)



model1 = lm.LinearRegression()

results1 = model1.fit(X_train1,y_train1)



print("R - Squared value:\n",stats.adj_r2_score(model1, X_train1, y_train1)) 



print(model1.intercept_, model1.coef_)
X_test1 = pd.get_dummies(X_test1)

y_pred1 = model1.predict(X_test1) 

#print("Predicted value:\n", y_pred) 

#print("Originial value:\n", y_test) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test1, y_pred1)))
regressor = DecisionTreeRegressor(random_state = 0, max_depth = 4)  

regressor.fit(X_train1, y_train1) 

y_pred_d = regressor.predict(X_test1)



print(sqrt(mean_squared_error(y_test1, y_pred_d)))
regr = RandomForestRegressor(random_state=0,

                            n_estimators=100, max_depth = 4)

regr.fit(X_train1, y_train1)

y_pred_rf = regr.predict(X_test1)



print(sqrt(mean_squared_error(y_test1, y_pred_rf)))
X_train1["GrLivArea"] = np.log1p(X_train1["GrLivArea"])

X_train1["TotRmsAbvGrd"] = np.log1p(X_train1["TotRmsAbvGrd"])

X_train1["TotalBsmtSF"] = np.log1p(X_train1["TotalBsmtSF"])

X_train1["OverallQual"] = np.log1p(X_train1["OverallQual"])

X_train1["GarageArea"] = np.log1p(X_train1["GarageArea"])

y_train1["SalePrice"] = np.log1p(y_train1["SalePrice"])



X_test1["GrLivArea"] = np.log1p(X_test1["GrLivArea"])

X_test1["TotRmsAbvGrd"] = np.log1p(X_test1["TotRmsAbvGrd"])

X_test1["TotalBsmtSF"] = np.log1p(X_test1["TotalBsmtSF"])

X_test1["OverallQual"] = np.log1p(X_test1["OverallQual"])

X_test1["GarageArea"] = np.log1p(X_test1["GarageArea"])

y_test1["SalePrice"] = np.log1p(y_test1["SalePrice"])
model1 = lm.LinearRegression()

results_n = model1.fit(X_train1,y_train1)

y_pred_n = model1.predict(X_test1) 

#print("Predicted value:\n", y_pred) 

#print("Originial value:\n", y_test) 

print("RMSE:\n", np.sqrt(metrics.mean_squared_error(y_test1, y_pred_n)))
regressor_n = DecisionTreeRegressor(random_state = 0, max_depth = 4)  

regressor_n.fit(X_train1, y_train1) 

y_pred_dn = regressor_n.predict(X_test1)



print(sqrt(mean_squared_error(y_test1, y_pred_dn)))
regr_n = RandomForestRegressor(random_state=0,

                            n_estimators=100, max_depth = 4)

regr_n.fit(X_train1, y_train1)

y_pred_rf_n = regr_n.predict(X_test1)



print(sqrt(mean_squared_error(y_test1, y_pred_rf_n)))