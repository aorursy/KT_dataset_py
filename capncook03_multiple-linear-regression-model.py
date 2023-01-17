# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# reading the dataset

housing = pd.read_csv("/kaggle/input/multiple-linear-regression/Housing.csv")
housing
housing.shape
housing.info()
housing.describe()
# let's visualize a pairplot

sns.pairplot(housing)
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad',y = 'price',data = housing,palette = 'pastel')
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = housing,palette = 'pastel')
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = housing,palette = 'pastel')
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data = housing,palette = 'pastel')
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = housing,palette = 'pastel')
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = housing,palette = 'pastel')
plt.figure(figsize = (10,5))
sns.boxplot(x = 'furnishingstatus',y = 'price',data = housing,hue = 'airconditioning',palette = 'pastel')
# yes/no variables

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

housing[varlist] = housing[varlist].apply(lambda x: x.map({'yes':1,'no':0}))
housing
# furnishingstatus to dummy variables

status = pd.get_dummies(housing['furnishingstatus'])
status
status = pd.get_dummies(housing['furnishingstatus'],drop_first = True)
status
# concatenating with the housing dataframe

housing = pd.concat([housing,status],axis = 1)
housing
# dropping the furnishingstatus variable

housing.drop('furnishingstatus',axis = 1,inplace = True)
housing
import sklearn
from sklearn.model_selection import train_test_split
train,test = train_test_split(housing,test_size = 0.3,random_state = 100)
print(train.shape)
print(test.shape)
from sklearn.preprocessing import MinMaxScaler
# instantiating

scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
# rescaling the numeric variables
# we fit and transform on the train set

train[num_vars] = scaler.fit_transform(train[num_vars])
train
# we only transform on the test set

test[num_vars] = scaler.transform(test[num_vars])
test
plt.figure(figsize = (16, 10))
sns.heatmap(train.corr(),annot = True,cmap = "YlGnBu")
plt.scatter(x = 'area',y = 'price',data = housing)
X_train = train.drop('price',axis = 1)
y_train = train['price']
import statsmodels
import statsmodels.api as sm
# adding a constant
# using area as our predictor variable

X_train_sm = sm.add_constant(X_train['area'])
# creating the first model

lr1 = sm.OLS(y_train,X_train_sm).fit()
lr1.summary()
# adding the variable bathrooms
# creating the second model

X_train_sm = sm.add_constant(X_train[['area','bathrooms']])
lr2 = sm.OLS(y_train,X_train_sm).fit()
lr2.summary()
# building a model with all the variables
# creating the first model

X_train_sm = sm.add_constant(X_train)
lr1 = sm.OLS(y_train,X_train_sm).fit()
lr1.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# dropping the variable semi-furnished
# creating the second model

X = X_train.drop('semi-furnished',axis = 1)
X_train_sm = sm.add_constant(X)
lr2 = sm.OLS(y_train,X_train_sm).fit()
lr2.summary()
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# dropping the variable bedrooms
# creating the third model

X = X_train.drop(['semi-furnished','bedrooms'],axis = 1)
X_train_sm = sm.add_constant(X)
lr3 = sm.OLS(y_train,X_train_sm).fit()
lr3.summary()
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# dropping the variable basement
# creating the fourth model

X = X_train.drop(['semi-furnished','bedrooms','basement'],axis = 1)
X_train_sm = sm.add_constant(X)
lr4 = sm.OLS(y_train,X_train_sm).fit()
lr4.summary()
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# to check if the error terms are normally distributed or not

y_train_pred = lr4.predict(X_train_sm)
res = y_train - y_train_pred
sns.distplot(res)
# to check if the error terms follow any visible pattern or not

sns.scatterplot(x = y_train,y = res)
X_test = test.drop('price',axis = 1)
y_test = test['price']
# adding a constant

X_test_sm = sm.add_constant(X_test.drop(["bedrooms", "semi-furnished", "basement"], axis = 1))
# predictions

y_test_pred = lr4.predict(X_test_sm)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# RMSE

rmse = np.sqrt(mean_squared_error(y_true = y_test,y_pred = y_test_pred))
rmse
# R-squared value

r2 = r2_score(y_true = y_test,y_pred = y_test_pred)
r2
# Adj. R-squared value
# n = 164 and p = 10

Adj_r2 = 1-(1-r2)*(164-1)/(164-10-1)
Adj_r2