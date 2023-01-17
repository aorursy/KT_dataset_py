import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
from collections import Counter
warnings.filterwarnings('ignore')
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import norm
from scipy import stats
from scipy.stats import skew

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/house-prices-advanced-regression-techniques/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
# Check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# Check data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.describe()
# Checking Categorical Data
train.select_dtypes(include=['object']).columns
# Checking Categorical Data
test.select_dtypes(include=['object']).columns
# Checking Numerical Data
train.select_dtypes(include=['int64','float64']).columns
cat = len(train.select_dtypes(include=['object']).columns)
num = len(train.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')
# Checking Numerical Data
test.select_dtypes(include=['int64','float64']).columns
cat = len(test.select_dtypes(include=['object']).columns)
num = len(test.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')
# Correlation Matrix Heatmap
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
# Top 10 Heatmap
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr
# Overall Quality vs Sale Price
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
# Living Area vs Sale Price
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')
# Removing outliers manually (Two points in the bottom right)
train = train.drop(train[(train['GrLivArea']>4000) 
                         & (train['SalePrice']<300000)].index).reset_index(drop=True)
train = train.drop(train[train['GrLivArea']>4000].index).reset_index(drop= True)
# Living Area vs Sale Price
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')
# Garage Cars Area vs Sale Price
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
# Removing outliers manually (More than 4-cars, less than $300k)
train = train.drop(train[(train['GarageCars']>3) 
                         & (train['SalePrice']<300000)].index).reset_index(drop=True)
# Garage Area vs Sale Price
sns.boxplot(x=train['GarageCars'], y=train['SalePrice'])
# Garage Area vs Sale Price
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
# Removing outliers manually (More than 1000 sqft, less than $300k)
train = train.drop(train[(train['GarageArea']>1000) 
                         & (train['SalePrice']<300000)].index).reset_index(drop=True)
# Garage Area vs Sale Price
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')
# Basement Area vs Sale Price
sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg')
# First Floor Area vs Sale Price
sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg')
# Total Rooms vs Sale Price
sns.boxplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'])
# Total Rooms vs Sale Price
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
for i in train.columns:
    if(train[i].isnull().sum()/train.shape[0]>0.4):
        train.drop(i,axis = 1,inplace=True)
for i in test.columns:
    if(test[i].isnull().sum()/test.shape[0]>0.4):
        test.drop(i,axis = 1,inplace=True)       
for i in train.columns:
    if (train[i].dtype == object):
        m = max(set(list(train[i])), key=list(train[i]).count)
        train[i].fillna(m,inplace=True)
    else:
        train[i].fillna(train[i].mean(),inplace=True)
    
for i in test.columns:
    if (test[i].dtype == object):
        m = max(set(list(test[i])), key=list(test[i]).count)
        test[i].fillna(m,inplace=True)
    else:
        test[i].fillna(test[i].mean(),inplace=True)
train.shape
sns.distplot(train["SalePrice"])
sns.distplot(np.log(train["SalePrice"]))
train["SalePrice"] = np.log(train["SalePrice"])
#histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea'] = np.log(train['GrLivArea'])
#histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
cols = []
for i in train.columns:
    if (train[i].dtype==object):
        cols.append(i)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cols:
    train[i] = le.fit_transform(train[i])
    test[i] = le.transform(test[i])

X_training = train.drop(["Id", "SalePrice"], axis=1)
y_training = train["SalePrice"]
X_test = test.drop("Id", axis=1)
X_training.columns
X_test.columns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb

models = []
models.append(('LR', LinearRegression()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor(n_estimators=10)))
models.append(('Lasso', Lasso()))
models.append(('Ridge', Ridge()))
models.append(('XGB', xgb.XGBRegressor(max_depth=10, n_estimators=300, learning_rate=0.05)))
# evalutate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X_training, y_training, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
lr = LinearRegression()
lr.fit(X_training, y_training)
submission_predictions = np.exp(lr.predict(X_test))
submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": submission_predictions
    })

submission.to_csv("saleprice.csv", index=False)
print(submission.shape)
