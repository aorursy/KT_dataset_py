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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.feature_selection import VarianceThreshold
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
#concat the both datasets so I get the same featrues for the test set
dataset = pd.concat([train,test])
dataset
#check which numerical features that have nan values
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_values = dataset.select_dtypes(include=numerics)
numerical_values.isnull().sum()
dataset.fillna(dataset.mean(), inplace=True)
#select all category features and check which of them has nan values
categorical_values = dataset.select_dtypes(include=object)
categorical_values.isnull().sum() / len(dataset)
#drop all columns aht have more than 80% missing values
dataset.drop(['MiscFeature', 'Fence', 'PoolQC', 'Alley', 'FireplaceQu'], axis=1, inplace=True)
#update teh categorical_values variable after removing some features
categorical_values = dataset.select_dtypes(include=object)
categorical_values = dataset.select_dtypes(include=object)
categorical_values
#create a function to find all categorical features that have more than 5 unique values
def find_features(dataset):
        
    features = []

    for x in range(len(dataset.columns)):
        if dataset[dataset.columns[x]].nunique() > 5:
            features.append(dataset.columns[x])
            
    return features
features_to_remove = find_features(categorical_values)
features_to_remove
#I remove all categorical features that have more than 5 unique values
dataset.drop(features_to_remove, axis=1, inplace=True)
categorical_values = dataset.select_dtypes(include=object)
categorical_values.isnull().sum()
#Fill all nan values for the categorical features
dataset = dataset.fillna(dataset.mode().iloc[0])
#create function to loop throug categorical features and add dummy values
def dummy_df(df, todummylist):
    for x in todummylist:
       dummies = pd.get_dummies(df[x], prefix=x, dummy_na = False)
       df = df.drop(x, 1)
       df = pd.concat([df, dummies], axis = 1)
    return df
#create a new dataset where the categorical variables are transformed to numerical features
dummies = list(categorical_values)
dataset_eda = dummy_df(dataset, dummies)
sns.countplot(data = train, x='MSZoning', palette='cool')
sns.barplot(data = train , x='MSZoning', y='SalePrice', palette='cool')
sns.countplot(data = train , x='LotShape', palette='GnBu')
sns.barplot(data = train , x='LotShape', y='SalePrice', palette='GnBu')
sns.countplot(data = train , x='LandContour', palette='cividis')
sns.barplot(data = train , x='LandContour', y='SalePrice', palette='cividis')
sns.barplot(data = train, x = 'OverallQual', y='SalePrice')
sns.scatterplot(data = train , x='GrLivArea', y='SalePrice')
sns.barplot(data = train , x='GarageCars', y='SalePrice', palette='mako')
sns.scatterplot(data = train , x='GarageArea', y='SalePrice', palette='mako')
#check the correlation efter I've transformed the categorical variables
corr=dataset_eda.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:20], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-20:])
#create function to decide which features that are correlated which eachother.
def correlation(dataset, threshold):
    
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr           
corr_features = correlation(dataset_eda, 0.8)
len(set(corr_features))
corr_features
dataset_eda.drop(labels=corr_features, axis=1, inplace=True)
dataset_eda
train_data = dataset_eda[:len(train)]
test_data = dataset_eda[len(train):]
train_data.shape, test_data.shape
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X = train_data.drop(['SalePrice', 'Id'], axis=1)
y= train_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#check how many features that have constant values
constants = VarianceThreshold(threshold=0)
constants.fit(X_train)
sum(constants.get_support())
#check which features that are constant and will be dropped
[x for x in X_train.columns if x not in X_train.columns[constants.get_support()]]
#remove the features that have constant values
X_train = constants.transform(X_train)
X_test = constants.transform(X_test)
#check how many features that have 99% the same values
quasi_constants = VarianceThreshold(threshold=0.01)
quasi_constants.fit(X_train)

sum(quasi_constants.get_support())
X_train = quasi_constants.transform(X_train)
X_test = quasi_constants.transform(X_test)
X_train.shape, X_test.shape
regressor = RandomForestRegressor(criterion = 'mse', max_depth = 5)
regressor.fit(X_train, y_train.ravel())
## Applying grid search  to find the best model and the best parameters
parameters = [{'n_estimators': [10,20,50,80,100,200,300,400, 500,600,800],
               'criterion': ['mae', 'mse'],
               'max_depth': [3,5,10,15,20]
                }]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, cv=10)

grid_search = grid_search.fit(X_train, y_train)

best_para = grid_search.best_params_
print(best_para)
y_pred = regressor.predict(X_test)
#check accuracy with Cross val score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())
r2_score(y_test, y_pred)