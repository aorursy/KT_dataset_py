import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head(5)
test.head(5)
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
sns.heatmap(train.isnull(), cbar= False, cmap = 'YlGnBu')
df = pd.concat([train,test], axis = 0, sort = False)
df.shape
# Storing the columns with missing values in a list
Missing_val_col = [col for col in df.columns if df[col].isnull().sum() > 1]
Missing_val_col
# Checking the percentage of missing values. If any feature has more than 40% missing value then drop that feature. 
for col in Missing_val_col:
    if (df[col].isnull().sum() > int(0.40 * train.shape[0])):
        print(col)                   
train.drop(['Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1, inplace = True)
test.drop(['Alley', 'FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1, inplace = True)
#Checking the numerical features. 
numerical_data = [col for col in train.columns if train[col].dtypes != 'O']
train[numerical_data].head()
# Missing value features in numerical type column
numerical_missing = [col for col in numerical_data if train[col].isnull().sum() > 1 ]
numerical_missing
# Checking categorical features and treating missing values in that
category_data = [col for col in train.columns if train[col].dtypes == 'O']
train[category_data].head()

category_missing = [col for col in category_data if train[col].isnull().sum() > 1 ]
category_missing
# Filling missing data
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)
train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0], inplace = True)
train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0], inplace = True)
train['MSZoning'].fillna(train['MSZoning'].mode()[0], inplace = True)
train['Utilities'].fillna(train['Utilities'].mode()[0], inplace = True)
train['BsmtFullBath'].fillna(train['BsmtFullBath'].mode()[0], inplace = True)
train['BsmtHalfBath'].fillna(train['BsmtHalfBath'].mode()[0], inplace = True)
train['Functional'].fillna(train['Functional'].mode()[0], inplace = True)

test['LotFrontage'].fillna(test['LotFrontage'].mean(), inplace = True)
test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0], inplace = True)
test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode()[0], inplace = True)
test['MSZoning'].fillna(test['MSZoning'].mode()[0], inplace = True)
test['Utilities'].fillna(test['Utilities'].mode()[0], inplace = True)
test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0], inplace = True)
test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0], inplace = True)
test['Functional'].fillna(test['Functional'].mode()[0], inplace = True)

# Replacing categorical data with mode
for i in category_missing:
    train[i].fillna(train[i].mode()[0], inplace = True)
    test[i].fillna(test[i].mode()[0], inplace = True)
# Rechecking for the missing value
sns.heatmap(train.isnull(),cbar = False, cmap = 'twilight')
train_dummy = pd.get_dummies(train[category_data], columns = category_data)
train_dummy
# We Build a Regression model by splitting the training and test data
# Here I am considering the relationship of numerical features with saleprice. 
df_x = train[numerical_data]
df_x.drop('SalePrice', axis = 1)
X = df_x
y = train['SalePrice']
# Split the data into train and test with 70% data being used for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
import statsmodels.api as sm
X_add_constant = sm.add_constant(X_train)
model = sm.OLS(y_train, X_add_constant)
results = model.fit()
print(results.summary())
X_test = sm.add_constant(X_test)
y_pred = results.predict(X_test)
error = y_test - y_pred
sns.distplot(error)
