import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/mti-bootcamp-day-3/train_dataset.csv')

test = pd.read_csv('../input/mti-bootcamp-day-3/test_dataset.csv')

sample = pd.DataFrame(columns = ['Id', 'SalePrice'])

sample['Id'] = test['Id']
train.head()
train.describe()
train.columns
train.dtypes
train.isnull().sum().sort_values(ascending = False).head(10)
plt.figure(figsize= [15,15])

sns.heatmap(train.corr().abs(), annot = True, square = True)
train.corr()['SalePrice'].abs().sort_values(ascending = False)
sns.distplot(train['SalePrice'])
df_all = pd.concat([train, test], axis = 0)
categorical_col = ['Alley', 'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'FireplaceQu', 'GarageQual', 'HouseStyle', 'LotShape', 'MSZoning', 'MiscFeature', 

                   'PoolQC', 'Street', 'Utilities']



for col in categorical_col:

  df_all[col] = df_all[col].fillna('Tidak Ada')
# Missing Values imputation

numerical_col = ['BsmtFullBath', 'BsmtHalfBath', 'LotFrontage', 'TotalBsmtSF']

for col in numerical_col:

  df_all[col] = df_all[col].fillna(df_all[col].median())  
df_all.shape
df_all = pd.get_dummies(df_all, columns = categorical_col, drop_first = False)

df_all.shape
df_all['Total_Bath'] = df_all['FullBath'] + (0.5*df_all['HalfBath']) + df_all['BsmtFullBath'] + (0.5*df_all['BsmtHalfBath'])

df_all = df_all.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis = 1)
df_all.head()
df_all = df_all.drop(['Id', 'MiscVal'], axis = 1)
from sklearn.preprocessing import StandardScaler



train_preprocessed = df_all[:train.shape[0]]

test_preprocessed = df_all[train.shape[0]:]



X = train_preprocessed.drop(['SalePrice'], axis = 1)

y = train['SalePrice']

X_subm = test_preprocessed.drop(['SalePrice'], axis = 1)



scaler = StandardScaler()

X = scaler.fit_transform(X)

X_subm = scaler.transform(X_subm)
# Split Train - Validation Set

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1, shuffle = True, random_state = 101)
# Model Training

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error



model = LinearRegression()

model.fit(X_train, y_train)



print('MSLE train : ', mean_squared_log_error(y_train, abs(model.predict(X_train))))
# Validation

y_val_pred = model.predict(X_val)



print('MSLE : ', mean_squared_log_error(y_val, y_val_pred))
print('Intercept :' , model.intercept_)

print('Coef : ', model.coef_)
y_subm = model.predict(X_subm)
sample['SalePrice'] = y_subm
sample.to_csv('submission.csv', index = False)