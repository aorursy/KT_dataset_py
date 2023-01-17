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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train.head()
# Save the Id 
train_id = train['Id']
test_id = test['Id']
sns.jointplot(x='LotFrontage', y='SalePrice', data=train, kind='reg', )
plt.grid(True)
train = train.drop(train[(train['LotFrontage']>250) & (train['SalePrice']<400000)].index)
sns.jointplot(x='LotFrontage', y='SalePrice', data=train, kind='reg', )
plt.grid(True)
sns.jointplot(x='LotArea', y='SalePrice', data=train, kind='reg' )
train = train.drop(train[(train['LotArea']>150000) & (train['SalePrice']<300000)].index)
sns.jointplot(x='LotArea', y='SalePrice', data=train, kind='reg' )
plt.grid(True)
sns.jointplot(x='BsmtFinSF1', y='SalePrice', data=train, kind='reg', )
plt.grid(True)
train = train.drop(train[(train['BsmtFinSF1']>1000) & (train['SalePrice']>600000)].index)
sns.jointplot(x='BsmtFinSF1', y='SalePrice', data=train, kind='reg')
plt.grid(True)
sns.jointplot(x='WoodDeckSF', y='SalePrice', data=train, kind='reg')
plt.grid(True)
train = train.drop(train[(train['WoodDeckSF']>0) & (train['SalePrice']>500000)].index)
sns.jointplot(x='WoodDeckSF', y='SalePrice', data=train, kind='reg')
plt.grid(True)
sns.jointplot(x='OpenPorchSF', y='SalePrice', data=train, kind='reg')
plt.grid(True)
train = train.drop(train[((train['OpenPorchSF']>500) & (train['SalePrice']<300000)) | ((train['OpenPorchSF']<100) 
              & (train['SalePrice']>500000))].index)
sns.jointplot(x='OpenPorchSF', y='SalePrice', data=train, kind='reg')
plt.grid(True)
sns.distplot(train['SalePrice'], fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

print('\n mean is {:.2f} and sigma is {:.2f} \n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'], fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

print('\n mean is {:.2f} and sigma is {:.2f} \n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
# Set 'Id' as Index for the Sake if Simplicity
train = train.set_index('Id')
test = test.set_index('Id')
train_num = train.shape[0]
test_num = test.shape[0]
df = pd.concat([train, test], axis=0)
df.head()
df['MSSubClass'] = df['MSSubClass'].astype('str')
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull())
df.isnull().sum()
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
num_feat = df.select_dtypes(exclude='object').columns
cat_feat = df.select_dtypes(include='object').columns
num_feat = num_feat[:-1]
num_feat
# Input mean value in Numerical Features
for col in num_feat:
    df[col] = df[col].fillna(df[col].mean())
    
# Input mode value in Categorical Features
for col in cat_feat:
    df[col] = df[col].fillna(df[col].mode()[0])
# Handle Features contains year values

df['YrSold_YearBuilt'] = df['YrSold'] - df['YearBuilt']
df['GarageYrBlt'] = df['GarageYrBlt'].astype('str')
df['YearRemodAdd'] = df['YearRemodAdd'].astype('str')
df.drop(['YrSold', 'YearBuilt'], axis=1, inplace=True)
train = df.iloc[:train_num, :]
test = df.iloc[train_num:, :]
# Plotting Heatmap 
corrmat = train.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corrmat, vmin=0, vmax=1, cmap='coolwarm')
#saleprice correlation matrix
k = 15 
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cf = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(16,10))
hm = sns.heatmap(cf, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                 cmap='coolwarm', xticklabels=cols.values)
train.drop(['GrLivArea', '1stFlrSF', 'OverallQual', 'GarageCars'], axis=1, inplace=True)
test.drop(['GrLivArea', '1stFlrSF', 'OverallQual', 'GarageCars'], axis=1, inplace=True)
df = pd.concat([train, test], axis=0)
df.head()
cat_feat = df.select_dtypes(include='object').columns
cat_feat
df_dummy = pd.get_dummies(df, columns=cat_feat, drop_first=True)
# Separate train and test set
train = df_dummy.iloc[:train_num, :]
test = df_dummy.iloc[train_num:, :]
test.drop('SalePrice', axis=1, inplace=True)
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=51)
model1 = xgb.XGBRegressor()
model1.fit(X_train, y_train)
xgb_pred = model1.predict(X_test)
r2_score(y_test, xgb_pred)
# Input Parameter values: 

param = {'max_depth': [3,5,6,8],
        'learning_rate': [0.05, 0.1, 0.15, 0.25, 0.3],
        'n_estimators': [100,200,300,500],
        'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'min_child_weight': [1,3,5,7]}
# Instantiate model
regressor = xgb.XGBRegressor()
random_search = RandomizedSearchCV(regressor, param_distributions=param, n_iter =5, scoring = 'neg_mean_squared_error',
                                   n_jobs=-1, cv=5, verbose=2)
random_search.fit(X_train, y_train)
random_search.best_estimator_
model2 = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0.1, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=3,
             min_child_weight=3, monotone_constraints='()',
             n_estimators=300, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
model2.fit(X, y)
test_pred = model2.predict(test)
test_pred = np.expm1(test_pred)
test_pred_df = pd.DataFrame(test_pred, columns=['SalePrice'])
test_id_df = pd.DataFrame(test_id, columns=['Id'])
submission = pd.concat([test_id_df, test_pred_df], axis=1)
submission.head()
# Save the predictions
submission.to_csv(r'submission.csv', index=False)