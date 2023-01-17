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
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print('Training set has {} rows and {} columns'.format(df.shape[0], df.shape[1]))
print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
df.isnull().sum().sort_values(ascending=False) / df.shape[0]
# drop the columns that have more than 50% of mising values
col_todrop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id', 'Utilities']
df = df.drop(col_todrop, axis=1)
test = test.drop(col_todrop, axis=1)
df.isnull().sum().sort_values(ascending=False).nlargest(16)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('darkgrid');

cat_toexplore = ['FireplaceQu', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual', 'BsmtExposure',
                 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType', 'Electrical']
numerical_toexplore = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']

for col in cat_toexplore:
    sns.barplot(x=df[col].value_counts().index, y=df[col].value_counts());
    plt.title('Count of {}'.format(col));
    plt.ylabel('');
    plt.show()
fillna_mode = cat_toexplore.copy()
for col in numerical_toexplore:
    sns.distplot(df[col], kde=False);
    plt.title('Distribution of {}'.format(col));
    plt.xlabel('');
    plt.show()
df[numerical_toexplore].describe()
fillna_mean = numerical_toexplore.copy()
for col in fillna_mode:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in fillna_mean:
    df[col] = df[col].fillna(df[col].mean())
df.isnull().sum().sum()
# test set as well
for col in fillna_mode:
    test[col] = test[col].fillna(test[col].mode()[0])
for col in fillna_mean:
    test[col] = test[col].fillna(test[col].mean())
test.isnull().sum().nlargest(15)
for col in ['MSZoning', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Exterior1st', 
            'Exterior2nd', 'KitchenQual', 'GarageCars', 'SaleType']:
    test[col] = test[col].fillna(test[col].mode()[0])
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']:
    test[col] = test[col].fillna(test[col].mean())
test.isnull().sum().sum()
dummies_cols = ['MSZoning', 'Street', 'LandContour', 'LotConfig',
                    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                    'Foundation',
                    'Heating', 'CentralAir', 'Electrical', 
                'GarageType',
                    'SaleType', 'SaleCondition', 'BsmtFinType1', 'BsmtFinType2', 'LotShape', 'Functional']

level_cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
              'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtExposure', 'GarageFinish', 'LandSlope',
              'PavedDrive', 'OverallCond', 
              'YrSold', 'MoSold']
df.shape
test.shape
train_copy = df.copy()
test_copy = test.copy()

total_copy = pd.concat([train_copy, test_copy], axis=0)
total_copy.shape
i = 0
for col in dummies_cols:
    print(col)
    
    dummies_df = pd.get_dummies(total_copy[col], drop_first=True)
    total_copy.drop([col], axis=1, inplace=True)
    if i == 0:
        final_df = dummies_df.copy()
    else:
        final_df = pd.concat([final_df, dummies_df], axis=1)
    i += 1
final_df = pd.concat([final_df, total_copy], axis=1)
final_df.shape
from sklearn.preprocessing import LabelEncoder

for col in level_cols:
    le = LabelEncoder()
    le.fit(list(final_df[col].values))
    final_df[col] = le.transform(list(final_df[col].values))
    
print(final_df.shape)
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.shape
train_df = final_df.iloc[:1460, :]
test_df = final_df.iloc[1460:, :]
test_df.drop(['SalePrice'], axis=1, inplace=True)
train_df.shape
test_df.shape
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

X = train_df.drop(['SalePrice'], axis=1)
y = train_df['SalePrice']
reg = XGBRegressor()
scores = cross_val_score(reg, X, y, cv=5)
print(scores)
print(scores.mean())
corr = train_df.corr()
corr
corr.loc['SalePrice', abs(corr['SalePrice']) > 0.5]
final_cols = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'FullBath', 'GarageCars']
X = X[final_cols]

reg = XGBRegressor()
scores = cross_val_score(reg, X, y, cv=5)
print(scores)
print(scores.mean())
X = train_df.drop(['SalePrice'], axis=1)
y = train_df['SalePrice']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)
scaled_test_X = scaler.fit_transform(test_df)

scaled_y = np.log1p(y)
from sklearn.model_selection import GridSearchCV

params = {'learning_rate': [0.01, 0.05, 0.1], 
          'booster': ['gbtree'],
          'n_estimators': [300, 900, 1300, 1700],
          'max_depth': [3, 5, 9],
          'min_child_weight': [1, 3, 5]}

xgb_grid = GridSearchCV(XGBRegressor(), params, cv=3, n_jobs=-1, verbose=True).fit(scaled_X, scaled_y)
xgb_grid.best_score_
xgb_grid.best_params_
reg = XGBRegressor(booster= 'gbtree',
                   learning_rate = 0.05,
                   max_depth = 3,
                   min_child_weight = 1,
                   n_estimators = 900)

scores = cross_val_score(reg, scaled_X, scaled_y, cv=5)
print(scores)
print(scores.mean())
from sklearn.model_selection import KFold

n_folds = 5

def rmse_cv(model, X, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(scaled_X)
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
score = rmse_cv(reg, scaled_X, scaled_y)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
reg.fit(scaled_X, scaled_y)

y_pred = reg.predict(scaled_test_X)
y_pred = np.expm1(y_pred)
y_pred
prediction = pd.DataFrame(y_pred)
sub_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission = pd.concat([sub_df['Id'], prediction], axis=1)
submission.columns = ['Id', 'SalePrice']
submission.to_csv('HP_submission3.csv', index=False)