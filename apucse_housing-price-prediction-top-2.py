# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv')

print(train_data.shape)
print(test_data.shape)
train_data.columns
test_data.columns
train_data.info()
train_data.head()
tr_data = train_data.copy()
tr_data = pd.get_dummies(tr_data, drop_first=True)
tr_data.head()
tr_data.shape
cols = tr_data.columns[1:15]
cols
fig, ax = plt.subplots(len(cols), figsize=(8,55))
fig.subplots_adjust(hspace=1)
for i, c in enumerate(cols):
    ax[i].scatter(tr_data[c], tr_data['SalePrice'])
    ax[i].set_yticks(range(0, tr_data['SalePrice'].max(), 100000))
    #ax[i].grid()
    ax[i].set_title(c)
plt.show()
tr_data.describe()
#ax1=sns.boxplot(y='MSSubClass', data=tr_data)
fig, ax = plt.subplots(len(cols[1:]),1, figsize=(6,30))
fig.subplots_adjust(hspace=1)
for i, col in enumerate(cols[1:]):
    plt.sca(ax[i])
    #plt.figure(i)
    sns.set(style="whitegrid")
    sns.boxplot(x=col, data=tr_data)
    #.swarmplot(x=col, data=tr_data, color="gray")    
fig, bx = plt.subplots(len(cols[1:]),1, figsize=(6,30))
fig.subplots_adjust(hspace=1)
for i, col in enumerate(cols[1:]):
    plt.sca(bx[i])
    #plt.figure(i)
    sns.set(style="whitegrid")
    sns.violinplot(x=col, data=tr_data, color='0.3')
    #sns.swarmplot(x=col, color="k", size=3, data=tr_data, ax = v.ax);
outliers = []



outliers.extend(tr_data[tr_data['OverallQual']==10][tr_data['SalePrice']<200_000].index.tolist())
outliers.extend(tr_data[tr_data['LotArea']>100_000].index.tolist())
outliers.extend(tr_data[tr_data['LotFrontage']>300].index.tolist())
outliers.extend(tr_data[tr_data['YearBuilt']<1900][tr_data['SalePrice']>200_000].index.tolist())
outliers.extend(tr_data[tr_data['YearRemodAdd']<2000][tr_data['SalePrice']>600_000].index.tolist())
outliers.extend(tr_data[tr_data['MasVnrArea']==1600].index.tolist())
outliers.extend(tr_data[tr_data['TotalBsmtSF']>3000][tr_data['SalePrice']<300_000].index.tolist())
outliers.extend(tr_data[tr_data['1stFlrSF']>2700][tr_data['SalePrice']<500_000].index.tolist())
outliers.extend(tr_data[tr_data['BsmtFullBath']==3.0].index.tolist())
outliers.extend(tr_data[tr_data['GrLivArea']>3300][tr_data['SalePrice']<300_000].index.tolist())
outliers.extend(tr_data[tr_data['FullBath']==0.0][tr_data['SalePrice']>300_000].index.tolist())
outliers.extend(tr_data[tr_data['GarageArea']>1200][tr_data['SalePrice']<200_000].index.tolist())
outliers.extend(tr_data[tr_data['OpenPorchSF']>500].index.tolist())


outliers=np.unique(outliers)
print(outliers)
print(len(outliers))
train_data.drop(outliers, axis=0, inplace=True)
y = train_data['SalePrice']
train_data.shape
df = pd.concat([train_data.drop(['SalePrice'], axis=1), test_data], join='outer')
df.drop(['Id'], axis=1)
df.shape
df.MSSubClass.isna().sum()
isEmpty = [x for x in df if df[x].isna().sum() != 0]
print(isEmpty)

data_float=[]
data_int = []
for x in isEmpty:
    if df[x].dtype == 'float64':
        print(x)
        data_float.append(x)
    if df[x].dtype == 'int64':
        print(x)
        data_int.append(x)

print(len(isEmpty))        
print(len(data_float))
print(len(data_int))
to_mode = [
    'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'Functional',
    'SaleType', 'LotFrontage'
]

to_none = [
    'Alley', 'MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
    'MiscFeature'
]

to_zero = [
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea'
]
for x in to_mode:
    df[x+'for_na'] = df[x].apply(lambda a:0 if pd.isnull(a)==True else 1)
    df[x] = df[x].fillna(df[x].mode()[0])
    
df.shape
for x in to_zero:
    df[x+'for_na'] = df[x].apply(lambda a:0 if pd.isnull(a)==True else 1)
    df[x] = df[x].fillna(df[x].mean())
for x in to_none:
    df[x+'for_na'] = df[x].apply(lambda a:0 if pd.isnull(a)==True else 1)
    df[x] = df[x].fillna('None')
    
df.shape
df.isna().sum().sum()
train_data.corr()[-1:]
cols_to_drop = [ 'YrSold', 'MoSold', 'BsmtHalfBath', 'BsmtFinSF2', 'KitchenAbvGr',
                'LowQualFinSF', 'BedroomAbvGr', '3SsnPorch', 
               ]

train_data.corr()[-1:][cols_to_drop]
df.drop(cols_to_drop, axis=1, inplace=True)
df.shape
df = pd.get_dummies(df)
print(df.isna().sum().sum())
df.head(1)
df.shape
X = df[:train_data.shape[0]]
train = df[:train_data.shape[0]]
test = df[train_data.shape[0]:]
train.shape, test.shape
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
X = StandardScaler().fit_transform(X)
test = StandardScaler().fit_transform(test)
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
#Ridge
params = {
    'alpha': [25, 35],
    'max_iter': [None, 1000, 5000],
    'solver': ['svd', 'lsqr', 'sag', 'saga', 'sparse_cg', 'sparse_cg']
}

M1 = GridSearchCV(
    Ridge(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)
M1.fit(X, y)

print(M1.best_estimator_)

mean_absolute_error(y_test, M1.predict(X_test))
# Lasso
params = {
    'alpha': [0.1, 1, 3],
    'max_iter': [50000],
}

M2 = GridSearchCV(
    Lasso(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)

M2.fit(X, y)
print(M2.best_estimator_)

mean_absolute_error(y_test, M2.predict(X_test))
# SVC
params = {
    'kernel': ['rbf', 'sigmoid', 'linear'],
    'C'  : [0,0.5,1,4],
    'gamma' : [None, 0.01, 0.1, 1, 3]  
}

M4 = GridSearchCV(
    SVR(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)

M4.fit(X, y)
print(M4.best_estimator_)

mean_absolute_error(y_test, M4.predict(X_test))
#Gradient Boost
params = {
    'n_estimators': [500],
    'learning_rate': [0.01, 0.03, 0.1, 1],
    'loss': ['ls'],
}

M5 = GridSearchCV(
    GradientBoostingRegressor(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
).fit(X,y)

print(M5.best_estimator_)

mean_absolute_error(y_test, M5.predict(X_test))
# XG boost
params = {
    'learning_rate': [0.003, 0.01],
    'n_estimators': [3000, 4000],
    'max_depth': [2, 3],
    'min_child_weight': [0, 1],
    'gamma': [0],
    'subsample': [0.5, 0.7],
    'colsample_bytree':[0.5, 0.7],
    'objective': ['reg:squarederror'],
    'scale_pos_weight': [1, 2],
    'reg_alpha': [0.00001, 0.001]
}

M6 = GridSearchCV(
    XGBRegressor(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
).fit(X,y)

print(M6.best_estimator_)

mean_absolute_error(y_test, M6.predict(X_test))
final_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.5, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.01, max_delta_step=0, max_depth=3,
             min_child_weight=0, monotone_constraints='()',
             n_estimators=3000, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=1e-05, reg_lambda=1, scale_pos_weight=1, subsample=0.5,
             tree_method='exact', validate_parameters=1, verbosity=None)

final_model.fit(X,y)
mean_absolute_error(y_test, final_model.predict(X_test))
preds = final_model.predict(test)
test.shape
preds
submit_file = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submit_file['SalePrice'] = preds
submit_file.to_csv('submission.csv', index=False)
submit_file
