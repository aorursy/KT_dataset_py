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
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
train_df = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sns.set_style('whitegrid')
sns.jointplot(train_df['LotFrontage'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['LotFrontage']>200) & (train_df['SalePrice']<300000)].index).reset_index(drop=True)
train_df = train_df.drop(train_df[(train_df['LotFrontage']<200) & (train_df['SalePrice']>600000)].index).reset_index(drop=True)
sns.jointplot(train_df['LotFrontage'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['LotArea'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['LotArea']>50000) & (train_df['SalePrice']<0.6)].index).reset_index(drop=True)
sns.jointplot(train_df['LotFrontage'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['MasVnrArea'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['MasVnrArea']>1500) & (train_df['SalePrice']<300000)].index).reset_index(drop=True)
sns.jointplot(train_df['MasVnrArea'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['BsmtFinSF1'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['BsmtUnfSF'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['BsmtUnfSF']>2000) & (train_df['SalePrice']<500000)].index).reset_index(drop=True)
sns.jointplot(train_df['BsmtUnfSF'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['TotalBsmtSF'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['TotalBsmtSF']>3000) & (train_df['SalePrice']<600000)].index).reset_index(drop=True)
sns.jointplot(train_df['TotalBsmtSF'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['1stFlrSF'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['2ndFlrSF'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['GrLivArea'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<200000)].index).reset_index(drop=True)
sns.jointplot(train_df['GrLivArea'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['GarageArea'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['GarageArea']>=1200) & (train_df['SalePrice']<300000)].index).reset_index(drop=True)
sns.jointplot(train_df['GarageArea'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['WoodDeckSF'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['WoodDeckSF']>=600) & (train_df['SalePrice']<400000)].index).reset_index(drop=True)
sns.jointplot(train_df['WoodDeckSF'], train_df['SalePrice'], kind = 'reg')
sns.jointplot(train_df['OpenPorchSF'], train_df['SalePrice'], kind = 'reg')
train_df = train_df.drop(train_df[(train_df['OpenPorchSF']>=400) & (train_df['SalePrice']<400000)].index).reset_index(drop=True)
sns.jointplot(train_df['OpenPorchSF'], train_df['SalePrice'], kind = 'reg')
print(train_df.shape)
print(test.shape)
ntrain = train_df.shape[0]
ntest = test.shape[0]
y_train = train_df['SalePrice']
df = pd.concat([train_df,test])
df.drop('SalePrice', axis=1, inplace=True)
df.head()
df.info()
plt.figure(figsize=(21,8))
sns.set_style('whitegrid')
sns.heatmap(df.isnull(), cmap = 'viridis', yticklabels = False, cbar = False)
feature_with_na_values = [feature for feature in df.columns if df[feature].isnull().sum()>1]
for feature in feature_with_na_values:
    print(feature, np.round(df[feature].isnull().mean(),4), '% of missing value')
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
print(df.shape)
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
len(categorical_features)
numerical_features = [feature for feature in df.columns if feature not in categorical_features]
len(numerical_features)
categorical_feature_with_na_values = [feature for feature in categorical_features if df[feature].isnull().sum()>1]
len(categorical_feature_with_na_values)
numerical_feature_with_na_values = [feature for feature in numerical_features if df[feature].isnull().sum()>1]
len(numerical_feature_with_na_values)
for col in categorical_feature_with_na_values:
    print(df[col].value_counts())
df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df[col] = df[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df[col] = df[col].fillna('None')
df = df.drop(['Utilities'], axis=1)
for col in ('MSZoning', 'MasVnrType','Functional'):
    df[col] = df[col].fillna(df[col].mode()[0])
print(numerical_feature_with_na_values)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.median()))
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df[col] = df[col].fillna(0)
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(df[col].mode()[0])
plt.figure(figsize=(20,8))
sns.set_style('whitegrid')
sns.heatmap(df.isnull(), cmap = 'viridis', yticklabels = False, cbar = False)
year_features = [feature for feature in df.columns if 'Yr' in feature or 'Year' in feature]
print(year_features)
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    df[feature] = (df['YrSold'].astype(int) - df[feature]).astype('int')
df[year_features].head()
for feature in numerical_features:
    print(feature, df[feature].nunique())
print(df['MSSubClass'].nunique())
print(df['OverallCond'].nunique())
print(df['MoSold'].nunique())
print(df['YrSold'].nunique())
#MSSubClass=The building class
df['MSSubClass'] = df['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
df['OverallCond'] =df['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df.drop(['2ndFlrSF','TotalBsmtSF','1stFlrSF'],axis=1,inplace=True)
df.shape
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))

# shape        
print('Shape all_data: {}'.format(df.shape))
from scipy.stats import norm, skew
numeric_feat = [feature for feature in df.columns if df[feature].dtype != 'O']
numeric_feat.remove('Id')
skewed_feat = df[numeric_feat].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame({'Skew' :skewed_feat})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
lamda = 0.15
for feat in skewed_features:
    df[feat] = boxcox1p(df[feat],lamda)
df = pd.get_dummies(df)
print(df.shape)
df.sort_values(by = 'Id')
train_df = df[:ntrain]
test_df = df[ntrain:]
print('Shape of Train Data :' + str(train_df.shape))
print('Shape of Test Data :' + str(test_df.shape))
train_df['Street'].value_counts()
train_df['LandSlope'].value_counts()
X = train_df.drop(['Id','Street','LandSlope'],axis=1)
y = y_train 
X.shape
y.shape
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=101)
X_train.shape
xg = xgboost.XGBRegressor()
param_grid = dict(learning_rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], max_depth = [3,4,5,6,8,10,12,15],
                 min_child_weight = [1,3,5,7], gamma = [0.0,0.1,0.2,0.3,0.4], colsample_bytree = [0.3,0.4,0.5,0.6])
grid_xg = RandomizedSearchCV(xg, param_grid, cv=10, scoring = 'r2')
grid_xg.fit(X_train,y_train)
print(grid_xg.best_score_)
print(grid_xg.best_params_)
y_pred_train = grid_xg.predict(X_train)
print(r2_score(y_train, y_pred_train))
print(mean_squared_error(y_train, y_pred_train))
y_pred_test = grid_xg.predict(X_test)
print(r2_score(y_test, y_pred_test))
print(mean_squared_error(y_test, y_pred_test))
y_pred_xg = grid_xg.predict(test_df.drop(['Id','Street','LandSlope'],axis=1))
y_pred = pd.DataFrame(y_pred_xg, columns=["SalePrice"])
sub = pd.concat([test_df['Id'].astype(int), y_pred], axis=1)
sub.to_csv("submission3.csv", index=False)