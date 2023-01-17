# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_FOLDER = '/kaggle/input/house-prices-advanced-regression-techniques/'

test = pd.read_csv(os.path.join(DATA_FOLDER, 'test.csv'))

train = pd.read_csv(os.path.join(DATA_FOLDER, 'train.csv'))

sub = pd.read_csv(os.path.join(DATA_FOLDER, 'sample_submission.csv'))
test.head()
np.shape(test)
test.describe()
test.nunique()
train.head()
np.shape(train)
train.info()
plt.subplots(figsize=(9,6))

sns.distplot(train['SalePrice'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(train['SalePrice'])

fig = plt.figure()

stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])



#Check again for more normal distribution



plt.subplots(figsize=(9,6))

sns.distplot(train['SalePrice'], fit=stats.norm)



# Get the fitted parameters used by the function



(mu, sigma) = stats.norm.fit(train['SalePrice'])



# plot with the distribution

train_corr = train.select_dtypes(include=[np.number])

del train_corr['Id']

corr = train_corr.corr()

plt.subplots(figsize=(20,9))

sns.heatmap(corr, annot=True)
top_feature = corr.index[abs(corr['SalePrice']>0.5)]

plt.subplots(figsize=(9,6))

top_corr = train[top_feature].corr()

sns.heatmap(top_corr, annot=True)

plt.show()
plt.figure(figsize=(9, 6))

sns.boxplot(x=train.OverallQual, y=train.SalePrice)
plt.scatter(x ='TotalBsmtSF', y = 'SalePrice', data = train)

plt.xlabel('Total Basement in Square Feet')
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

features = pd.concat((train, test), sort=False).reset_index(drop=True)

features.drop(['SalePrice'], axis=1, inplace=True)

features.head()
features_na = (features.isnull().sum() / len(features)) * 100

features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)[:20]

missing_data = pd.DataFrame({'Missing Ratio' :features_na})

missing_data
for col in features.columns:

    length = len(features)

    null_sum = features[col].isnull().sum()

    if null_sum / length >= 0.80:

        features[col] = features[col].fillna('None')
features["FireplaceQu"] = features["FireplaceQu"].fillna("None")
features["LotFrontage"] = features.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    features[col] = features[col].fillna('None')

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    features[col] = features[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')

features["MasVnrType"] = features["MasVnrType"].fillna("None")

features["MasVnrArea"] = features["MasVnrArea"].fillna(0)
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

features = features.drop(['Utilities'], axis=1)

features["Functional"] = features["Functional"].fillna("Typ")

features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

features['MSSubClass'] = features['MSSubClass'].fillna("None")
features_na = (features.isnull().sum() / len(features)) * 100

features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)[:20]
features['MSSubClass'] = features['MSSubClass'].apply(str)



features['OverallCond'] = features['OverallCond'].astype(str)



features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
categorical_cols = features.select_dtypes(include=['object']).columns

for col in categorical_cols:

    dum = pd.get_dummies(features[col],prefix=col, drop_first=True)

    features.drop(col,axis=1,inplace=True)

    features = pd.concat([features,dum],axis=1)

print(features.shape)
train = features[:ntrain]

test = features[ntrain:]
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return rmse
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)

print("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from vecstack import stacking

from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Lasso

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



models = [lasso, GBoost] 

S_train, S_test = stacking(models,                   

                           train, y_train, test,   

                           regression=True, 

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=mean_squared_log_error,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,  

                           random_state=0,

                           verbose=2)

model_lasso = lasso.fit(S_train, y_train)

sm_pred = np.expm1(model_lasso.predict(S_test))

submission = pd.DataFrame(data = sub['Id'], columns= ['Id'])

submission['SalePrice'] = sm_pred
submission.to_csv('res.csv', index=False)
submission.head()