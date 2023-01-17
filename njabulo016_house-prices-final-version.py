# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install vecstack
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
desc = open('../input/data_description.txt', "r") 

print(desc.read())
plt.figure(figsize=(10,8))



#saleprice correlation matrix

cols = train.corr().nlargest(10, 'SalePrice')['SalePrice'].index

corr_mat = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)



#plot corr matrix

sns.heatmap(corr_mat, annot=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, center=0.25)

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)

plt.title('OverallQual vs SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.title('GrLivArea vs SalePrice')

plt.show()
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.title('GrLivArea vs SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GarageCars'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageCars', fontsize=13)

plt.title('GarageCars vs SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['GarageArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)

plt.title('GarageArea vs SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.title('TotalBsmtSF vs SalePrice')

plt.show()
fig, ax = plt.subplots()

ax.scatter(x = train['1stFlrSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('1stFlrSF', fontsize=13)

plt.title('1stFlrSF vs SalePrice')

plt.show()
from scipy import stats

from scipy.stats import norm, skew #for some statistics
sns.distplot(train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice'] , fit=norm);



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
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
f, ax = plt.subplots(figsize=(10, 8))

plt.xticks(rotation='90')

sns.barplot(x=features_na.index, y=features_na, color='red')

plt.xlabel('Features')

plt.ylabel('Percent of missing values')

plt.title('Percent missing data by feature')


print(features['PoolQC'].unique())

print(features['MiscFeature'].unique())

print(features['Alley'].unique())

print(features['Fence'].unique())
for col in features.columns:

    length = len(features)

    null_sum = features[col].isnull().sum()

    if null_sum / length >= 0.80:

        features[col] = features[col].fillna('None')
print(features['PoolQC'].unique())

print(features['MiscFeature'].unique())

print(features['Alley'].unique())

print(features['Fence'].unique())
features["FireplaceQu"] = features["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

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
features['MSZoning'].value_counts()
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])
features['Utilities'].value_counts()
features = features.drop(['Utilities'], axis=1)
features["Functional"] = features["Functional"].fillna("Typ")
features['Electrical'].value_counts()
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])
features['KitchenQual'].value_counts()
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])
print(features['Exterior1st'].value_counts())

print(features['Exterior2nd'].value_counts())
features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
features['SaleType'].value_counts()
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
features['MSSubClass'] = features['MSSubClass'].fillna("None")
features_na = (features.isnull().sum() / len(features)) * 100

features_na = features_na.drop(features_na[features_na == 0].index).sort_values(ascending=False)[:20]

missing_data = pd.DataFrame({'Missing Ratio' :features_na})

missing_data
#MSSubClass=The building class

features['MSSubClass'] = features['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

features['OverallCond'] = features['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
# Adding total sqfootage feature 

features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
# get dummies all categorical(object) columns



categorical_cols = features.select_dtypes(include=['object']).columns

for col in categorical_cols:

    dum = pd.get_dummies(features[col],prefix=col, drop_first=True)

    features.drop(col,axis=1,inplace=True)

    features = pd.concat([features,dum],axis=1)

print(features.shape)
train = features[:ntrain]

test = features[ntrain:]
train.head()
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return rmse
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(train, y_train)
# Reverse the log1p transformation

reg_pred = np.expm1(reg.predict(test))

reg_pred
# To see what your score is

score = rmsle_cv(reg)

print("\nRegression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.tree import DecisionTreeRegressor
tree = make_pipeline(RobustScaler(), DecisionTreeRegressor())
tree.fit(train, y_train)
# Reverse the log transformation

tree_pred = np.expm1(tree.predict(test))

tree_pred
# To see what your score is

score = rmsle_cv(tree)

print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
forest = make_pipeline(RobustScaler(), RandomForestRegressor(min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0, n_estimators=100, bootstrap=True))
forest.fit(train, y_train)
# To see what your score is

score = rmsle_cv(forest)

print("\nRandom Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.linear_model import Lasso
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
lasso.fit(train, y_train)
# Reverse the log transformation

lasso_pred = np.expm1(lasso.predict(test))

lasso_pred
score = rmsle_cv(lasso)

print("\nLasso(scaled) score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.linear_model import ElasticNet
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0))
ENet.fit(train, y_train)

enet_pred = np.expm1(ENet.predict(test))

enet_pred
score = rmsle_cv(ENet)

print("\nENet(scaled) score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=3,min_samples_split=10, 

                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)

print("\nGBoost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from vecstack import stacking
from sklearn.metrics import mean_squared_log_error
models = [lasso, ENet, GBoost] 

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

sm_pred
sample = pd.read_csv('../input/sample_submission.csv')

sample.head()
submission = pd.DataFrame(data = sample['Id'], columns= ['Id'])

submission['SalePrice'] = sm_pred
submission.to_csv('submit_2.csv', index=False)
submission.head()