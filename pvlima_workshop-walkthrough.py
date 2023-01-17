%matplotlib inline

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.api.types import is_numeric_dtype



sns.set()
rawtrain = pd.read_csv('../input/train.csv')

rawtest = pd.read_csv('../input/test.csv')
print('Train shape:', rawtrain.shape)

print('Test shape:', rawtest.shape)
rawtrain.dtypes.value_counts()
selected = ['GrLivArea',

 'LotArea',

 'BsmtUnfSF',

 '1stFlrSF',

 'TotalBsmtSF',

 'GarageArea',

 'BsmtFinSF1',

 'LotFrontage',

 'YearBuilt',

 'Neighborhood',

 'GarageYrBlt',

 'OpenPorchSF',

 'YearRemodAdd',

 'WoodDeckSF',

 'MoSold',

 '2ndFlrSF',

 'OverallCond',

 'Exterior1st',

 'YrSold',

 'OverallQual']
#features = [c for c in test.columns if c not in ['Id']]
train = rawtrain[selected].copy()

train['is_train'] = 1

train['SalePrice'] = rawtrain['SalePrice'].values

train['Id'] = rawtrain['Id'].values



test = rawtest[selected].copy()

test['is_train'] = 0

test['SalePrice'] = 1  #dummy value

test['Id'] = rawtest['Id'].values



full = pd.concat([train, test])



not_features = ['Id', 'SalePrice', 'is_train']

features = [c for c in train.columns if c not in not_features]
pd.Series(train.SalePrice).hist(bins=50);
pd.Series(np.log(train.SalePrice)).hist(bins=50);
full['SalePrice'] = np.log(full['SalePrice'])
def summary(df, dtype):

    data = []

    for c in df.select_dtypes([dtype]).columns:

        data.append({'name': c, 'unique': df[c].nunique(), 

                     'nulls': df[c].isnull().sum(),

                     'samples': df[c].unique()[:20] })

    return pd.DataFrame(data)
summary(full[features], np.object)
summary(full[features], np.float64)
summary(full[features], np.int64)
for c in full.select_dtypes([np.object]).columns:

    full[c].fillna('__NA__', inplace=True)

for c in full.select_dtypes([np.float64]).columns:

    full[c].fillna(0, inplace=True)
for c in full.columns:

    assert full[c].isnull().sum() == 0, f'There are still missing values in {c}'
mappers = {}

for c in full.select_dtypes([np.object]).columns:

    mappers[c] = {v:i for i,v in enumerate(full[c].unique())}

    full[c] = full[c].map(mappers[c]).astype(int)
for c in full.columns:

    assert is_numeric_dtype(full[c]), f'Non-numeric column {c}'
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn import metrics
def rmse(y_true, y_pred):

    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))
train = full[full.is_train==1][features].values

target = full[full.is_train==1].SalePrice.values

Xtrain, Xvalid, ytrain, yvalid = train_test_split(train, target, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.02, max_depth=4, random_state=42)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xvalid)

rmse(yvalid, ypred)
test = full[full.is_train==0]

ytestpred = model.predict(test[features].values)
ytestpred = np.exp(ytestpred)
subm = pd.DataFrame(ytestpred, index=test['Id'], columns=['SalePrice'])

subm.to_csv('submission.csv')
cols = full[features].select_dtypes([np.float64, np.int64]).columns

n_rows = math.ceil(len(cols)/2)

fig, ax = plt.subplots(n_rows, 2, figsize=(14, n_rows*2))

ax = ax.flatten()

for i,c in enumerate(cols):

    sns.boxplot(x=full[c], ax=ax[i])

    ax[i].set_title(c)

    ax[i].set_xlabel("")

plt.tight_layout()
limits = [('TotalBsmtSF', 4000), ('WoodDeckSF', 1400)]



full['__include'] = 1 

for c, val in limits:

    full.loc[full[c] > val, '__include'] = 0



full = full[(full.is_train==0)|(full['__include']==1)]



full = full.drop('__include', axis=1)



# these dates in the future are likely typos

full['GarageYrBlt'] = np.where(full.GarageYrBlt > 2010, full.YearBuilt, full.GarageYrBlt)
full['Age'] = 2010 - full['YearBuilt']

month_season_map = {12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}

full['SeasonSold'] = full['MoSold'].map(month_season_map).astype(int)

full['SimplOverallCond'] = full['OverallCond'].replace(

        {1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})

full['TimeSinceSold'] =  2010 - full['YrSold']

full['TotalArea1st2nd'] = full['1stFlrSF'] + full['2ndFlrSF']

full['TotalSF'] = full['TotalBsmtSF'] + full['1stFlrSF'] + full['2ndFlrSF']
train = full[full.is_train==1][features].values

target = full[full.is_train==1].SalePrice.values

Xtrain, Xvalid, ytrain, yvalid = train_test_split(train, target, test_size=0.2, random_state=42)



model = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.02, max_depth=4, random_state=42)

model.fit(Xtrain, ytrain)

ypred = model.predict(Xvalid)

rmse(yvalid, ypred)
model2 = ExtraTreesRegressor(n_estimators=1500, random_state=42)

model2.fit(Xtrain, ytrain)

ypred2 = model2.predict(Xvalid)

rmse(yvalid, ypred2)
blendpred = 0.7*ypred + 0.3*ypred2

rmse(yvalid, blendpred)
test = full[full.is_train==0]

ytestpred = model.predict(test[features].values)

ytestpred2 = model2.predict(test[features].values)

blendtestpred = 0.7*ytestpred + 0.3*ytestpred2



blendtestpred = np.exp(blendtestpred)



subm = pd.DataFrame(blendtestpred, index=test['Id'], columns=['SalePrice'])

subm.to_csv('submission_blend.csv')