import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder



sns.set_style("darkgrid")
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape, test.shape)
train.head()
train.describe(include=['number']).loc[['min','max','mean']].T.sort_values('max')
n = train.select_dtypes(include=object)

for c in n.columns:

    print('{:<14}'.format(c), train[c].unique())
plt.figure(figsize=(8,3))

sns.distplot(train['SalePrice'], fit=norm)

mu, sigma = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

plt.ylabel('Frequency')

print('skew={}'.format(skew(np.log1p(train['SalePrice']))))
cols = ['OverallQual','OverallCond','SaleType','SaleCondition']

sorted_data = train.sort_values(by='SalePrice')



fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(4 * 4, 3), sharey=True)

for i, c in zip(np.arange(len(axes)), cols):

    sns.boxplot(x=c, y='SalePrice', data=sorted_data, ax=axes[i])



fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(4 * 4, 3), sharey=True)

for i, c in zip(np.arange(len(axes)), cols):

    sns.countplot(x=c, data=sorted_data, ax=axes[i])
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 4), sharey=True)

axes = np.ravel(axes)

col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']

for i, c in zip(range(5), col_name):

    train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='red')



# delete outliers

print(train.shape)

train = train[train['GrLivArea'] < 4000]

train = train[train['LotArea'] < 100000]

print(train.shape)



for i, c in zip(range(5,10), col_name):

    train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='navy')
all_data = train.append(test, sort=False).reset_index(drop=True)

all_data.shape
n = all_data.drop('SalePrice', axis=1).loc[:,all_data.isnull().any()].isnull().sum()

print('ALL:', all_data.shape[0])

print('-' * 30)

print(n.sort_values(ascending=False))
# drop feature

all_data.drop(['MiscFeature', 'Alley', 'Fence'], axis=1, inplace=True)



# fillna with 0

cols = ['GarageArea', 'GarageCars', 'GarageFinish', 'MasVnrArea', 

        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for c in cols:

    all_data[c].fillna(0, inplace=True)



# fillna with 'None'

cols = ['BsmtQual','BsmtCond','KitchenQual','FireplaceQu','GarageType','GarageQual','GarageCond',

        'PoolQC','BsmtFinType1','BsmtFinType2','BsmtExposure','MasVnrType']

for c in cols:

    all_data[c].fillna('None', inplace=True)



# fillna with other 

all_data.loc[all_data['GarageYrBlt'].isnull(),'GarageYrBlt'] = all_data.loc[all_data['GarageYrBlt'].isnull(),'YearBuilt']



# fillna with group median

all_data['LotFrontage'] = all_data.groupby(pd.qcut(all_data['LotArea'], 10))['LotFrontage'].transform(lambda x: x.fillna(x.median()))
n = all_data.drop('SalePrice', axis=1).loc[:,all_data.isnull().any()].isnull().sum()

print(n.sort_values(ascending=False))
for i, t in all_data.loc[:, all_data.columns != 'SalePrice'].dtypes.iteritems():

    if t == object:

        all_data[i].fillna(all_data[i].mode()[0], inplace=True)

    else:

        all_data[i].fillna(all_data[i].median(), inplace=True)
all_data['_OverallQualCond'] = all_data['OverallQual'] + (all_data['OverallCond'] - 5) * 0.5

all_data['_TotalSF'] = all_data['TotalBsmtSF'] + all_data['GrLivArea']

all_data['_PorchArea'] = all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch']

all_data['_TotalArea'] = all_data['_TotalSF'] + all_data['GarageArea'] + all_data['_PorchArea']

all_data['_Rooms'] = all_data['TotRmsAbvGrd'] + all_data['FullBath'] + all_data['HalfBath']

all_data['_BathRooms'] = all_data['FullBath'] + all_data['BsmtFullBath'] + (all_data['HalfBath'] + all_data['BsmtHalfBath']) * 0.7

all_data['_GrLAreaAveByRms'] = all_data['GrLivArea'] / all_data['_Rooms']
grp = train.groupby(['YrSold','MoSold'])

piv = grp.count()['SalePrice'].reset_index()

piv.columns = ['YrSold','MoSold','SoldCount']



plt.figure(figsize=(10, 3))

sns.pointplot(x='MoSold', y='SoldCount', hue='YrSold', data=piv, join=True)

plt.legend(loc='best', bbox_to_anchor=(1.05, 0.8, 0.2, 0))



all_data['_SaleSeason'] = all_data['MoSold'].replace({1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:1, 9:0, 10:0, 11:0, 12:0})
# year feature

cols = ['YrSold','YearBuilt','YearRemodAdd','GarageYrBlt']

print(all_data[cols].describe())



# correct invalid value

all_data.loc[all_data['GarageYrBlt'] > 2010, 'GarageYrBlt'] = all_data['YearBuilt']



# relation feature

all_data['_Remod_Sold'] = 0

all_data.loc[all_data['YrSold'] <= all_data['YearRemodAdd'], '_Remod_Sold'] = 1

all_data['_Built_Sold'] = 0

all_data.loc[all_data['YrSold'] <= all_data['YearBuilt'], '_Built_Sold'] = 1



# year group

#year_map = pd.concat(pd.Series('YearBin' + str(i+1), index=range(1871+i*10,1881+i*10)) for i in range(0, 14))

#all_data['_YearBuiltGrp'] = all_data['YearBuilt'].map(year_map)



# diff feature

all_data['_YrBlt_to_sold'] = all_data['YrSold'] - all_data['YearBuilt']

all_data['_YrRemod_to_sold'] = all_data['YrSold'] - all_data['YearRemodAdd']

all_data['_GrgYrBlt_to_sold'] = all_data['YrSold'] - all_data['GarageYrBlt']

print('-' * 60)

print(all_data[['_YrBlt_to_sold','_YrRemod_to_sold','_GrgYrBlt_to_sold']].describe())

all_data.drop(cols, axis=1, inplace=True)
# to categorical feature

cols = ['MSSubClass']

for c in cols:

    all_data[c] = all_data[c].astype(str)
#log transform skewed numeric features

_='''

'''

numeric_feats = all_data.drop('SalePrice', axis=1).dtypes[all_data.dtypes != "object"].index

skewed = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_feats = skewed[skewed > 1].index

print(skewed_feats)



for c in (skewed_feats):

    all_data[c] = np.log1p(all_data[c])
mssubclass_map = {'180':1, '30':2, '45':2, '190':3, '50':3, '90':3, '85':4, '40':4, '160':4, 

                  '70':5, '20':5, '75':5, '80':5, '150':5, '120':6, '60':6}

all_data['_MSSubClassBin'] = all_data['MSSubClass'].map(mssubclass_map)



plt.figure(figsize=(9, 3))

ax = sns.boxplot(x='MSSubClass', y='SalePrice', data=all_data[all_data['SalePrice'].notnull()], order=mssubclass_map.keys())
neighborhood_map = {

        "MeadowV":0, "IDOTRR":0, "BrDale":0, 

        "Blueste":1, "NPkVill":1,

        "OldTown":2, "BrkSide":2, "Edwards":2, 

        "SWISU":3, "Sawyer":3, 

        "Mitchel":4, "NAmes":4, 

        "SawyerW":5, "NWAmes":5, 

        "Gilbert":6, "Blmngtn":6, 

        "CollgCr":7, "ClearCr":7, "Crawfor":7, 

        "Veenker":8, "Somerst":8, "Timber":8, 

        "StoneBr":9, "NoRidge":9, "NridgHt":9,

    }

all_data["_NeighborhoodBin"] = all_data["Neighborhood"].map(neighborhood_map)



plt.figure(figsize=(12, 3))

ax = sns.boxplot(x='Neighborhood', y='SalePrice', data=all_data[all_data['SalePrice'].notnull()], order=neighborhood_map.keys())

_=ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')
# encode quality - Ex(Excellent), Gd（Good）, TA（Typical/Average）, Fa（Fair）, Po（Poor）

all_data.loc[(all_data['PoolArea'] > 0) & (all_data['PoolQC'] == 'None'), 'PoolQC'] = 'TA'

cols = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']

for c in cols:

    all_data[c].replace({'None':0, 'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}, inplace=True)



# plot

fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(4 * 4, 3 * 2), sharey=True)

axes = np.ravel(axes)

cols = ['BsmtExposure','CentralAir','GarageFinish','Utilities','LandSlope','Functional','LotShape','SaleCondition']

for i, c in zip(np.arange(len(axes)), cols):

    sns.boxplot(x=c, y='SalePrice', data=train.sort_values(by='SalePrice'), ax=axes[i])



# encode remaining columns

all_data['BsmtExposure'].replace({'Gd':4,'Av':3,'Mn':2,'No':1,'None':0}, inplace=True)

all_data['CentralAir'].replace({'Y':1,'N':0}, inplace=True)

all_data['GarageFinish'].replace({'Fin':3,'RFn':2,'Unf':1,'None':0}, inplace=True)

all_data['Utilities'].replace({'AllPub':3,'NoSewr':2,'NoSeWa':1,'ELO':0}, inplace=True)

all_data['LandSlope'].replace({'Gtl':2,'Mod':1,'Sev':0}, inplace=True)

all_data['Functional'].replace({'Typ':7,'Min1':6,'Min2':5,'Mod':4,'Maj1':3,'Maj2':2,'Sev':1,'Sal':0}, inplace=True)

all_data['LotShape'].replace({'Reg':3,'IR1':2,'IR2':1,'IR3':0}, inplace=True)



# encode to another 

all_data['_PriceCut'] = all_data['SaleCondition'].replace(

    {'AdjLand':1,'Abnorml':1,'Family':1,'Alloca':1,'Normal':0,'Partial':0})
# Condition1&2, Exterior1st&2nd --> marged dummies

def pair_features_to_dummies(df, col1, col2, prefix):

    d_1 = pd.get_dummies(df[col1].astype(str), prefix=prefix)

    d_2 = pd.get_dummies(df[col2].astype(str), prefix=prefix)

    for c in list(set(list(d_1.columns) + list(d_2.columns))):

        if not c in d_1.columns: d_1[c] = 0

        if not c in d_2.columns: d_2[c] = 0

    return (d_1 + d_2).clip(0, 1)



cond = pair_features_to_dummies(all_data,'Condition1','Condition2','Condition')

exterior = pair_features_to_dummies(all_data,'Exterior1st','Exterior2nd','Exterior')



all_data = pd.concat([all_data, cond, exterior], axis=1)

all_data.drop(['Condition1','Condition2','Exterior1st','Exterior2nd'], axis=1, inplace=True)

all_data.loc[:,cond.columns[0]:].head()
# Create new polynomial features about OverallQual

all_data['_TotalSF_OverallQual'] = all_data['_TotalSF'] * all_data['OverallQual']

all_data['_Neighborhood_OverallQual'] = all_data['_NeighborhoodBin'] + all_data['OverallQual']

all_data['_Functional_OverallQual'] = all_data['Functional'] + all_data['OverallQual']
for i, t in all_data.loc[:, all_data.columns != 'SalePrice'].dtypes.iteritems():

    if t == object:

        all_data = pd.concat([all_data, pd.get_dummies(all_data[i].astype(str), prefix=i)], axis=1)

        all_data.drop(i, axis=1, inplace=True)
#sns.boxplot(x='GarageCars', y='SalePrice', data=all_data[all_data['SalePrice'].notnull()])

all_data['GarageCars'] = all_data['GarageCars'].clip(0,3)



# They are either not very helpful or they cause overfitting.

all_data.drop('MSZoning_C (all)', axis=1, inplace=True)



# adverse effect

#drop_cols = ['MiscVal', 'MoSold', 'ExterCond', 'BsmtFinSF2', 'BedroomAbvGr']

drop_cols = ['MiscVal', 'MoSold']

all_data.drop(drop_cols, axis=1, inplace=True)



#all_data.loc[(all_data['BsmtFinType2']=='None') & (all_data['BsmtFinSF2']>0), 'BsmtFinType2'] = 'ALQ'

#all_data.drop('BsmtFinType2', axis=1, inplace=True)
train = all_data[all_data['SalePrice'].notnull()]

test = all_data[all_data['SalePrice'].isnull()].drop('SalePrice', axis=1)
X_train = train.drop(['SalePrice','Id'], axis=1)

Y_train = train['SalePrice']

X_test  = test.drop(['Id'], axis=1)



print(X_train.shape, Y_train.shape, X_test.shape)
from sklearn import ensemble, metrics

from sklearn import linear_model, preprocessing

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.model_selection import ShuffleSplit

from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb
#scaler = preprocessing.RobustScaler();

scaler = preprocessing.StandardScaler();

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
lasso = linear_model.Lasso(alpha=0.001, max_iter=5000, random_state=42)

lasso.fit(X_train_scaled, np.log1p(Y_train))

fi = pd.DataFrame({"Feature Importance":lasso.coef_}, index=X_train.columns)

fi = fi[fi["Feature Importance"] != 0].sort_values("Feature Importance")



fi.plot(kind="bar",figsize=(18,4))

plt.xticks(rotation=-90)

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models, weight):

        self.models = models

        self.weight = weight

        

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:

            model.fit(X, y)

        return self

    

    def predict(self, X):

        predictions = np.column_stack([(model.predict(X) * weight) for model, weight in zip(self.models_, self.weight)])

        return np.sum(predictions, axis=1)
KRR = KernelRidge(alpha=0.05, kernel='polynomial', degree=1, coef0=2.5)

lasso = linear_model.Lasso(alpha=0.001, max_iter=5000, random_state=42)

GBoost = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, 

                                            max_features='sqrt', loss='huber', random_state=42)



reg = AveragingModels(models=(KRR, lasso, GBoost), weight=[0.25, 0.35, 0.40])
def rmse_cv(model, x, y):

    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=5))

    return rmse



score = rmse_cv(reg, X_train_scaled, np.log1p(Y_train))

print(round(score.mean(), 5))
reg.fit(X_train_scaled, np.log1p(Y_train))

result = np.expm1(reg.predict(X_test_scaled))
submission = pd.DataFrame({

    "Id": test["Id"],

    "SalePrice": result

})

submission.to_csv("submission.csv", index=False)
submission.head(10)