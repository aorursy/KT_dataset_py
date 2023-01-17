import numpy as np 

import pandas as pd

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor

from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import seaborn as sns

import matplotlib.pyplot as plt

import os





%matplotlib inline

SEED = 42
INPUT_DIR='../input'

for dirname, _, filenames in os.walk(INPUT_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv(INPUT_DIR+'/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv(INPUT_DIR+'/house-prices-advanced-regression-techniques/test.csv')
train.head()
Y_train = train['SalePrice'].reset_index(drop=True)

train=train.drop(['Id'], axis=1)

test_ids = test['Id'].reset_index(drop=True)

test=test.drop(['Id'], axis=1)

print(len(train),len(test))
df = pd.concat([train, test],sort=False)

df = df.reset_index(drop=True)
df.info()
fig = plt.figure(figsize=(20,20))



# Columns with low correlation with other variables, will be dropped to get matrix

columns = ['PoolArea', 'MiscVal', 'BsmtHalfBath', 'MoSold', 'YrSold']

train_corr = train.drop(columns=columns).corr().round(2)



sns.heatmap(train_corr, 

            annot=True, 

            center = 0,

            cmap=sns.diverging_palette(20, 220, n=200));
fig, ax = plt.subplots(ncols=2,figsize=(16,6))

ax = ax.ravel()

sns.scatterplot(x='OverallQual',y='SalePrice', data=train, ax=ax[0]);

sns.boxplot(x='OverallQual',y='SalePrice', data=train,ax=ax[1]);
fig, ax = plt.subplots(ncols=1,figsize=(16,6))

ax = ax.ravel()

sns.distplot(train.GrLivArea, ax=ax[0]);
fig, ax = plt.subplots(ncols=2,figsize=(12,6))

ax = ax.ravel()

sns.distplot(train['1stFlrSF'],hist=False, label='1stFlrSF', ax=ax[0]);

sns.distplot(train['2ndFlrSF'],hist=False, label='2ndFlrSF', ax=ax[1]);
firstandsecondSF = train['1stFlrSF'] + train['2ndFlrSF']

train.GrLivArea.corr(firstandsecondSF)
sns.distplot(train['GrLivArea'],hist=False,label='GrLivArea');

sns.distplot(firstandsecondSF,hist=False, label='1stFlrSF + 2ndFlrSF');
totalSF = train['1stFlrSF'] + train['2ndFlrSF'] + train['LowQualFinSF']

train.GrLivArea.corr(totalSF)
sns.distplot(train['GrLivArea'], hist=False, label='GrLivArea');

sns.distplot(totalSF, hist=False, label='1stFlrSF + 2ndFlrSF + LowQualFinSF');
def print_missing(df):

    for col in df.columns.tolist():

        if df[col].isnull().sum():

             print('{}: {}'.format(col, df[col].isnull().sum()))
print_missing(df)
df.loc[df['Alley'].isnull(),'Alley'] = 'None'

df.loc[df['Fence'].isnull(),'Fence'] = 'None'

df.loc[df['MiscFeature'].isnull(),'MiscFeature'] = 'None'
df.loc[np.logical_and(df['PoolArea'] != 0, df['PoolQC'].isnull()), 

        ['PoolArea', 'PoolQC', 'OverallQual']]
df.loc[2420,['PoolQC']] = 2

df.loc[2599,['PoolQC']] = 2

df.loc[2503,['PoolQC']] = 3



df['PoolQC'].fillna('None', inplace=True)
df['MasVnrType'].value_counts()
df.loc[df['MasVnrArea'].isnull() | df['MasVnrType'].isnull()][['MasVnrArea','MasVnrType']]
df.loc[df['MasVnrArea'].isnull(),['MasVnrArea','MasVnrType']] = [0, 'None']

df.loc[df['MasVnrType'].isnull(),['MasVnrType']] = 'None'
print(len(df.loc[df['BsmtQual'].isnull() & df['BsmtCond'].isnull() & df['BsmtExposure'].isnull() &

             df['BsmtFinType1'].isnull() & df['BsmtFinType2'].isnull()]))
df.loc[(df['BsmtFinType1'].notnull()) & (df['BsmtQual'].isnull() | df['BsmtCond'].isnull() | df['BsmtExposure'].isnull() | df['BsmtFinType2'].isnull())][['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2',\

        'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']]
df.BsmtQual.loc[[2217, 2218]] = df.BsmtQual.mode()[0]

df.BsmtCond.loc[[2040, 2185, 2524]]= df.BsmtCond.mode()[0]

df.BsmtFinType2.loc[[332]] = df.BsmtFinType2.mode()[0]

df.BsmtExposure.loc[[948, 1487, 2348]] = df.BsmtExposure.mode()[0]



df.BsmtQual.fillna('None', inplace=True) 

df.BsmtCond.fillna('None', inplace=True)

df.BsmtExposure.fillna('None', inplace=True)

df.BsmtFinType1.fillna('None', inplace=True)

df.BsmtFinType2.fillna('None', inplace=True)

df.BsmtFinSF1.fillna(0, inplace=True)

df.BsmtFinSF2.fillna(0, inplace=True)

df.BsmtUnfSF.fillna(0, inplace=True)

df.TotalBsmtSF.fillna(0, inplace=True)

df.BsmtFullBath.fillna(0, inplace=True)

df.BsmtHalfBath.fillna(0, inplace=True)
len(df.loc[df.GarageType.isnull() & df.GarageFinish.isnull() & df.GarageQual.isnull() & df.GarageCond.isnull(),

       ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']])
df.loc[df.GarageType.notnull() & (df.GarageFinish.isnull() | df.GarageQual.isnull() | df.GarageCond.isnull()),

       ['GarageType','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']]
df.loc[[2127, 2576], ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageCars', 'GarageArea']] = [df['GarageCond'].mode()[0], df['GarageQual'].mode()[0], df['GarageFinish'].mode()[0], df['GarageCars'].mode()[0], df['GarageArea'].mean()]
indexes = df.loc[df.GarageYrBlt.notnull()].index

df.GarageYrBlt.iloc[indexes].corr(df.YearBuilt.iloc[indexes])
print(np.sort(df.GarageYrBlt.unique())[::-1])

print(np.sort(df.YearBuilt.unique())[::-1])
df.GarageYrBlt.replace(2207, 2007, inplace=True)

emp_garageYr = df.loc[df.GarageYrBlt.isnull()].index

df.GarageYrBlt.iloc[emp_garageYr] = df.YearBuilt.iloc[emp_garageYr]
indexes = df.loc[df.GarageYrBlt.notnull()].index

df.GarageYrBlt.iloc[indexes].corr(df.YearBuilt.iloc[indexes])
df.loc[df['GarageType'].isnull(), ['GarageYrBlt']] = 0

df['GarageType'].fillna('None', inplace=True)

df['GarageFinish'].fillna('None',inplace=True)

df['GarageCars'].fillna(0,inplace=True)

df['GarageArea'].fillna(0,inplace=True)

df['GarageQual'].fillna('None',inplace=True)

df['GarageCond'].fillna('None',inplace=True)
df['FireplaceQu'].fillna('None', inplace=True)
for feature in ['Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'MSZoning', 'Functional', 'SaleType', 'Utilities']:

    df[feature] = df.groupby(['Neighborhood', 'MSSubClass'])[feature].apply(lambda x: x.fillna(x.mode()[0]))



df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x : x.fillna(x.median()))
print_missing(df)
print('Training Set SalePrice Skew: {}'.format(train['SalePrice'].skew()))

print('Training Set SalePrice Kurtosis: {}'.format(train['SalePrice'].kurt()))

print('Training Set SalePrice Mean: {}'.format(train['SalePrice'].mean()))

print('Training Set SalePrice Median: {}'.format(train['SalePrice'].median()))

print('Training Set SalePrice Max: {}'.format(train['SalePrice'].max()))

fig, axs = plt.subplots(figsize=(8, 5))

g = sns.distplot(train.SalePrice, hist=True)

g=g.legend(["Skewness: {:.4}".format(Y_train.skew())])
train_corr = df[:len(train)].corr().abs().unstack().reset_index().sort_values(by=[0], ascending=False)

train_corr.rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: 'Correlation'}, inplace=True)

train_corr.drop(train_corr[train_corr['Correlation'] == 1.0].index, inplace=True)
train_corr[train_corr['Feature_1'] == 'SalePrice']
train_corr[1::2].head(15)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

df['TotalBath'] = df['FullBath']+df['BsmtFullBath']+(df['BsmtHalfBath']+df['HalfBath'])*0.5

df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch']
df['YearBuiltRemod'] = df['YearBuilt'] + df['YearRemodAdd']

print(f'Correaltion with Target variable: {df.SalePrice.corr(df.YearBuiltRemod).round(3)}')
sns.jointplot(x="YearBuiltRemod", y="SalePrice", data=df, kind="reg");
df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

df['isNewer'] = 0

df.loc[df['YrSold'] == df['YearBuilt'], 'isNewer'] = 1
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,12))

ax=ax.ravel()

for i,f in enumerate(['HasPool', 'HasFireplace', 'isNewer']):

    sns.barplot(x=f, y='SalePrice', data=df, ax=ax[i])
df_ = df.copy()

for f in ['MSSubClass', 'MoSold', 'YrSold', 'OverallQual', 'OverallCond']:

    df_[f] = df_[f].astype('category')

num_features = df_.drop(columns=['SalePrice']).select_dtypes(include=np.number).columns.tolist()

cat_features = df_.select_dtypes(exclude=np.number).columns.tolist()
print(len(num_features), len(cat_features))

df_num = df_[num_features]

df_cat = df_[cat_features]
fig = plt.figure(figsize=(18,24))

ax = fig.gca()

df_num.hist(ax=ax);
skew_df = df_num.skew().abs().sort_values(ascending=False).reset_index()

skew_df.rename(columns={"level_0": "Feature_1",0: 'Skewness'}, inplace=True)

skew_df[skew_df.Skewness >= 0.5]
from scipy.stats import boxcox_normmax

from scipy.special import boxcox1p



skew_features = [

    'LotFrontage', 'MasVnrArea', 'BsmtFinSF1',

    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '2ndFlrSF', 

    'LowQualFinSF', 'GrLivArea', 'WoodDeckSF', 'OpenPorchSF',

    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 

    'TotalSF', 'TotalPorchSF',

]



for feature in skew_features:

    df[feature] = boxcox1p(df[feature], boxcox_normmax(df[feature] + 1))
exterQual = {'Fa': 1, 'TA': 2, 'Gd': 3,'Ex': 4}

exterCond = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

bsmtQual = {'None': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

bsmtCond = {'None': 0, 'Po': 1, 'Fa': 2,'TA': 3, 'Gd': 4}

bsmtExposure = {'None': 0, 'No': 1, 'Mn': 2,'Av': 3, 'Gd': 4}

bsmtFinType1 = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

bsmtFinType2 = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

heatingQC = {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}

kitchenQual = {'Fa': 0, 'TA': 1, 'Gd': 2, 'Ex': 3}

firePlaceQu = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

garageQual = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

garageCond = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

functional = {'Typ': 0, 'Min1': 1, 'Min2': 1, 'Mod': 2, 'Maj1': 3, 'Maj2': 3, 'Sev': 4}

landSlope = { 'Sev': 1, 'Mod': 2, 'Gtl': 3 }



df['ExterQual'] = df['ExterQual'].map(exterQual)

df['ExterCond'] = df['ExterCond'].map(exterCond)

df['BsmtQual'] = df['BsmtQual'].map(bsmtQual)

df['BsmtCond'] = df['BsmtCond'].map(bsmtCond)

df['BsmtExposure'] = df['BsmtExposure'].map(bsmtExposure)

df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmtFinType1)

df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmtFinType2)

df['HeatingQC'] = df['HeatingQC'].map(heatingQC)

df['KitchenQual'] = df['KitchenQual'].map(kitchenQual)

df['FireplaceQu'] = df['FireplaceQu'].map(firePlaceQu)

df['GarageQual'] = df['GarageQual'].map(garageQual)

df['GarageCond'] = df['GarageCond'].map(garageCond)

df['Functional'] = df['Functional'].map(functional)

df['LandSlope'] = df['LandSlope'].map(landSlope)
df.drop(columns=['Street', 'Utilities', 'Fireplaces', 

                'PoolArea', 'PoolQC', 'GarageYrBlt',

                'GarageArea', 'TotalBsmtSF', '1stFlrSF',

                'FullBath', 'YearBuilt', 'YearRemodAdd',

                ], inplace=True)
df = pd.get_dummies(df,columns=df.select_dtypes(exclude=np.number).columns.tolist())

df.shape
fig = plt.figure(figsize=(12, 6))



sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=df)



plt.xlabel('GrLivArea', size=15)

plt.ylabel('SalePrice', size=15)

plt.tick_params(axis='x', labelsize=12)

plt.tick_params(axis='y', labelsize=12) 

    

plt.title('GrLivArea & OverallQual vs SalePrice', size=15, y=1.05)



plt.show()
fig = plt.figure(figsize=(12, 6))



sns.scatterplot(x='OverallQual', y='SalePrice', data=df);
print(train[np.logical_and(train['OverallQual'] < 5, train['SalePrice'] > 200000)].index)

print(train[np.logical_and(train['GrLivArea'] > 4000, train['SalePrice'] < 300000)].index)

print(df[np.logical_and(df['OverallQual']==4,  df['SalePrice'] > 200000)].index)

# train[train['BsmtFinSF1'] > 4000]

outliers=[457, 523, 1298]
X_train = df[:len(train)].drop(columns=['SalePrice'])

X_test = df[len(train):].drop(columns=['SalePrice'])

print(X_train.shape, Y_train.shape, X_test.shape)
sparse=[]

for feature in X_test.columns:

    counts = X_test[feature].value_counts()

    zeros = counts.iloc[0]

    if zeros/len(X_test) > 99.94:

        sparse.append(feature)

        print(feature)



X_test.drop(columns=sparse, inplace=True)

X_train.drop(columns=sparse, inplace=True)
X_train = X_train.drop(X_train.index[outliers])

Y_train = Y_train.drop(Y_train.index[outliers])

y_train = Y_train.apply(np.log1p)
scaler = RobustScaler()



x_train = pd.DataFrame(scaler.fit_transform(X_train),

                      index=X_train.index,

                      columns=X_train.columns)



X_test = pd.DataFrame(scaler.transform(X_test),

                      index=X_test.index,

                      columns=X_test.columns)
kfolds = KFold(n_splits=8, shuffle=True, random_state=42)



def cv_rmse(model, X=x_train, y=y_train):

    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=kfolds))

    return (rmse)



# rmsle scoring function

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def fit_model(model, X=x_train, y=y_train):

    return model.fit(X, y)
def grid_search(model, param_grid, cv=5, 

                scoring='neg_mean_squared_error'):

    

    grid_model = GridSearchCV(model, 

                          param_grid, 

                          cv=cv, 

                          scoring=scoring, 

                          return_train_score=True 

                         )

    fit_model(grid_model)

    cvres = grid_model.cv_results_

    for mean_score, std_score, params in zip(cvres["mean_test_score"], cvres["std_test_score"], cvres["params"]):

        print(np.sqrt(-mean_score), std_score, params)

    return grid_model
param_grid=[{'alpha': np.arange(13,14,0.1)}]

ridge = Ridge(random_state=SEED)

grid_ridge = grid_search(ridge, param_grid, cv=kfolds)
param_grid=[{'alpha': [0.0005, 0.0007]}]

lasso = Lasso(max_iter=1e7, random_state=SEED)

grid_lasso = grid_search(lasso, param_grid, cv=kfolds)
param_grid=[

    {

        'alpha': [ 0.008],

        'l1_ratio': [0.0056, 0.009, 0.01, 0.05],

    }

]

elastic = ElasticNet(max_iter=1e7, random_state=SEED)

grid_elastic = grid_search(elastic, param_grid, cv=kfolds)
param_grid=[

    {

        'C': np.logspace(1.5,2,4),

        'epsilon': [0.05, 0.06],

        'gamma':[0.0001]

    }

]

svr= SVR()

grid_svr = grid_search(svr, param_grid, cv=kfolds)
param_grid=[

    {

        'learning_rate': [0.01],

        'max_depth':[ 4 ],

    }

]

gbr = GradientBoostingRegressor(

    n_estimators=3000, max_features='sqrt', 

    min_samples_leaf=15, min_samples_split=10,

    loss='huber', random_state=SEED)

grid_gbr = grid_search(gbr, param_grid, cv=kfolds)
param_grid=[

    {

        'learning_rate': [0.005, 0.009, 0.01],

        'n_estimators': [5000]

    }

]

lightgbm = LGBMRegressor(objective='regression',

                        num_leaves=4,max_bin=200,

                        bagging_fraction=0.75,

                        bagging_freq=5,bagging_seed=7,

                        feature_fraction=0.2,

                        feature_fraction_seed=7,

                        random_state=SEED,verbose=-1

                        )

grid_lgb = grid_search(lightgbm, param_grid, cv=kfolds)
param_grid=[

    {

        'learning_rate': [0.1],

        'n_estimators': [3460],

        'max_depth':[ 3 ],

        'gamma':[0.001],

        'subsample':[0.7]

    }

]

xgb = XGBRegressor(n_jobs=3)

grid_xgb = grid_search(xgb, param_grid, cv=kfolds)
stacking = StackingRegressor(estimators=[

    ('ridge', grid_ridge.best_estimator_), 

    ('lasso', grid_lasso.best_estimator_),

    ('elastic',grid_elastic.best_estimator_),

    ('xgb', grid_xgb.best_estimator_),

    ('lgb', grid_lgb.best_estimator_)

                                        ],

                             final_estimator=grid_xgb.best_estimator_,

                            passthrough=True)



stacking_model=fit_model(stacking)

loss=rmsle(y_train, stacking_model.predict(x_train))

print("Stacking:\n Loss: %.4f"%(loss))
def blend(x):

    return (

            0.1*grid_elastic.best_estimator_.predict(x)+

            0.05*grid_lasso.best_estimator_.predict(x)+

            0.1*grid_ridge.best_estimator_.predict(x)+ 

            0.1*grid_svr.best_estimator_.predict(x)+

            0.15*grid_gbr.best_estimator_.predict(x)+ 

            0.1*grid_lgb.best_estimator_.predict(x)+

            0.1*grid_xgb.best_estimator_.predict(x)+

            0.3*stacking.predict(x)

           )

y_pred = blend(x_train)

scores=rmsle(y_train, y_pred)

print("Loss:\n %.4f"%(scores))



Y_test = blend(X_test)
Y_test = np.expm1(Y_test)

def save_csv(y):

    submission = pd.DataFrame()

    submission['Id'] = test_ids

    submission['SalePrice'] = y

    submission.to_csv('submission.csv',index=False)



save_csv(Y_test)