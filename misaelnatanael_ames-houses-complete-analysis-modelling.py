import numpy as np

import pandas as pd

pd.set_option('mode.chained_assignment',None)

from pandas import Series, DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df
nas = df.isna().sum()

nas[nas>0]
cols = ['Alley', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

df[cols] = df[cols].fillna('None')
df[(df['BsmtExposure'].isna()) & (df['TotalBsmtSF']>0)].iloc[:,30:40]
df[(df['BsmtFinType2'].isna()) & (df['TotalBsmtSF']>0)].iloc[:,30:40]
df.loc[(df['BsmtExposure'].isna()) & (df['TotalBsmtSF']>0),'BsmtExposure'] = 'No'

df.loc[(df['BsmtFinType2'].isna()) & (df['TotalBsmtSF']>0),'BsmtFinType2'] = 'Unf'

cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

df[cols] = df[cols].fillna('None')
nas = df.isna().sum()

nas[nas>0]
cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Condition1', 'Condition2', 'BldgType', 

        'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'GarageType', 

        'MiscFeature', 'SaleType', 'SaleCondition']

fig = plt.figure(figsize=(14,50))

for c,i in zip(cols, range(1,21)):

    ax = fig.add_subplot(10,2,i)

    sns.boxplot(x=c,y='SalePrice',data=df)

fig.suptitle('Relation between house sale price with all nominal variables',y=0.999)

fig.tight_layout(pad=4.0)

plt.figure(figsize=(14,5))

sns.boxplot(x='Exterior1st',y='SalePrice',data=df)

plt.figure(figsize=(14,5))

sns.boxplot(x='Exterior2nd',y='SalePrice',data=df)

plt.figure(figsize=(24,5))

sns.boxplot(x='Neighborhood',y='SalePrice',data=df)
cols = ['LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond',  'BsmtQual', 'BsmtCond', 

        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 

        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

fig = plt.figure(figsize=(14,57))

for c,i in zip(cols, range(1,24)):

    ax = fig.add_subplot(12,2,i)

    sns.boxplot(x=c,y='SalePrice',data=df)

fig.suptitle('Relation between house sale price with all ordinal variables', y=0.999)

fig.tight_layout(pad=4.0)
cols1 = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

         'GarageCars', 'MoSold', 'YrSold']

cols2 = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

fig = plt.figure(figsize=(15,16))

for c,i in zip(cols1, range(1,12)):

    ax = fig.add_subplot(4,3,i)

    sns.boxplot(x=c,y='SalePrice',data=df)

fig.suptitle('Relation between house sale price with all discrete numerical variables', y=0.999)

fig.tight_layout(pad=2.0)

fig = plt.figure(figsize=(22,5))

for c,i in zip(cols2, range(1,4)):

    ax = fig.add_subplot(1,3,i)

    sns.scatterplot(x=c,y='SalePrice',data=df)
cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

        'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 

        'PoolArea', 'MiscVal']

fig = plt.figure(figsize=(15,30))

for c,i in zip(cols, range(1,20)):

    ax = fig.add_subplot(7,3,i)

    sns.scatterplot(x=c,y='SalePrice',data=df)

fig.suptitle('Relation between house sale price with all discrete numerical variables', y=0.999)

fig.tight_layout(pad=2.0)
plt.rcParams['figure.figsize'] = [42, 24]

sns.heatmap(df.drop(columns=['Id','MSSubClass']).corr(method='pearson'),annot=True,cmap='RdBu_r')
fig = plt.figure(figsize=(14,11))

fig.add_subplot(2,2,1)

sns.boxplot(x='GarageCars',y='GarageArea',data=df)

fig.add_subplot(2,2,2)

sns.boxplot(x='TotRmsAbvGrd',y='GrLivArea',data=df)

fig.add_subplot(2,2,3)

sns.scatterplot(x='YearBuilt',y='GarageYrBlt',data=df)

fig.add_subplot(2,2,4)

sns.scatterplot(x='1stFlrSF',y='TotalBsmtSF',data=df)
plt.figure(figsize=(7,5))

sns.scatterplot(x='GrLivArea',y='SalePrice',data=df)
df = df[df['GrLivArea']<4000]
nas = df.isna().sum()

nas[nas>0]
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df.loc[:,'Electrical'] = imp.fit_transform(np.array(df['Electrical']).reshape(-1,1))
df['MasVnrType'] = df['MasVnrType'].fillna('None')

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df.loc[df['GarageYrBlt'].isna(),'GarageYrBlt'] = df.loc[df['GarageYrBlt'].isna(),'YearBuilt']
df.loc[df['LotFrontage'].isna(),'LotFrontage'] = 0
nas = df.isna().sum()

nas[nas>0]
#encoding ordinal features

df = df.replace({'LotShape' : {'Reg' : 0, 'IR1' : 1, 'IR2' : 2, 'IR3' : 3},

                'Utilities' : {'AllPub' : 4, 'NoSewr' : 3, 'NoSeWa' : 2, 'ELO' : 1},

                'LandSlope' : {'Gtl' : 1, 'Mod' : 2, 'Sev' : 3},

                'ExterQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'ExterCond' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'BsmtQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'BsmtCond' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'BsmtExposure' : {'Gd' : 4, 'Av' : 3, 'Mn' : 2, 'No' : 1, 'None' : 0},

                'BsmtFinType1' : {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'None' : 0},

                'BsmtFinType2' : {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'None' : 0},

                'HeatingQC' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'Electrical' : {'SBrkr' : 5, 'FuseA' : 4, 'FuseF' : 3, 'FuseP' : 2, 'Mix' : 1},

                'KitchenQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'Functional' : {'Typ' : 8, 'Min1' : 7, 'Min2' : 6, 'Mod' : 5, 'Maj1' : 4, 'Maj2' : 3, 'Sev' : 2, 'Sal' : 1},

                'FireplaceQu' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'GarageFinish' : {'Fin' : 3, 'RFn' : 2, 'Unf' : 1, 'None' : 0},

                'GarageQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'GarageCond' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'PavedDrive' : {'Y' : 2, 'P' : 1, 'N' : 0},

                'PoolQC' : {'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'None' : 0},

                'Fence' : {'GdPrv' : 4, 'MnPrv' : 3, 'GdWo' : 2, 'MnWw' : 1, 'None' : 0},

                })
#encoding nominal features

df2 = pd.concat([df,pd.get_dummies(df['MSSubClass'],prefix='SubClass',drop_first=True)],axis=1)

df3 = pd.concat([df2,pd.get_dummies(df2[['MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',

                                        'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                                        'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',

                                        'GarageType','MiscFeature','SaleType','SaleCondition']],drop_first=True)],axis=1)

df4 = df3.drop(columns=['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',

                                        'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                                        'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',

                                        'GarageType','MiscFeature','SaleType','SaleCondition'])
#drop Id column

data = df4.drop(columns='Id')

data
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

X = data.drop('SalePrice',axis=1)

y = data['SalePrice']
from sklearn.metrics import mean_squared_log_error, make_scorer



def funct(y_true,y_pred):

    y_new = np.maximum(y_pred,np.zeros(len(y_pred)))

    t = mean_squared_log_error(y_true,y_new)

    return np.sqrt(t)



RMSLE = make_scorer(funct, greater_is_better=False)
print('Number of features: %d' %(len(X.columns)))
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

fs = SelectKBest(score_func=f_regression, k='all')

fs.fit(X,y)

indices = np.argsort(fs.scores_)[::-1]

cc = DataFrame({'feature score':Series(fs.scores_),'features':Series(X.columns)})    

plt.figure(figsize=(10,35))

sns.barplot(x='feature score',y='features',data=cc.head(50).sort_values(by='feature score',ascending=False))

plt.title('Feature importances based on regression')
new_col = np.array(Series(X.columns[indices]).head(100))

X_new1 = X[new_col]
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

pipe = Pipeline([("scaler", StandardScaler()),("reg", Ridge())])

param_grid = {'reg__alpha': [630, 640, 650, 680, 700]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring=RMSLE)

grid.fit(X_new1,y)

print('Best hyperparameter: ', grid.best_params_)

print('Best cross validation score: ', -grid.best_score_)
y_true = y

y_pred = grid.predict(X_new1)

plt.figure(figsize=(7,5))

sns.scatterplot(y_true,y_pred)

plt.plot(y_true,y_true,color='k')

plt.xlabel('SalePrice (actual)')

plt.ylabel('SalePrice (predicted)')
from sklearn.ensemble import ExtraTreesRegressor

forest = ExtraTreesRegressor(n_estimators=100,

                              random_state=0,n_jobs=4)

forest.fit(X, y)

indices = np.argsort(forest.feature_importances_)[::-1]

cc = DataFrame({'feature score':Series(forest.feature_importances_),'features':Series(X.columns)})    

plt.figure(figsize=(10,35))

sns.barplot(x='feature score',y='features',data=cc.head(50).sort_values(by='feature score',ascending=False))
new_col = np.array(Series(X.columns[indices]).head(100))

X_new2 = X[new_col]
from sklearn.ensemble import ExtraTreesRegressor

forest = ExtraTreesRegressor(random_state=0,n_jobs=4,n_estimators=500)

param_grid = {'min_samples_split': [2, 3, 4], 'max_depth' : [40, 50]}

grid = GridSearchCV(forest, param_grid=param_grid, cv=5, scoring=RMSLE)

grid.fit(X_new2,y)

print('Best hyperparameter: ', grid.best_params_)

print('Best cross validation score: ', -grid.best_score_)
y_true = y

y_pred = grid.predict(X_new2)

plt.figure(figsize=(7,5))

sns.scatterplot(y_true,y_pred)

plt.plot(y_true,y_true,color='k')

plt.xlabel('SalePrice (actual)')

plt.ylabel('SalePrice (predicted)')
from sklearn.ensemble import GradientBoostingRegressor

gdb = GradientBoostingRegressor(n_estimators = 300, random_state=0)

gdb.fit(X,y)

indices = np.argsort(gdb.feature_importances_)[::-1]

cc = DataFrame({'feature score':Series(gdb.feature_importances_),'features':Series(X.columns)})    

plt.figure(figsize=(10,35))

sns.barplot(x='feature score',y='features',data=cc.head(50).sort_values(by='feature score',ascending=False))
new_col = np.array(Series(X.columns[indices]).head(100))

X_new3 = X[new_col]
from sklearn.ensemble import GradientBoostingRegressor

gdb = GradientBoostingRegressor(random_state=0, n_estimators=400)

param_grid = {'min_samples_split' : [25, 30, 35], 'min_samples_leaf' : [2, 3]}

grid = GridSearchCV(gdb, param_grid=param_grid, cv=5, scoring=RMSLE)

grid.fit(X_new3,y)

print('Best hyperparameter: ', grid.best_params_)

print('Best cross validation score: ', -grid.best_score_)
y_true = y

y_pred = grid.predict(X_new3)

plt.figure(figsize=(7,5))

sns.scatterplot(y_true,y_pred)

plt.plot(y_true,y_true,color='k')

plt.xlabel('SalePrice (actual)')

plt.ylabel('SalePrice (predicted)')
from sklearn.metrics import r2_score



#Ridge regression

pipe = Pipeline([("scaler", StandardScaler()),("reg", Ridge(alpha=630))])

score = cross_val_score(pipe, X_new1, y, cv=5, scoring='r2')

print('R\u00b2 score for Ridge regression: %f' %(score.mean()))



#Extra Trees

forest = ExtraTreesRegressor(random_state=0, n_jobs=4, n_estimators=500, max_depth=40, min_samples_split=3)

score = cross_val_score(forest, X_new2, y, cv=5, scoring='r2')

print('R\u00b2 score for Extra Trees: %f' %(score.mean()))



#Gradient Boosting

gdb = GradientBoostingRegressor(random_state=0, n_estimators=400, min_samples_leaf=3, min_samples_split=25)

score = cross_val_score(forest, X_new3, y, cv=5, scoring='r2')

print('R\u00b2 score for Gradient Boosting: %f' %(score.mean()))
dt = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

dt
nas = dt.isna().sum()

nas[nas > 0]
cols = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','MasVnrType', 'BsmtFinType1', 'BsmtFinType2']

dt[cols] = dt[cols].fillna('None')

dt.loc[dt['SaleType'].isna(),'SaleType'] = 'Oth'
cols = ['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath',

       'GarageCars','GarageArea']

dt[cols] = dt[cols].fillna(0)
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional']

dt[cols] = imp.fit_transform(dt[cols])
dt[(dt['BsmtQual'].isna()) & (dt['TotalBsmtSF']>0)].iloc[:,30:40]
dt[(dt['BsmtCond'].isna()) & (dt['TotalBsmtSF']>0)].iloc[:,30:40]
dt[(dt['BsmtExposure'].isna()) & (dt['TotalBsmtSF']>0)].iloc[:,30:40]
dt.loc[(dt['BsmtQual'].isna()) & (dt['TotalBsmtSF']>0),'BsmtQual'] = dt['BsmtQual'].value_counts().idxmax()

dt.loc[(dt['BsmtCond'].isna()) & (dt['TotalBsmtSF']>0),'BsmtCond'] = dt['BsmtCond'].value_counts().idxmax()

dt.loc[(dt['BsmtExposure'].isna()) & (dt['TotalBsmtSF']>0),'BsmtExposure'] = 'No'

cols = ['BsmtQual','BsmtCond','BsmtExposure']

dt[cols] = dt[cols].fillna('None')
dt[(dt['GarageFinish'].isna()) & (dt['GarageArea']>0)].iloc[:,57:67]
dt[(~dt['GarageType'].isna()) & (dt['GarageArea'] == 0)].iloc[:,57:67]
dt.loc[(dt['GarageFinish'].isna()) & (dt['GarageArea']>0),'GarageFinish'] = 'Unf'

dt.loc[(dt['GarageQual'].isna()) & (dt['GarageArea']>0),'GarageQual'] = dt['GarageQual'].value_counts().idxmax()

dt.loc[(dt['GarageCond'].isna()) & (dt['GarageArea']>0),'GarageCond'] = dt['GarageCond'].value_counts().idxmax()

dt.loc[(~dt['GarageType'].isna()) & (dt['GarageArea'] == 0),'GarageType'] = 'None'

cols = ['GarageType','GarageFinish','GarageQual','GarageCond']

dt[cols] = dt[cols].fillna('None')

dt.loc[dt['GarageYrBlt'].isna(),'GarageYrBlt'] = dt.loc[dt['GarageYrBlt'].isna(),'YearBuilt']
nas = dt.isna().sum()

nas[nas > 0]
#encoding ordinal features

dtt = dt.replace({'LotShape' : {'Reg' : 0, 'IR1' : 1, 'IR2' : 2, 'IR3' : 3},

                'Utilities' : {'AllPub' : 4, 'NoSewr' : 3, 'NoSeWa' : 2, 'ELO' : 1},

                'LandSlope' : {'Gtl' : 1, 'Mod' : 2, 'Sev' : 3},

                'ExterQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'ExterCond' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'BsmtQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'BsmtCond' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'BsmtExposure' : {'Gd' : 4, 'Av' : 3, 'Mn' : 2, 'No' : 1, 'None' : 0},

                'BsmtFinType1' : {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'None' : 0},

                'BsmtFinType2' : {'GLQ' : 6, 'ALQ' : 5, 'BLQ' : 4, 'Rec' : 3, 'LwQ' : 2, 'Unf' : 1, 'None' : 0},

                'HeatingQC' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'Electrical' : {'SBrkr' : 5, 'FuseA' : 4, 'FuseF' : 3, 'FuseP' : 2, 'Mix' : 1},

                'KitchenQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1},

                'Functional' : {'Typ' : 8, 'Min1' : 7, 'Min2' : 6, 'Mod' : 5, 'Maj1' : 4, 'Maj2' : 3, 'Sev' : 2, 'Sal' : 1},

                'FireplaceQu' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'GarageFinish' : {'Fin' : 3, 'RFn' : 2, 'Unf' : 1, 'None' : 0},

                'GarageQual' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'GarageCond' : {'Ex' : 5, 'Gd' : 4, 'TA' : 3, 'Fa' : 2, 'Po' : 1, 'None' : 0},

                'PavedDrive' : {'Y' : 2, 'P' : 1, 'N' : 0},

                'PoolQC' : {'Ex' : 4, 'Gd' : 3, 'TA' : 2, 'Fa' : 1, 'None' : 0},

                'Fence' : {'GdPrv' : 4, 'MnPrv' : 3, 'GdWo' : 2, 'MnWw' : 1, 'None' : 0},

                })
#encoding nominal features

dt2 = pd.concat([dtt,pd.get_dummies(dtt['MSSubClass'],prefix='SubClass',drop_first=True)],axis=1)

dt3 = pd.concat([dt2,pd.get_dummies(dt2[['MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',

                                        'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                                        'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',

                                        'GarageType','MiscFeature','SaleType','SaleCondition']],drop_first=True)],axis=1)

dt4 = dt3.drop(columns=['MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood',

                                        'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',

                                        'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',

                                        'GarageType','MiscFeature','SaleType','SaleCondition'])

dt4['Exterior1st_Stone'] = np.zeros(len(dt4))

dt4
X_test = dt4[new_col]

y_pred = grid.predict(X_test)

final = pd.concat([dt, Series(y_pred,name='SalePrice')], axis=1)

final
final.to_csv("output_ames.csv") 