# import libraries & parkages

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import datetime

import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score



#load some ML models

from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR
from scipy import stats

from scipy.stats import norm, skew #for some statistics



def myPlot_Norm(data, feature):

    plt.subplots(figsize=(12,4))

    plt.subplot(1,2,1)

    y=data[feature]

    sns.distplot(y, fit=norm)

    plt.title('hist plot without log')

    fig = plt.subplot(1,2,2)

    res = stats.probplot(y, plot=plt)

    

def myPlot_NormWlog(data,feature):

    plt.subplots(figsize=(12,4))

    plt.subplot(1,2,1)

    y=np.log1p(data[feature])

    sns.distplot(y, fit=norm)

    plt.title('hist plot With log')

    fig = plt.subplot(1,2,2)

    res = stats.probplot(y, plot=plt)

    

def rmse_cv(model,X_train,y):

    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error",cv = 5))

    return rmse
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
myPlot_Norm(train, 'SalePrice')

myPlot_NormWlog(train, 'SalePrice')
train['SalePrice'] = np.log1p(train['SalePrice'])
# take care of the outlier

plt.scatter(train['GrLivArea'],train['SalePrice'])

plt.xlim(0,6000)

plt.grid()
train.drop(train[train['GrLivArea']>4500].index,inplace=True)

plt.scatter(train['GrLivArea'],train['SalePrice'])

plt.xlim(0,6000)

plt.grid()
counts_missed = train.isnull().sum().sort_values(ascending=False)

percentage_missed = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)*100

MissingData_train = pd.concat([counts_missed, percentage_missed],

                              axis=1,

                              keys=['counts_missed','percentage_missed'])

MissingData_train.head(20)
features_to_drop = MissingData_train[(MissingData_train['percentage_missed']>2)].index.values
print('number of features to drop:', len(features_to_drop))

features_to_drop
plt.scatter(train.MiscVal, train.SalePrice)
features_to_drop = np.append(features_to_drop,['MiscVal'])
df_combine = [train,test]

for df in df_combine:

    print('Before dropping:', df.shape)

    df.drop(labels=features_to_drop, axis=1, inplace=True)

    print('AFTER dropping:', df.shape)
id_train = train['Id']

id_test = test['Id']
df_combine = pd.concat([train, test], keys=['train','test'], ignore_index=False)

df_combine.drop(labels='Id',axis=1, inplace=True)

df_combine.shape
df_combine.MSSubClass.unique()
# 'MSSubClass' looks like categorical feature

df_combine['MSSubClass'].replace({20: 'A',30: 'B',40: 'C',45: 'D',

                                  50: 'E',60: 'F',70: 'G',75: 'H',

                                  80: 'I',85: 'J',90: 'K',120:'L',

                                  150:'M',160:'N',180:'O',190:'P'}, inplace= True)



# below features are presenting the quality, should be in numerical.

df_combine['ExterQual'].replace({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}, inplace=True)

df_combine['ExterCond'].replace({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}, inplace=True)

df_combine['HeatingQC'].replace({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}, inplace=True)

df_combine['KitchenQual'].replace({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0}, inplace=True)
counts_missed = df_combine.isnull().sum().sort_values(ascending=False)

percentage_missed = (df_combine.isnull().sum()/df_combine.isnull().count()).sort_values(ascending=False)*100

MissingData_train = pd.concat([counts_missed, percentage_missed],

                              axis=1,

                              keys=['counts_missed','percentage_missed'])

MissingData_train = MissingData_train[MissingData_train.index != 'SalePrice']

MissingData_train.head(20)
freatures_cat = ['MasVnrType',  'MSZoning','Utilities', 'Functional',

                 'Exterior1st', 'Exterior2nd','Electrical',  'SaleType']

freautres_num = ['MasVnrArea', 'GarageArea', 'GarageCars', 'KitchenQual',

                 'BsmtFinSF1','TotalBsmtSF', 'BsmtFinSF2','BsmtUnfSF','BsmtFullBath','BsmtHalfBath']



for feature in freatures_cat:

    df_combine[feature].fillna(df_combine[feature].value_counts().index[0], inplace=True)

for feature in freautres_num:

    df_combine[feature].fillna(df_combine[feature].median(), inplace=True)
df_combine.isnull().sum().sort_values(ascending=False)[0:3]
df_combine.shape
# display the 20 most-correlated features

# saleprice correlation matrix

df = df_combine[0:train.shape[0]]

corrmat = df.corr()

k = 20 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

plt.subplots(figsize=(12,12))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=cols.values, xticklabels=cols.values, vmax=0.8)

plt.show()
#'GarageArea' - 'GarageCars' - 0.89

#'TotRmsAbvGrd' - 'GrLivArea' - 0.83

#'1stFlrSF' - 'TotalBsmtSF' - 0.80

plt.subplots(figsize=(16,4)), plt.subplot(1,3,1)

sns.regplot(df_combine.GarageArea, df_combine.GarageCars)

plt.subplot(1,3,2)

sns.regplot(df_combine.TotRmsAbvGrd, df_combine.GrLivArea)

plt.subplot(1,3,3)

sns.regplot(df_combine['1stFlrSF'], df_combine.TotalBsmtSF)
# drop 'GarageArea', 'TotRmsAbvGrd','1stFlrSF'

df_combine.drop(labels=['GarageArea', 'TotRmsAbvGrd','1stFlrSF'], axis=1, inplace=True)

df_combine.shape
# display the 20 most-correlated features

# saleprice correlation matrix

df = df_combine[0:train.shape[0]]

corrmat = df.corr()

k = 20 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

plt.subplots(figsize=(12,12))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=cols.values, xticklabels=cols.values, vmax=0.8)

plt.show()
df_combine.isnull().sum().sort_values(ascending=False)[0:3]
# a list of the categotical features we need to consider

features_cat = df_combine.select_dtypes(include='object').columns.values

features_num = df_combine.select_dtypes(exclude='object').columns.values

print('# of numerical features:', len(features_num))

print(type(features_num))

print(features_num.shape)

print('# of categorical features:', len(features_cat))

#features_cat

features_num
for feature in features_num:

    myPlot_Norm(df_combine[0:train.shape[0]], feature)
df_combine['Has2ndFlr'] = df_combine['2ndFlrSF'].apply(lambda x: 1 if x>0 else 0)

df_combine['HasBsmt'] = df_combine['TotalBsmtSF'].apply(lambda x: 1 if x>0 else 0)

df_combine['HasWoodDeck'] = df_combine['WoodDeckSF'].apply(lambda x: 1 if x>0 else 0)

df_combine['HasPorch'] = df_combine['OpenPorchSF'].apply(lambda x: 1 if x>0 else 0)

df_combine['HasMasVnr'] = df_combine['MasVnrArea'].apply(lambda x: 1 if x>0 else 0)
# below are the features showing skews and can have a log transform before we move further

features_for_log_trans = ['2ndFlrSF','BedroomAbvGr','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','ExterQual','Fireplaces',

                          'GarageCars','GrLivArea','HeatingQC','KitchenQual','LotArea','MasVnrArea',

                          'OpenPorchSF','OverallCond', 'OverallQual','TotalBsmtSF','WoodDeckSF']
for feature in features_for_log_trans:

    df_combine[feature] = np.log1p(df_combine[feature])
df = df_combine[0:train.shape[0]]

for f in features_num:

    plt.subplots(figsize=(8,5))

    sns.regplot(df[f], df['SalePrice'])

    plt.grid()

    plt.xlabel(f)

    plt.ylabel('SalePrice')
week_num_features = ['BedroomAbvGr', 'BsmtFullBath','BsmtHalfBath','HalfBath','KitchenAbvGr',

                     'MoSold', 'PoolArea','YrSold']

for feature in week_num_features:

    df_combine[feature] = df_combine[feature].astype(str)
# regenerate the list of the categotical features we need to consider

features_cat = df_combine.select_dtypes(include='object').columns.values

features_num = df_combine.select_dtypes(exclude='object').columns.values

print('# of numerical features:', len(features_num))

print('# of categorical features:', len(features_cat))

features_cat
df = df_combine[0:train.shape[0]]

for f in features_cat:

    plt.subplots(figsize=(8,5))

    sns.boxplot(df[f], df['SalePrice'])

    plt.grid()

    plt.xlabel(f)

    plt.ylabel('SalePrice')
df_combine['CentralAir'].replace({'Y': 1, 'N':0}, inplace=True)

df_combine['Electrical'].replace({'Mix':0, 'FuseP':1, 'FuseF':2, 'FuseA':3, 'SBrkr':4}, inplace=True)

df_combine['Heating'].replace({'Floor':0, 'Grav':1, 'Wall':2, 'OthW':3, 'GasW':4, 'GasA':5}, inplace=True)

df_combine['HouseStyle'].replace({'1.5Unf':0, 'SFoyer':1, '1.5Fin':2, '2.5Unf':3, '1Story':4,

                                 'SLvl':5, '2Story':6, '2.5Fin':7}, inplace=True)

df_combine['LandContour'].replace({'Bnk':0, 'Lvl':1, 'Low':2, 'HLS':3}, inplace=True)

df_combine['LandSlope'].replace({'Gtl':0, 'Mod':1, 'Sev':2}, inplace=True)

df_combine['LotShape'].replace({'Reg':0, 'IR1':1, 'IR3':2, 'IR2':3}, inplace=True)

df_combine['MSZoning'].replace({'C (all)':0, 'RM':1, 'RH':2, 'RL':3, 'FV':4}, inplace=True)

df_combine['MasVnrType'].replace({'None':0, 'BrkCmn':1, 'BrkFace':2, 'Stone':3}, inplace=True)

df_combine['PavedDrive'].replace({'N':0, 'P':1, 'Y':2}, inplace=True)

df_combine['Street'].replace({'Grvl':0, 'Pave':1}, inplace=True)

df_combine['Utilities'].replace({'NoSewa':0, 'AllPub':1}, inplace=True)
print('Before encoding:', df_combine.shape)



df_combine_coded = pd.get_dummies(df_combine)

print('After encoding:', df_combine_coded.shape)



x = df_combine_coded[0:train.shape[0]].drop(labels='SalePrice', axis=1)

y = df_combine_coded[0:train.shape[0]]['SalePrice']

x_test = df_combine_coded[train.shape[0]:].drop(labels='SalePrice', axis=1)

print('Train/Test after encoding:',x.shape, y.shape, x_test.shape)
from sklearn.preprocessing import RobustScaler

RS = RobustScaler()

x = RS.fit_transform(x)

x_test = RS.fit_transform(x_test)
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

print('x_train ---- y_train ---- x_val ---- y_val ---- x_test')

x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape
Ns = [100,200,300,400,500,600,700]

score_rf = [rmse_cv(RandomForestRegressor(n_estimators=n), x, y).mean() for n in Ns]

print(score_rf)

score_rf = pd.Series(score_rf, index=Ns)

score_rf.plot()
reg_rf = RandomForestRegressor(n_estimators=600)

reg_rf.fit(x, y)

# Need to use np.expm1 since we used np.log1p on 'SalePrice'

y_pred = np.expm1(reg_rf.predict(x_test))  

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_RF.csv', index=False)
alphas = [1, 3, 5, 7, 9, 10,11,12, 13, 15, 20,30,50]

score_ridge = [rmse_cv(Ridge(alpha=a), x, y).mean() for a in alphas]

print(score_ridge)

score_ridge = pd.Series(score_ridge, index=alphas)

score_ridge.plot()
reg_ridge = Ridge(alpha=10)

reg_ridge.fit(x, y)

y_pred = np.expm1(reg_ridge.predict(x_test))

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_ridge.csv', index=False)
alphas = np.logspace(-4,-3,10)

score_lasso = [rmse_cv(Lasso(alpha=a), x, y).mean() for a in alphas]

print(score_lasso)

score_lasso = pd.Series(score_lasso, index=alphas)

score_lasso.plot()
score_lasso
reg_lasso = Lasso(alpha=0.000278)

reg_lasso.fit(x, y)

y_pred = np.expm1(reg_lasso.predict(x_test))

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_lasso.csv', index=False)
alphas = np.logspace(-4,-3,10)

score_elastic = [rmse_cv(ElasticNet(alpha=a, l1_ratio=0.97), x, y).mean() for a in alphas]

print(score_elastic)

score_elastic = pd.Series(score_elastic, index=alphas)

score_elastic.plot()
score_elastic
reg_elastic = ElasticNet(alpha=0.000278,l1_ratio=0.97)

reg_elastic.fit(x, y)

y_pred = np.expm1(reg_elastic.predict(x_test))

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_elastic.csv', index=False)
reg_gbr = GradientBoostingRegressor(n_estimators=400,

                                    learning_rate=0.056234, 

                                    max_depth=3,

                                    min_samples_leaf=5, 

                                    min_samples_split=20)

score_gbr = rmse_cv(reg_gbr, x, y)

print('{:.5f}:+/-{:.5f}'.format(score_gbr.mean(),score_gbr.std()))
reg_gbr = GradientBoostingRegressor(n_estimators=400,

                                    learning_rate=0.056234, 

                                    max_depth=3,

                                    min_samples_leaf=5, 

                                    min_samples_split=20)

reg_gbr.fit(x, y)

y_pred = np.expm1(reg_gbr.predict(x_test))

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_GradientBoosting.csv', index=False)
from xgboost import XGBRegressor

reg_XGBoost = XGBRegressor(n_estimators=700,

                       max_depth=3,

                       learning_rate=0.07,

                       subsample=0.9,

                       colsample_bytree=0.7)

score_xgboost = rmse_cv(reg_XGBoost, x, y)

print('{:.5f}:+/-{:.5f}'.format(score_xgboost.mean(),score_xgboost.std()))
reg_XGBoost = XGBRegressor(n_estimators=700,

                       max_depth=3,

                       learning_rate=0.07,

                       subsample=0.9,

                       colsample_bytree=0.7)

reg_XGBoost.fit(x, y)

y_pred = np.expm1(reg_XGBoost.predict(x_test))

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_XGBoost.csv', index=False)
from sklearn.svm import SVR
Cs = [0.01, 0.05, 0.1, 0.5]

score_svm= [rmse_cv(SVR(C=c, kernel='linear'), x, y).mean() for c in Cs]

print(score_svm)

score_svm = pd.Series(score_svm, index=Cs)

score_svm.plot()
reg_svm = SVR(C=c, kernel='linear')

reg_svm.fit(x, y)

y_pred = np.expm1(reg_svm.predict(x_test))

results_to_submission = pd.DataFrame({

    'Id': id_test,

    'SalePrice': y_pred

})

results_to_submission.to_csv('submission_svm.csv', index=False)