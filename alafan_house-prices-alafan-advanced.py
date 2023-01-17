# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import cross_val_score,KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest,f_classif

from sklearn.model_selection import GridSearchCV

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)



import lightgbm as lgb

from catboost import CatBoostRegressor

import xgboost as xgb

from xgboost import XGBRFRegressor



from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import validation_curve, learning_curve,ShuffleSplit



import seaborn as sns

import matplotlib.pylab as plt

import category_encoders as ce

import sklearn

from matplotlib import pyplot as plt



%matplotlib inline

def scoring(clf, X, y):

    kf = KFold(5, shuffle=True, random_state=19).get_n_splits(X)

    cvs = (cross_val_score(clf, X, y, cv=kf,scoring='neg_mean_squared_log_error')*(-1))**0.5

    print('Mean: {}, std: {}'.format(cvs.mean(),cvs.std()))

    return cvs
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head()
feature_list = list(test.columns)

#X_train = pd.DataFrame()
def feature_split(train):

    quant_ftr = []

    cat_ftr = []

    for feature in train.columns:

        if train[feature].dtype in ['float64', 'int64']:

            quant_ftr.append(feature)

        else:

            cat_ftr.append(feature)

    return quant_ftr,cat_ftr





def label_encode(train, ):

    quant_ftr, cat_ftr = feature_split(train)

    encoder = LabelEncoder()

    encoded = train[cat_ftr].fillna('NAN').apply(encoder.fit_transform)

    data = train[quant_ftr].join(encoded)

    return data

y = train['SalePrice']

train_l = label_encode(train)

test_l = label_encode(test)
train_l.head()
lgbm = lgb.LGBMRegressor()

cat = CatBoostRegressor(random_state=42, silent=True)

xgboost = xgb.XGBRegressor(random_state=42,silent=True)
#Learning curves

def learning_curve_plot(estimator, X, y,ax,train_sizes=np.linspace(.1, 1.0, 5),n_jobs = None):

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,

                                                            train_sizes=train_sizes,

                                                            scoring='neg_mean_squared_log_error')

    print('READY: ',estimator)

    train_scores = (train_scores*-1)**0.5

    test_scores = (test_scores*-1)**0.5

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,

                         train_scores_mean + train_scores_std, alpha=0.1,

                         color="r")

    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,

                         test_scores_mean + test_scores_std, alpha=0.1,

                         color="g")

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",

                 label="Training score")

    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",

                 label="Cross-validation score")

    ax.legend(loc="best")

    ax.set_title('Validation with '+str(estimator)[:list(str(estimator)).index('(')])

    

fig,ax = plt.subplots(1,2,figsize =(16,5))

for est, i in zip([xgboost,lgbm],[0,1,]):

    learning_curve_plot(est, train_l,y, ax[i])
param_range =  list(range(1,11))

train_scores, test_scores = validation_curve(xgboost, train_l, y, "max_depth",

                                           param_range,

                                             scoring='neg_mean_squared_log_error',

                                           cv=5,)
#Validation Curve max_depth

train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



plt.title("Validation Curve with XGBoost")

plt.xlabel("max_depth")

plt.ylabel("Score")

#plt.ylim(0.95, 1.05)

lw = 2

#fig,ax = plt.subplots()

plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)

plt.fill_between(param_range, train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, alpha=0.2,

                 color="darkorange", lw=lw)

plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std, alpha=0.2,

                 color="navy", lw=lw)

plt.legend(loc="best")

plt.show()
baseline = scoring(xgboost, train_l[feature_list].fillna(0),y).mean()
fig, ax = plt.subplots(1,2,figsize=(15,5))



sns.distplot(train.SalePrice, fit = norm, ax = ax[0])

stats.probplot(train.SalePrice,plot = ax[1])
train['SalePrice'] = np.log(train_l['SalePrice'])
def showprob(ftr, train = train):

    fig, ax = plt.subplots(1,2,figsize=(15,5))



    sns.distplot(train[ftr], fit = norm, ax = ax[0])

    stats.probplot(train[ftr],plot = ax[1])

showprob('SalePrice')
from numpy import loadtxt

from xgboost import plot_importance

from matplotlib import pyplot



xgboost.fit(train_l[list(test.columns)], y)

# plot feature importance

fig, ax = plt.subplots(1,1,figsize=(10,20))

plot_importance(xgboost, ax = ax)



pyplot.show()
#cat_baseline = scoring(cat, train_l[feature_list].fillna(0),y).mean()
def update_baseline(baseline = baseline,clf = xgboost, train = train_l, ftr = feature_list, y = y):

    scr = scoring(clf, train_l[ftr],y).mean()

    diff = round((baseline - scr)/baseline*100,2)

    if baseline<scr:

        print('Getting WORSE at {}% (old = {})'.format(diff, baseline))

        return baseline

    else:

        print('Getting BETTER at {}% (old = {})'.format(diff,baseline))

        return scr

def check_baseline(baseline = baseline):



    ftr = list(test.columns)

    train_l = label_encode(train)

    test_l = label_encode(test)

    baseline = update_baseline(baseline, ftr=ftr,train = train)

    return baseline
del train['Id']

del test['Id']
baseline = check_baseline(baseline)
#correlation matrix

corrmat = train.corr()



f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_l[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 

                 annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
sns.pairplot(train[cols])
nan = pd.DataFrame()

for feature in list(test.columns):

    nan.loc[feature,'share'] = len(train[pd.isna(train[feature])])/len(train)

nan = nan[nan.share>0].sort_values(by = 'share', ascending = False).reset_index()

nan.columns = ['feature','share']
nan[nan.share>0.05]
for f in nan[nan.share>0.1].feature:

    del train[f]

    del test[f]
baseline = check_baseline(baseline)
corrmat = train.corr()
corrmat['SalePrice'] = corrmat.SalePrice.apply(abs)
corrmat.SalePrice.sort_values(ascending = False).head(10)
showprob('OverallQual')
showprob('GrLivArea')
train['GrLivArea'] = np.log(train.GrLivArea)

test['GrLivArea'] = np.log(test.GrLivArea)
showprob('GrLivArea')


baseline = check_baseline(baseline)
showprob('GarageCars')
showprob('GarageArea')


del train['GarageArea']

del test['GarageArea']

baseline = check_baseline(baseline)
train['GarageYrBlt'] = train.apply(lambda x: 1 if x['GarageYrBlt']==x['YearBuilt'] else 0,axis = 1)

test['GarageYrBlt'] = test.apply(lambda x: 1 if x['GarageYrBlt']==x['YearBuilt'] else 0,axis = 1)
baseline = check_baseline(baseline)
sns.scatterplot(x = train['1stFlrSF'],y = train['TotalBsmtSF'])


del train['1stFlrSF']

del test['1stFlrSF']

baseline = check_baseline(baseline)
showprob('TotalBsmtSF')
train.loc[train['TotalBsmtSF']>0,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

test.loc[test['TotalBsmtSF']>0,'TotalBsmtSF'] = np.log(test['TotalBsmtSF'])



showprob('TotalBsmtSF')
train['Remoded'] = train.apply(lambda x: 1 if x['YearRemodAdd']!=x['YearBuilt'] else 0, axis = 1)

test['Remoded'] = test.apply(lambda x: 1 if x['YearRemodAdd']!=x['YearBuilt'] else 0, axis = 1)
baseline = check_baseline(baseline)
sns.scatterplot(x = train.GrLivArea, y = train.TotRmsAbvGrd)
train = train.drop(train[train['GrLivArea']>4000].index)
del train['TotRmsAbvGrd']

del test['TotRmsAbvGrd']
baseline = check_baseline(baseline)
basement = ['BsmtCond','BsmtExposure','BsmtFinSF1','BsmtFinSF2','BsmtFinType1','BsmtFinType2','BsmtFullBath','BsmtHalfBath',

           'BsmtQual','BsmtUnfSF','TotalBsmtSF']
sns.pairplot(train_l[basement+['SalePrice']])
train[basement+['SalePrice']].head(10)
train[['YearRemodAdd','YearBuilt']].head()
train['Remoded'] = train.apply(lambda x: 1 if x['YearRemodAdd']!=x['YearBuilt'] else 0, axis = 1)

test['Remoded'] = test.apply(lambda x: 1 if x['YearRemodAdd']!=x['YearBuilt'] else 0, axis = 1)
del train['YearRemodAdd']

del test['YearRemodAdd']
ftr = list(test.columns)

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)
#cat_baseline =  update_baseline(clf = cat, ftr = list(test.columns),baseline = cat_baseline)
train[train.PoolArea>0][['PoolQC','PoolArea','LotArea']].head()
train['PoolQC'].value_counts()
del train['PoolArea']

del test['PoolArea']

del train['PoolQC']

del test['PoolQC']
ftr = list(test.columns)



train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)

#cat_baseline =  update_baseline(clf = cat, ftr = list(test.columns),baseline = cat_baseline)
train[train.MiscVal>0][['MiscFeature','MiscVal']].head()
train.MiscFeature.value_counts()
ftr = list(test.columns)

ftr.remove('MiscFeature')

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)
del train['MiscFeature']

del test['MiscFeature']
ftr = list(test.columns)

ftr.remove('MiscVal')

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)
del train['MiscVal']

del test['MiscVal']
train[['FireplaceQu','Fireplaces']].head()
ftr = list(test.columns)

ftr.remove('FireplaceQu')

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)
del train['FireplaceQu']

del test['FireplaceQu']
train['LotShape'] = train.LotShape.apply(lambda x: 1 if x=='Regular' else 0)

test['LotShape'] = test.LotShape.apply(lambda x: 1 if x=='Regular' else 0)
ftr = list(test.columns)

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)
train.LandContour.value_counts()
train['LandContourB'] = train.LandContour.apply(lambda x: 1 if x=='Lvl' else 0)

test['LandContourB'] = test.LandContour.apply(lambda x: 1 if x=='Lvl' else 0)



ftr = list(test.columns)

ftr.remove('LandContour')

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)

del train['LandContourB']

del test['LandContourB']
xgboost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

ftr = list(test.columns)

train_l = label_encode(train)

baseline = update_baseline(baseline=baseline, ftr = ftr)
#'LowQualFinSF','RoofMati','BedroomAbvGr','Fence','RoofStyle','BsmtCond','Foundation','YrSold','GarageQual','HeatingQC','BldgType','MasVnrType',

feature_list = list(test.columns)

for ftr in ['HalfBath'

           ]:

    try:

        feature_list.remove(ftr)

    except:

        pass

baseline = update_baseline(baseline = baseline, ftr = feature_list)
import category_encoders as ce





target_ftr = ['Neighborhood','MSZoning','Condition1','Condition2',

              'HouseStyle','OverallQual','OverallCond','ExterQual']



train_l = label_encode(train)[feature_list]

test_l = label_encode(test)[feature_list]



target_enc = ce.TargetEncoder(cols=target_ftr)

target_enc.fit(train_l[target_ftr], y)



# Transform the features, rename the columns with _target suffix, and join to dataframe

train_l = train_l.join(target_enc.transform(train_l[target_ftr]).add_suffix('_target'))

test_l = test_l.join(target_enc.transform(test_l[target_ftr]).add_suffix('_target'))

feature_list = [x for x in list(test_l.columns) if x not in target_ftr]




xgboost.fit(train_l[feature_list].fillna(0),y)

pred = xgboost.predict(test_l[feature_list].fillna(0))



result = pd.DataFrame({'Id': test.Id, 'SalePrice': pred})

result.to_csv('submission.csv',index = False)