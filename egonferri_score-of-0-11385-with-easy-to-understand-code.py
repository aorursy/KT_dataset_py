import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore') #to avoid some ugly warnings



sns.set(palette='OrRd_r') #sets the color palette of the plot

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(len(train.columns), len(train.index))
train.head()
train.describe()
train.columns
train['SalePrice'].describe()
plt.subplots(figsize=(15, 8))

sns.distplot(train['SalePrice'])
train['SalePrice'].skew()
f, ax= plt.subplots(figsize=(15, 8)) #size of the plot

sns.boxplot(train['SalePrice'], color='navajowhite', linewidth=0.5)

print(train['SalePrice'].median())
corrmat = train.corr()

plt.subplots(figsize=(25, 10))

sns.heatmap(corrmat, square=True, cmap='Oranges', vmax=.7, vmin=0)
correlations=corrmat.nlargest(100, 'SalePrice')['SalePrice']

correlations
cols=corrmat.nlargest(25, 'SalePrice').index

train_2=train.filter(items=cols)

corrmat = train_2.corr()

plt.subplots(figsize=(15, 10))

sns.heatmap(corrmat, square=True, cmap='Oranges', vmin=0)
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

sns.pairplot(train[cols], size = 2.5, kind='reg', diag_kind='kde')

plt.show()
def varplot(var): #a function that takes as imput two variables and return the regplot between them

    sns.regplot(x=train[var], y=train['SalePrice'])
varplot('SalePrice')
varplot('GrLivArea')
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

varplot('GrLivArea')
train["SalePrice"] = np.log1p(train["SalePrice"])

plt.subplots(figsize=(15, 8))

sns.distplot(train['SalePrice'] , fit=norm)
ntrain = train.shape[0]

ntest = test.shape[0]

train_ID = train['Id']

test_ID = test['Id']

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.head()
low_corr=correlations[correlations<0.2].index
low_corr=low_corr.drop('YrSold').drop('MoSold').drop('OverallCond').drop('MSSubClass') #they are categorical not numerical
low_corr
all_data=all_data.drop(columns=low_corr)
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(33)
def replacer(var, value='None', mode=False, mean=False):

    if mode ==True:

        value= all_data[var].mode()[0]

    all_data[var] = all_data[var].fillna(value)

for col in ('FireplaceQu', 'MasVnrType','MiscFeature','PoolQC','Alley','Fence','GarageType'):

    replacer(col)

for col in ('MasVnrArea', 'BsmtFullBath','GarageCars','GarageArea','BsmtUnfSF','BsmtFinSF1','TotalBsmtSF','GarageQual','GarageFinish',

            'GarageCond','GarageYrBlt','BsmtCond','BsmtExposure','BsmtQual','BsmtFinType2','BsmtFinType1'):

    replacer(col, 0)

for col in ('MSZoning', 'Utilities','Functional','Exterior1st','Exterior2nd','KitchenQual','Electrical','SaleType'):

    replacer(col, mode=True)

    

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(2)
all_data.dtypes[all_data.dtypes=='int64']
def categorize(var):

    all_data[var] = all_data[var].astype(str)
for col in ('MoSold', 'YrSold', 'MSSubClass', 'OverallCond','YearBuilt','YearRemodAdd'):

    categorize(col)
all_data.dtypes[all_data.dtypes=='int64']
cat=all_data.dtypes[all_data.dtypes=='object']

cat.index
to_label=('Functional', 'LandSlope','LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold', 'YearBuilt', 'YearRemodAdd')

quality_label=('FireplaceQu','ExterQual','ExterCond','HeatingQC','KitchenQual')
all_data[cat.index].head(5)
def encoder(var, df):

    cleanup = {var: {'None':0,'Po':1,'Fa':2, 'TA':3, 'Gd':4,'Ex':5 }}

    df.replace(cleanup, inplace=True)

for c in quality_label:

    encoder(c, all_data)

for c in to_label:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))
all_data[cat.index].head(5)
all_data.dtypes[all_data.dtypes=='int64']
numeric = all_data.dtypes[all_data.dtypes != "object"].index



skewed = all_data[numeric].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed})

skewness.head(15)
skewness=skewness.Skew[abs(skewness.Skew)>0.3]

skewness.head(10)
skewness.index
def normalization(cat):

    all_data[cat] = np.log1p(all_data[cat])
for cat in skewness.index:

    normalization(cat)
plt.subplots(figsize=(15, 8))

sns.distplot(all_data['GrLivArea'] , fit=norm)
all_data = pd.get_dummies(all_data)
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

import xgboost as xgb

import lightgbm as lgb

from sklearn.ensemble import GradientBoostingRegressor

from mlxtend.regressor import StackingRegressor
train = all_data[:ntrain]

test = all_data[ntrain:]
k = 5



def rmsle_cv(model):

    kf = KFold(n_splits=k, shuffle=True, random_state=40).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = k))

    return(rmse)
lr = LinearRegression()

lr.fit(train, y_train)
predicted_prices_linear = np.expm1(lr.predict(test))
score =rmsle_cv(lr)

print("\nLinear score:",score.mean(), score.std())
alpha=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
scores=[]

for al in alpha:

    ridge = make_pipeline(RobustScaler(), Ridge(alpha =al))

    score=rmsle_cv(ridge)

    scores.append(score.mean())

plt.plot(alpha,scores, color= 'gold')
alpha=range(8, 18)

scores=[]

for al in alpha:

    ridge = make_pipeline(RobustScaler(), Ridge(alpha =al))

    score=rmsle_cv(ridge)

    scores.append(score.mean())

plt.plot(alpha,scores, color= 'gold')

plt.plot(alpha[6], scores[6], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("alpha", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')

ridge = make_pipeline(RobustScaler(), Ridge(alpha =14, random_state=1))

score = rmsle_cv(ridge)

print("\nRidge score:",score.mean(), score.std())
ridge.fit(X=train, y=y_train)

predicted_prices_ridge_log=(ridge.predict(test))

predicted_prices_ridge=np.expm1(ridge.predict(test))
alpha=[0.0002,0.0003, 0.0004, 0.0005,0.0006, 0.0007]
scores=[]

for al in alpha:

    lasso = make_pipeline(RobustScaler(), Lasso(alpha =al, random_state=1))

    score=rmsle_cv(lasso)

    scores.append(score.mean())

plt.plot(alpha,scores, color= 'gold')

plt.plot(alpha[3], scores[3], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("alpha", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.00050, random_state=1))
score = rmsle_cv(lasso)

print("\nLasso score:",score.mean(), score.std())
lasso.fit(X=train, y=y_train)
predicted_prices_lasso_log=lasso.predict(test)

predicted_prices_lasso=np.expm1(lasso.predict(test))
alpha=[0.4,0.5,0.6,0.7,0.75, 0.8, 1]

scores=[]

for al in alpha:

    KRR = KernelRidge(alpha=al, kernel='polynomial', degree=2, coef0=2.5)

    score=rmsle_cv(KRR)

    scores.append(score.mean())

plt.plot(alpha,scores, color= 'gold')

plt.plot(alpha[4], scores[4], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("alpha", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')
coef=[0,1,2,2.5,3,4]

scores=[]

for cof in coef:

    KRR = KernelRidge(alpha=0.75, kernel='polynomial', degree=2, coef0=cof)

    score=rmsle_cv(KRR)

    scores.append(score.mean())

plt.plot(coef,scores, color= 'gold')

plt.plot(coef[3], scores[3], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("coef", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')
KRR=KernelRidge(alpha=0.75, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)

print("\nKRR score:",score.mean(), score.std())
KRR.fit(X=train, y=y_train)

predicted_prices_KRR_log=KRR.predict(test)

predicted_prices_KRR=np.expm1(KRR.predict(test))
alpha=[0.0003, 0.0004, 0.0005,0.0006, 0.0007, 0.0008]
scores=[]

for al in alpha:

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=al, l1_ratio=.9, random_state=3))

    score=rmsle_cv(ENet)

    scores.append(score.mean())

plt.plot(alpha,scores, color= 'gold')

plt.plot(alpha[3], scores[3], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("alpha", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')

rateos=[0, 0.1, 0.3, 0.5, 0.7, 0.9,1]

scores=[]

for rat in rateos:

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0006, l1_ratio=rat, random_state=3))

    score=rmsle_cv(ENet)

    scores.append(score.mean())

plt.plot(rateos,scores, color= 'gold')

plt.plot(rateos[5], scores[5], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("alpha", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0006, l1_ratio=0.9, random_state=3))
score = rmsle_cv(ENet)

print("\nEnet score:",score.mean(), score.std())
ENet.fit(X=train, y=y_train)
predicted_prices_Enet_log=ENet.predict(test)

predicted_prices_Enet=np.expm1(ENet.predict(test))
n_estim=[10, 100, 1000, 2000]

scores=[]

for n in n_estim:

    GBoost = GradientBoostingRegressor(n_estimators=n, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

    score=rmsle_cv(GBoost)

    scores.append(score.mean())

plt.plot(n_estim,scores, color= 'gold')

plt.plot(n_estim[2], scores[2], '*', ms=30, mec='darkred', mfc='none', mew=2)

plt.xlabel("n_estim", fontsize= 25, color='darkred')

plt.ylabel("score", fontsize= 25, color='darkred')

GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,

                                   max_depth=3, max_features='sqrt',

                                   min_samples_leaf=10, min_samples_split=10, 

                                   loss='huber', random_state =5)



score = rmsle_cv(GBoost)

print("\nGBoost score:",score.mean(), score.std())
GBoost.fit(X=train, y=y_train)

predicted_prices_GBoost_log=GBoost.predict(test)

predicted_prices_GBoost=np.expm1(GBoost.predict(test))

XGB=xgb.XGBRegressor(colsample_bytree=0.45, gamma=0.05, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.8, n_estimators=1000,

                             reg_alpha=0.47, reg_lambda=0.86,

                             subsample=0.5, silent=1,

                             random_state =7, nthread = -1)
score = rmsle_cv(XGB)

print("\nxgb score:",score.mean(), score.std())
XGB.fit(X=train, y=y_train)

predicted_prices_xgb_log=XGB.predict(test)

predicted_prices_xgb=np.expm1(XGB.predict(test))

LGB = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.23,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(LGB)

print("\nlgb score:",score.mean(), score.std())
LGB.fit(X=train, y=y_train)

predicted_prices_lgb_log=LGB.predict(test)

predicted_prices_lgb=np.expm1(LGB.predict(test))


stregr = StackingRegressor(regressors=[ENet, GBoost, XGB], 

                           meta_regressor=lasso)
score = rmsle_cv(stregr)

print("\nstregr score:",score.mean(), score.std())
stregr.fit(X=train, y=y_train)
predicted_prices_stregr_log=stregr.predict(test)

predicted_prices_stregr=np.expm1(stregr.predict(test))
predictions=pd.DataFrame()
for pred in ['Enet','GBoost', 'KRR', 'lasso','lgb','linear', 'ridge', 'stregr', 'xgb']:

    predictions[pred]=eval('predicted_prices_'+pred)
predictions.head()
def rmsle(model):

    model=eval(model)

    y_pred = model.predict(train)

    return np.sqrt(mean_squared_error(y_train, y_pred))

scores=pd.DataFrame(index=['score mean','score std', 'rmsle train'])



for model in ['ENet', 'KRR', 'lasso','lr', 'ridge', 'stregr','XGB','LGB','GBoost' ]:

    score=rmsle_cv(eval(model))

    rmsle_train=rmsle(model)

    scores[model] = (score.mean(), score.std(), rmsle_train) 
scores
predicted_prices=predicted_prices_lasso*0.43+predicted_prices_KRR*0.14+predicted_prices_stregr*0.10+predicted_prices_lgb*0.33
lower=np.percentile(a=predicted_prices,q=0.9)

upper=np.percentile(a=predicted_prices,q=99.2)

plt.subplots(figsize=(15, 8))

sns.distplot(predicted_prices)

plt.axvline(lower, 0,1, color='orange')

plt.axvline(upper, 0,1, color='orange')
predicted_prices[predicted_prices<lower]=predicted_prices[predicted_prices<lower]*0.81
predicted_prices[predicted_prices>upper]=predicted_prices[predicted_prices>upper]*1.08
lower2=np.percentile(a=predicted_prices,q=0.03) 

upper2=np.percentile(a=predicted_prices,q=99.7)

predicted_prices[predicted_prices<lower2]=predicted_prices[predicted_prices<lower2]*0.97

predicted_prices[predicted_prices>upper2]=predicted_prices[predicted_prices>upper2]*1.03
my_submission = pd.DataFrame({'Id': test_ID, 'SalePrice': predicted_prices})

my_submission.to_csv('submission.csv', index=False)