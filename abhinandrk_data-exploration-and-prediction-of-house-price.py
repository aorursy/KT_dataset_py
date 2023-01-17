import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats
#os.chdir("/home/rk/Desktop/kaggle/House price/all")
trainSample=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
testSample=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train=pd.concat(objs=[trainSample,testSample],axis=0).reset_index(drop=True)
y_train=trainSample[['Id','SalePrice']]
train=train.drop(["SalePrice"],axis=1)




#analysing 'SalePrice'
sns.distplot(trainSample['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % trainSample['SalePrice'].skew())
print("Kurtosis: %f" % trainSample['SalePrice'].kurt())
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([trainSample['SalePrice'], trainSample[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([trainSample['SalePrice'], trainSample[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([trainSample['SalePrice'], trainSample[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
data = pd.concat([trainSample['SalePrice'], trainSample[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#correlation matrix
corrmat = trainSample.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(trainSample[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(trainSample[cols], size = 2.5)
plt.show();
#handling missing value
missing_val=pd.DataFrame(train.isnull().sum())
missing_val=missing_val.reset_index()

missing_val=missing_val.rename(columns={'index':'variables',0:'missing_percentage'})
missing_val['missing_percentage']=((missing_val['missing_percentage'])/len(train))*100
missing_val.sort_values(by='missing_percentage',ascending=False)
#missing data ......This method is better..thanks to pedro:)
total = train.isnull().sum().sort_values(ascending=False)
percent = ((train.isnull().sum()/train.isnull().count())*100).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
#train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(trainSample['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([trainSample['SalePrice'], trainSample[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)
#Delete points from target variables also ....That makes sense ,right? :)
y_train=y_train.drop(y_train[y_train['Id']==1299].index)
y_train=y_train.drop(y_train[y_train['Id']==524].index)

#now we don't need 'Id' column in target variable so drop that column
y_train=y_train[['SalePrice']]
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([trainSample['SalePrice'], trainSample[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot
sns.distplot(trainSample['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSample['SalePrice'], plot=plt)
#applying log transformation......becoz graph shows postive skewnss .
#if it is positive skewnwss we can apply log transform :)
y_train = np.log(y_train)
#transformed histogram and normal probability plot
sns.distplot(trainSample['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSample['SalePrice'], plot=plt)
#histogram and normal probability plot
sns.distplot(trainSample['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSample['GrLivArea'], plot=plt)
#data transformation ....same postive skewnwss :::)))
train['GrLivArea'] = np.log(train['GrLivArea'])
#transformed histogram and normal probability plot
sns.distplot(trainSample['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSample['GrLivArea'], plot=plt)
#next please :)
#histogram and normal probability plot
sns.distplot(trainSample['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSample['TotalBsmtSF'], plot=plt)
#look above graph looks like house without basement.in this case we cant apply log..This is big boss of probelms:)
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
y_train.shape[0]
#histogram and normal probability plot
sns.distplot(trainSample[trainSample['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(trainSample[trainSample['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot
plt.scatter(trainSample['GrLivArea'], trainSample['SalePrice']);
#scatter plot
plt.scatter(trainSample[trainSample['TotalBsmtSF']>0]['TotalBsmtSF'], trainSample[trainSample['TotalBsmtSF']>0]['SalePrice']);
train.head(10)
#convert categorical variable into dummy

train = pd.get_dummies(train)
train.head(10)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


test = train[1458:]
trainnew = train[:1458]

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(trainnew.values)
    rmse= np.sqrt(-cross_val_score(model, trainnew.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
#################################################################################
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
###############################################################################
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
###################################################33333
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_xgb.fit(trainnew, y_train)
xgb_train_pred = model_xgb.predict(trainnew)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
xgb_pred = np.expm1(model_xgb.predict(test))
ans=pd.DataFrame({'id':range(1461,1461+len(xgb_pred)),
                  'SalePrice':xgb_pred
                 })

#ansnew=ans[['id','SalePrice']]
#ansnew.to_csv("try.csv",index=False)
#ans.to_csv("FINAL.csv",index=False)
#try all the model available.....Learning never ends :)
GBoost.fit(trainnew, y_train)
GB_train_pred = GBoost.predict(trainnew)
print(rmsle(y_train, GB_train_pred))
#thanks for reading my note book.....