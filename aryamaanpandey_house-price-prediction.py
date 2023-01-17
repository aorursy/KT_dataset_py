# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV,KFold
import xgboost as xgb
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.info()
trainID = train['Id']
testID = test['Id']
train.drop('Id',axis = 1,inplace = True)
test.drop('Id',axis = 1,inplace = True)
corr = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Reds", square=True)
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig,ax = plt.subplots()
fig = plt.scatter(x = train['GrLivArea'],y = train['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()
fig,ax = plt.subplots()
fig = plt.scatter(x = train['TotalBsmtSF'],y = train['SalePrice'])
plt.xlabel('Total Basement SF')
plt.ylabel('Sale Price')
plt.show()
fig,ax = plt.subplots()
fig = ax.scatter(x = train['1stFlrSF'],y = train['SalePrice'])
plt.xlabel('1st Floor SF')
plt.ylabel('Sale Price')
plt.show()
data = pd.concat([train['SalePrice'],train['GarageCars']],axis=1)
fig,ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x = train['GarageCars'],y='SalePrice',data=data)
fig,ax = plt.subplots()
fig = plt.scatter(x = train['GarageArea'],y = train['SalePrice'])
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.show()
data = pd.concat([train['SalePrice'],train['FullBath']],axis = 1)
fig,ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x = train['FullBath'],y = 'SalePrice',data=data)
data = pd.concat([train['SalePrice'],train['TotRmsAbvGrd']],axis=1)
fig,ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x = train['TotRmsAbvGrd'],y = 'SalePrice',data=data)
from collections import Counter
cnt = Counter(train['TotRmsAbvGrd'])
print(cnt)
#Applying Tukey Method
q1 = np.percentile(train['GrLivArea'],25)
q2 = np.percentile(train['GrLivArea'],50)
q3 = np.percentile(train['GrLivArea'],75)
iqr = q3 - q1
idx_dropped = []
for i in range(len(train['GrLivArea'])):
    if(train['GrLivArea'][i] < q1-1.5*iqr):
        idx_dropped.append(i)
    if(train['GrLivArea'][i] > q3 + 1.5*iqr):
        idx_dropped.append(i)
for i in idx_dropped:
    train = train.drop([i])
train = train.reset_index(drop=True)
#Applying Tukey Method
q1 = np.percentile(train['TotalBsmtSF'],25)
q2 = np.percentile(train['TotalBsmtSF'],50)
q3 = np.percentile(train['TotalBsmtSF'],75)
iqr = q3 - q1
idx_dropped = []
for i in range(len(train['TotalBsmtSF'])):
    if(train['TotalBsmtSF'][i] < q1-1.5*iqr):
        idx_dropped.append(i)
    if(train['TotalBsmtSF'][i] > q3 + 1.5*iqr):
        idx_dropped.append(i)
for i in idx_dropped:
    train = train.drop([i])
train = train.reset_index(drop=True)
#Applying Tukey Method
q1 = np.percentile(train['1stFlrSF'],25)
q2 = np.percentile(train['1stFlrSF'],50)
q3 = np.percentile(train['1stFlrSF'],75)
iqr = q3 - q1
idx_dropped = []
for i in range(len(train['1stFlrSF'])):
    if(train['1stFlrSF'][i] < q1-1.5*iqr):
        idx_dropped.append(i)
    if(train['1stFlrSF'][i] > q3 + 1.5*iqr):
        idx_dropped.append(i)
for i in idx_dropped:
    train = train.drop([i])
train = train.reset_index(drop=True)
#Applying Tukey Method
q1 = np.percentile(train['GarageArea'],25)
q2 = np.percentile(train['GarageArea'],50)
q3 = np.percentile(train['GarageArea'],75)
iqr = q3 - q1
idx_dropped = []
for i in range(len(train['GarageArea'])):
    if(train['GarageArea'][i] < q1-1.5*iqr):
        idx_dropped.append(i)
    if(train['GarageArea'][i] > q3 + 1.5*iqr):
        idx_dropped.append(i)
for i in idx_dropped:
    train = train.drop([i])
train = train.reset_index(drop=True)
#GrLivArea
fig,ax = plt.subplots()
fig = plt.scatter(x = train['GrLivArea'],y = train['SalePrice'])
plt.ylabel('Sale Price')
plt.xlabel('GrLivArea')
plt.show()
#Garage Area
fig,ax = plt.subplots()
fig = plt.scatter(x = train['GarageArea'],y = train['SalePrice'])
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.show()
#1stFloorSF
fig,ax = plt.subplots()
fig = ax.scatter(x = train['1stFlrSF'],y = train['SalePrice'])
plt.xlabel('1st Floor SF')
plt.ylabel('Sale Price')
plt.show()
#TotalBsmtSF
fig,ax = plt.subplots()
fig = plt.scatter(x = train['TotalBsmtSF'],y = train['SalePrice'])
plt.xlabel('Total Basement SF')
plt.ylabel('Sale Price')
plt.show()
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print("Mean = " + str(mu))
print("SD = " + str(sigma))
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Probability Plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print("Mean = " + str(mu))
print("SD = " + str(sigma))
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Probability Plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
y = train['SalePrice']
train = train.drop(['SalePrice'],axis = 1)
dataset = pd.concat([train,test]).reset_index(drop=True)
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(dataset.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].median())
dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())
dataset['GarageCars'] = dataset['GarageCars'].fillna(2.0)
dataset['KitchenQual'] = dataset['KitchenQual'].fillna('TA')
dataset['Electrical'] = dataset['Electrical'].fillna('SBrkr')
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].median())
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].median())
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean())
dataset['SaleType'] = dataset['SaleType'].fillna('WD')
dataset['Exterior1st'] = dataset['Exterior1st'].fillna('VinylSd')
dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna('VinylSd')
dataset['Functional'] = dataset['Functional'].fillna('Typ')
dataset['Utilities'] = dataset['Utilities'].fillna('AllPub')
dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0.0)
dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0.0)
dataset['MSZoning'] = dataset['MSZoning'].fillna('RL')
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0.0)
dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna('Unf')
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna('Unf')
dataset['BsmtQual'] = dataset['BsmtQual'].fillna('TA')
dataset['BsmtCond'] = dataset['BsmtCond'].fillna('TA')
dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna('No')
dataset['GarageType'] = dataset['GarageType'].fillna('Attchd')
dataset['GarageCond'] = dataset['GarageCond'].fillna('TA')
dataset['GarageQual'] = dataset['GarageQual'].fillna('TA')
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].median())
dataset['GarageFinish'] = dataset['GarageFinish'].fillna('Unf')
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].median())
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('Gd')
dataset['Fence'] = dataset['Fence'].fillna('MnPrv')
dataset['Alley'] = dataset['Alley'].fillna('Grvl')
dataset['MiscFeature'] = dataset['MiscFeature'].fillna('Shed')
dataset['PoolQC'] = dataset['PoolQC'].fillna('Gd')
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in dataset.columns:
    if dataset[i].dtype in numeric_dtypes:
        numeric.append(i)
skew_features = dataset[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)
for i in skew_index:
    dataset[i] = boxcox1p(dataset[i], boxcox_normmax(dataset[i] + 1))
dataset = pd.get_dummies(dataset).reset_index(drop=True)
dataset.shape
X = dataset[:len(train)]
test_X = dataset[len(train):].reset_index(drop=True)
xgb = xgb.XGBRegressor(learning_rate=0.2,
                       n_estimators=4000,
                       max_depth=4,
                       min_child_weight=1,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)
#kf = KFold(n_splits = 12,random_state = 42,shuffle = True)
#elastic=ElasticNet(normalize=True)
#search=GridSearchCV(estimator=elastic,param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},scoring='neg_mean_squared_error',n_jobs=1,refit=True,cv=kf)
#search.fit(X,y)
#print(search.best_params_)
#print(search.best_score_)
elastic=ElasticNet(normalize=True,alpha = 0.0001,l1_ratio = 0.8)
#kf = KFold(n_splits = 12,random_state = 42,shuffle = True)
#pipeline = Pipeline([('scale', RobustScaler()),('model',SVR())])
#param_grid = {'model__C': [0.1, 1, 10, 100, 1000],  
#              'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#              'model__epsilon':[0.001,0.01,0.1]}
#search = GridSearchCV(pipeline,param_grid = param_grid,scoring = 'neg_mean_squared_error',verbose = 1,cv = kf,n_jobs=100)
#search.fit(X,y)
#(search.best_params_)
#print(search.best_score_)
svr = make_pipeline(RobustScaler(),SVR(C=100,epsilon = 0.01,gamma = 0.0001))
#kf = KFold(n_splits = 12,random_state = 42,shuffle = True)
#rf = RandomForestRegressor()
#param_grid = {'max_depth': [10,20],
# 'min_samples_leaf': [4,8],
# 'min_samples_split': [4,8],
# 'n_estimators': [1400, 1600]}
#search = GridSearchCV(estimator=rf,param_grid = param_grid,scoring = 'neg_mean_squared_error',verbose = 1,cv = kf,n_jobs=100)
#search.fit(X,y)
#print(search.best_params_)
#print(search.best_score_)
rf = RandomForestRegressor(n_estimators=1400,
                          max_depth=20,
                          min_samples_split=4,
                          min_samples_leaf=4,
                          max_features=None,
                          oob_score=True,
                          random_state=42)
#kf = KFold(n_splits = 12,random_state = 42,shuffle = True)
lgb = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)
sreg = StackingCVRegressor(regressors=(xgb,lgb,svr,elastic,rf),meta_regressor=xgb,use_features_in_secondary=True)
kf = KFold(n_splits = 12,random_state = 42,shuffle = True)
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
#score = cv_rmse(xgb)
#print("xgb: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#score = cv_rmse(lgb)
#print("lgb: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#score = cv_rmse(elastic)
#print("ElasticNet: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#score = cv_rmse(rf)
#print("Random Forest: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#score = cv_rmse(svr)
#print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#score = np.sqrt(-cross_val_score(sreg,np.array(X),np.array(y),scoring="neg_mean_squared_error",cv=kf))
#print("Stacked Regressor: {:.4f} ({:.4f})".format(score.mean(), score.std()))
sreg_model = sreg.fit(np.array(X), np.array(y))
lgb_model = lgb.fit(X, y)
xgb_model = xgb.fit(X, y)
svr_model = svr.fit(X, y)
elastic_model = elastic.fit(X, y)
rf_model = rf.fit(X, y)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
def blended_predictions(X):
    return ((0.2 * elastic_model.predict(X)) + \
            (0.15 * svr_model.predict(X)) + \
            (0.05 * xgb_model.predict(X)) + \
            (0.2 * lgb_model.predict(X)) + \
            (0.05 * rf_model.predict(X)) + \
            (0.35 * sreg_model.predict(np.array(X))))
blended_score = rmsle(y, blended_predictions(X))
print('RMSLE score on train data:')
print(blended_score)
ans = pd.DataFrame()
ans['Id'] = testID
ans['SalePrice'] = np.floor(np.expm1(blended_predictions(test_X)))
ans.to_csv("lowesterror.csv",index=False)
