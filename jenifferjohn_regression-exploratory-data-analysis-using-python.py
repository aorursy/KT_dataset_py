# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import datetime
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import lightgbm 
import sklearn.metrics as metrics
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)

os.getcwd()
traindf=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
testdf=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
traindf.head()
testdf.head()
traindf.shape
testdf.shape
traindf.info()
testdf.info()
sns.distplot(traindf['SalePrice'])
plt.xticks(rotation=90)
for i in traindf.select_dtypes(exclude='object').columns:
    sns.scatterplot(x=traindf[i],y=traindf['SalePrice'])
    plt.show()
for i in traindf.select_dtypes(include='object').columns:
    sns.boxplot(x=traindf[i],y=traindf['SalePrice'])
    plt.xticks(rotation=90)
    plt.show()
data=pd.concat([traindf,testdf],axis=0,sort=False,ignore_index=False)
data.shape
data.info()
mv = data.isnull().sum().sort_values(ascending=False)
mp = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_val = pd.concat([mv, mp], axis=1)
missing_val.columns=['value', 'percentage']
missing_val[missing_val['value']>0]

categ_cols=['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','SaleType']

for col in categ_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)
    data[col].replace(0,data[col].mode()[0], inplace=True)
    print(col,data[col].mode()[0])
print('Done')

spl_cols=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

for col in spl_cols:
    data[col] = data[col].fillna('None')
print('Done')
year_cols=['GarageYrBlt']

for col in year_cols:
    data[col].fillna(data[col].mode()[0],inplace=True)
print('Done')

#If no gararge, then garage age will be zero

numeric_cols=['LotFrontage','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageCars','GarageArea']

for col in numeric_cols:
    data[col] = data[col].replace(0,data[col].median())
    data[col] = data[col].replace(np.nan,data[col].median())
    print(col,data[col].median())

print('Done')
mv = data.isnull().sum().sort_values(ascending=False)
mp = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_val = pd.concat([mv, mp], axis=1)
missing_val.columns=['value', 'percentage']
missing_val[missing_val['value']!=0]
data.drop(['Id'],axis=1,inplace=True)
data['GarageYrBlt']=data['GarageYrBlt'].apply(int) #converting year to int before converting it into object, just to avoid '.O'

#To be converted to object:
obj_cols =['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']
for i in obj_cols:
    data[i]= data[i].astype('object')
    
print('done')
data.describe().T
data.describe(include='object').T
data.select_dtypes(exclude='object').columns
for col in data.select_dtypes(exclude='object').columns:
    sns.boxplot(data[col])
    plt.show()
X= data.drop(['SalePrice'],axis=1)
y= data['SalePrice']
X.shape,y.shape
X['TotalSF']=X['TotalBsmtSF']+ X['1stFlrSF']+ X['2ndFlrSF']

X['TotalBath'] = (X['FullBath'] + (0.5 * X['HalfBath']) + X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath']))

X['TotalPorchSF'] = X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch']

X['PropertyAge']=X['YrSold'].astype(float) - X ['YearBuilt'].astype(float)

X['GarageAge']=X['YrSold'].astype(float) - X['GarageYrBlt'].astype(float)

X.shape
ls_numeric =list(X.select_dtypes(exclude='object').columns)
for i in ls_numeric:
    if i=='PropertyAge':
        ls_numeric.remove(i)
#applying log(x+1) transformation to handle skewness of the data, transformation will normalize the data

for i in ls_numeric :
    if(abs(X[i].skew())>0.5):
        X[i]= X[i].apply( lambda x: np.log(x+1))
print('done')
X.describe().T
#Before creating dummies,we are removing columns that do not add value to our computation
X= X.drop(['Street','MoSold','YearBuilt','GarageYrBlt','YearRemodAdd','YrSold'],axis=1)
X.shape
categ_cols= X.select_dtypes(include='object').columns
print(len(categ_cols))
dummies=pd.get_dummies(X[categ_cols])
dummies.shape
X= pd.concat([X,dummies],axis=1)
X.drop(categ_cols,axis=1,inplace=True)
X.shape
X_train=X[ :(traindf.shape[0])]
X_test=X[traindf.shape[0]:]
y_train=y[:(traindf.shape[0])]
y_test=y[traindf.shape[0]:]
X_train.shape,X_test.shape,y_train.shape,y_test.shape
ss=RobustScaler()

Xs= ss.fit_transform(X)

X_trains = ss.fit_transform(X_train) 
#here we are fitting and transforming because we make the model learn our data while fittig
X_tests = ss.transform(X_test) 
#test data should not be learnt by machine, we use only for predicting,so we just transform
X_trains = pd.DataFrame(X_trains,columns= X_train.columns)
X_tests = pd.DataFrame(X_tests,columns= X_test.columns) 
lasso = LassoCV()
lasso.fit(X_trains,y_train)
print("Best alpha using built-in LassoCV: %f" % lasso.alpha_)
print("Best score using built-in LassoCV: %f" %lasso.score(X_trains,y_train))
coef = pd.Series(lasso.coef_, index = X_trains.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 100.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
lis=imp_coef[imp_coef!=0].index
lis
X_trainss = X_trains[lis]
X_testss = X_tests[lis]
X_trainss.shape,X_testss.shape
X_trains.shape,X_tests.shape
# Let's perform a cross-validation to find the best combination of alpha and l1_ratio
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import r2_score

cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 
                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 
                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')

cv_model.fit(X_trainss, y_train)

print('Optimal alpha: %.8f'%cv_model.alpha_)
print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
print('Number of iterations %d'%cv_model.n_iter_)

def rmsqe(y, y_pred):
    return np.sqrt(metrics.mean_squared_error(y,y_pred))
print('Lasso r2 score:',lasso.score(X_trains,y_train))
lasso_train_pred = lasso.predict(X_trains)
lasso_test_pred = lasso.predict(X_tests)


# train model with best parameters from CV
enet = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)
enet.fit(X_trainss, y_train)
print('Elastic-Net r2_score:',r2_score(y_train, enet.predict(X_trainss))) 
elastic_train_pred = enet.predict(X_trainss)
elastic_test_pred = enet.predict(X_testss)

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10,loss='huber', random_state =5)
gbr.fit(X_trainss, y_train)
print('GradientBoostingRegressor r2_score:',r2_score(y_train, gbr.predict(X_trainss))) 
gbr_train_pred = gbr.predict(X_trainss)
gbr_test_pred = gbr.predict(X_testss)

lgb=lightgbm.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split')
lgb.fit(X_trains, y_train)
print('LGBMRegressor r2_score:',r2_score(y_train, lgb.predict(X_trains))) 
lgb_train_pred = lgb.predict(X_trains)
lgb_test_pred = lgb.predict(X_tests)

sub = pd.DataFrame()
sub['Id'] = testdf['Id']
sub['SalePrice'] = lgb_test_pred
sub.to_csv('submission.csv',index=False)