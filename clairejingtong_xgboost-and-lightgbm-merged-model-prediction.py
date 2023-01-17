import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sns.distplot(train['SalePrice'])
corrmat = train.corr()
f,ax = plt.subplots(figsize=(16,8))
sns.heatmap(corrmat,vmax=0.8, square=True)  
k = 10 # number of variables for matrix
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,annot=True,square=True,fmt='.2f'
                 ,annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)

sns.set()
cols =['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train[cols],size=2.5)
total = train.isnull().sum().sort_values(ascending=False)
percent =(train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)
train = train.drop((missing_data[missing_data['Total']>1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max()
var = 'GrLivArea'
data = pd.DataFrame(train[var],columns=[var]).join(train['SalePrice'])
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 523].index)
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'],train[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
# histogram and normal probability plot
sns.distplot(train['SalePrice'],fit =norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
# applying log transformation
train['SalePrice_log']= np.log(train['SalePrice'])
sns.distplot(train['SalePrice_log'],fit=norm)
fig=plt.figure()
res=stats.probplot(train['SalePrice_log'],plot =plt)
sns.distplot(train['GrLivArea'], fit=norm);
plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
train['GrLivArea_log'] = np.log(train['GrLivArea'])

sns.distplot(train['GrLivArea_log'],fit = norm)
plt.figure()
res = stats.probplot(train['GrLivArea_log'],plot=plt)
sns.distplot(train['TotalBsmtSF'],fit=norm)
plt.figure()
res = stats.probplot(train['TotalBsmtSF'],plot = plt)
train['TotalBsmtSF_log'] = np.log(train['TotalBsmtSF']+1)


sns.distplot(train[train['TotalBsmtSF_log']> 0]['TotalBsmtSF_log'],fit = norm)
plt.figure()
res = stats.probplot(train[train['TotalBsmtSF_log']>0]['TotalBsmtSF'],plot=plt)
train
X = pd.DataFrame(train,columns=['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt'])
y = train['SalePrice_log'].copy()
y
y.isnull().sum()
y.shape
X_test_Id = test[['Id','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']]
print('X_train shape:',X.shape)
print('X_test shape:',X_test_Id.shape)
X_test_Id.isnull().sum()
X_test_Id['GarageCars'] = X_test_Id['GarageCars'].fillna(np.mean(X_test_Id['GarageCars']))
X_test_Id['TotalBsmtSF'] = X_test_Id['TotalBsmtSF'].fillna(np.mean(X_test_Id['TotalBsmtSF']))
X_test = X_test_Id.iloc[:,1:]
X_test.isnull().sum()
x_train,x_val,y_train,y_val = train_test_split(X,y,test_size=0.3)
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150,learning_rate = 0.1, ganmma=0,subsample=0.8,\
                            colsample_bytress=0.9,max_depth=7)
    model.fit(x_train,y_train)
    return model
def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators=150)
    param_grid ={
        'learning_rate':[0.01,0.05,0.1,0.2]
    }
    gbm = GridSearchCV(estimator,param_grid)
    gbm.fit(x_train,y_train)
    return gbm
print('Train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(y_val,val_lgb)
print('MAE of val with lgb:', MAE_lgb)

print('Predict lgb...')
model_lgb_pre = build_model_lgb(X,y)
subA_lgb = model_lgb_pre.predict(X_test)

print('Train xgb...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(y_val,val_xgb)
print('MAE of val with xgb:',MAE_xgb)

print('Predict xgb...')
model_xgb_pre = build_model_xgb(X,y)
subA_xgb = model_xgb_pre.predict(X_test)
plt.scatter(x_val.index, y_val, color='black')
plt.scatter(x_val.index, val_xgb, color='blue')
plt.xlabel('x')
plt.ylabel('SalePrice_log')
plt.legend(['True Price','Predicted Price'],loc='upper right')
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb
val_Weighted[val_Weighted<0]=10
print('MAE of val with Weighted ensemble:',mean_absolute_error(y_val,val_Weighted))
sub_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*subA_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*subA_xgb

sub = pd.DataFrame()
sub['Id'] = X_test_Id['Id']
sub['SalePrice'] = np.exp(sub_Weighted)
sub.to_csv('./sub_Weighted.csv',index=False)
sub
