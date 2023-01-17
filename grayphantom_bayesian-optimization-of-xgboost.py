
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings

import warnings
warnings.filterwarnings("ignore")


dtrain=pd.read_csv('../input/train.csv')
dtest=pd.read_csv('../input/test.csv')
print(dtrain.shape)

print(dtest.shape)
dtrain.head()
dtest.head()
test_ID=dtest['Id']
dtrain=dtrain.drop(['Id'],axis=1)
dtest=dtest.drop(['Id'],axis=1)
y_train=dtrain['SalePrice']
data=pd.concat([dtrain,dtest],ignore_index=True)
data=data.drop(['SalePrice'],axis=1)
data.shape
plt.figure(figsize=(7,6))
sns.distplot(y_train)
plt.show()
y_train=np.log1p(y_train)
sns.distplot(y_train)
plt.show()
total = data.isnull().sum().sort_values(ascending=False)
percent = ((data.isnull().sum()/data.isnull().count()).sort_values(ascending=False))*100
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

data['PoolQC']=data['PoolQC'].fillna("None")
data['MiscFeature']=data['MiscFeature'].fillna("None")
data['Alley']=data['Alley'].fillna("None")
data['Fence']=data['Fence'].fillna("None")
data['FireplaceQu']=data['FireplaceQu'].fillna("None")
data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    data[col] = data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    data[col] = data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    data[col] = data[col].fillna('None')
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
data=data.fillna(data.mean())
data.shape
corr=dtrain.corr()['SalePrice'].sort_values()[::-1]
c=corr.head(15).index
plt.figure(figsize=(12,8))
sns.heatmap(dtrain[c].corr(),annot=True)
data=data.drop(['GarageCars','1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageYrBlt'],axis=1)
num_f=data.dtypes[data.dtypes!=object].index
skew_f=data[num_f].apply(lambda x: skew(x.dropna()))
skew_f=skew_f[skew_f>0.75] #include only those features that have skewness greater than 75%
skew_f=skew_f.index
#Apply log transformation
data[skew_f]=np.log1p(data[skew_f])
data = pd.get_dummies(data)# for handling categorical data
data.shape
x_train=data[:dtrain.shape[0]]
test=data[dtrain.shape[0]:]
#from sklearn.linear_model import Lasso,Ridge,RidgeCV,LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
dtrain=xgb.DMatrix(x_train,label=y_train)
dtest=xgb.DMatrix(test)
def_params={'min_child_weight': 1,
        'max_depth': 6,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_lambda': 1,
        'reg_alpha': 0,
        'learning-rate':0.3,
        'silent':1 # not a hyperparameter ,used to silence XGBoost
        }
cv_res=xgb.cv(def_params,dtrain,nfold=5)
cv_res.tail()
from skopt import BayesSearchCV 
import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

params={'min_child_weight': (0, 50,),
        'max_depth': (0, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'reg_lambda':(1e-5,100,'log-uniform'),
        'reg_alpha':(1e-5,100,'log-uniform'),
        'learning-rate':(0.01,0.2,'log-uniform')
        }

bayes=BayesSearchCV(xgb.XGBRegressor(),params,n_iter=10,scoring='neg_mean_squared_error',cv=5,random_state=42)
res=bayes.fit(x_train,y_train)
print(res.best_params_)
final_params={'colsample_bytree': 0.50, 'max_depth': 7, 'min_child_weight':13, 'reg_alpha': 0.112, 'reg_lambda': 0.0008, 'subsample': 0.65,'eta':0.11,'silent':1}
cv_res=xgb.cv(final_params,dtrain,num_boost_round=1000,early_stopping_rounds=100,nfold=5)
cv_res.loc[30:,['train-rmse-mean','test-rmse-mean']].plot()
cv_res.tail()
model_xgb=xgb.train(final_params,dtrain=dtrain,num_boost_round=151)
final_pred=np.expm1(model_xgb.predict(dtest))
s=pd.DataFrame()
s['Id']=test_ID
s['SalePrice']=final_pred
s.to_csv('my_submission.csv',index=False)