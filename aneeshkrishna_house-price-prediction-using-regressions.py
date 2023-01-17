import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.metrics import r2_score,mean_squared_error,make_scorer

import seaborn as sns

from scipy.stats import skew,norm
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape,test.shape)
plt.scatter(x='GrLivArea',y='SalePrice',data=train)

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')
train['GrLivArea'].sort_values(ascending=False)[:2]
train=train[train['GrLivArea']<4670]

train.shape
train_id=train['Id']

test_id=test['Id']

train.drop('Id',axis=1,inplace=True)

test.drop('Id',axis=1,inplace=True)

print(train.shape,test.shape)
sns.distplot(train['SalePrice'],fit=norm)

plt.title('saleprice distribution')

plt.ylabel('frequency')

print(train['SalePrice'].skew())
train['SalePrice']=np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'],fit=norm)

plt.title('saleprice distribution')

plt.ylabel('frequency')

print(train['SalePrice'].skew())
all_data=pd.concat((train,test)).reset_index(drop=True)

all_data.drop('SalePrice',axis=1,inplace=True)

all_data.shape
y=train['SalePrice']
all_data.isnull().sum().sort_values(ascending=False)[:35]
all_data['GarageCars'].fillna(0,inplace=True)

all_data['GarageArea'].fillna(0,inplace=True)

all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0],inplace=True)

all_data['Electrical'].fillna(all_data['Electrical'].mode()[0],inplace=True)

all_data['SaleType'].fillna(all_data['SaleType'].mode()[0],inplace=True)

all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0],inplace=True)

all_data['TotalBsmtSF'].fillna(0,inplace=True)

all_data['Exterior2nd'].fillna('NA',inplace=True)

all_data['BsmtFinSF1'].fillna(0,inplace=True)

all_data['BsmtFinSF2'].fillna(0,inplace=True)

all_data['BsmtUnfSF'].fillna(0,inplace=True)

all_data['BsmtFullBath'].fillna('NA',inplace=True)

all_data['Functional'].fillna(all_data['Functional'].mode()[0],inplace=True)

all_data['Utilities'].fillna(all_data['Utilities'].mode()[0],inplace=True)

all_data['BsmtHalfBath'].fillna('NA',inplace=True)

all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0],inplace=True)

all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].mean(),inplace=True)

all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0],inplace=True)

all_data.isnull().sum().sort_values(ascending=False)[:20]
all_data['BsmtFinType2'].fillna('NA',inplace=True)

all_data['BsmtCond'].fillna('NA',inplace=True)

all_data['BsmtExposure'].fillna('NA',inplace=True)

all_data['GarageType'].fillna('NA',inplace=True)

all_data['GarageQual'].fillna('NA',inplace=True)

all_data['GarageCond'].fillna('NA',inplace=True)

all_data['GarageFinish'].fillna('NA',inplace=True)

all_data.isnull().sum().sort_values(ascending=False)[:10]


all_data.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)

all_data.isnull().sum().sort_values(ascending=False)[:5]
corr_data=all_data.corr()

corr_data['LotFrontage'].sort_values(ascending=False)[:6]
sns.scatterplot(x='LotFrontage',y='LotArea',data=all_data)

plt.xlabel('LotFrontage')

plt.ylabel('LotArea')
all_data['LotFrontage'].fillna(all_data['LotFrontage'].mean(),inplace=True)
all_data['GarageYrBlt'].fillna(0,inplace=True)

all_data.isnull().sum().sort_values(ascending=False)[:5]
all_data['BsmtQual'].fillna('NA',inplace=True)

all_data['BsmtFinType1'].fillna('NA',inplace=True)

all_data.isnull().sum().sort_values(ascending=False)[:5]
all_data.dtypes[all_data.dtypes!='object'][:60]
all_data['MoSold']=all_data['MoSold']/13

all_data['YrSold']+=all_data['MoSold']
all_data.drop('MoSold',axis=1,inplace=True)
all_data['MSSubClass']=all_data['MSSubClass'].apply(str)

all_data['OverallCond']=all_data['OverallCond'].apply(str)

all_data['OverallQual']=all_data['OverallQual'].apply(str)
numeric=all_data.dtypes[all_data.dtypes!='object'].index

numeric
corr_data=all_data.corr()

sns.heatmap(corr_data)
all_data[numeric].skew()
for features in numeric:

    if all_data[features].skew()>0.7:

        all_data[features]=np.log1p(all_data[features])

all_data[numeric].skew()
all_data.head()
all_data=pd.get_dummies(all_data)

all_data.shape
all_data.head()
n=train.shape[0]

n
train=all_data[:n]

test=all_data[n:]

print('train:',train.shape,'test:',test.shape)
scorer=make_scorer(mean_squared_error,greater_is_better=False)
x_train,x_test,y_train,true_p=train_test_split(train,y,random_state=121,test_size=0.2,shuffle=True)

print('x_train,x_test,y_train,true_p\n')

print(x_train.shape,x_test.shape,y_train.shape,true_p.shape)
def rmse_train_cv(model):

    rmse=np.sqrt(-cross_val_score(model,x_train,y_train,cv=5,scoring=scorer))

    return rmse

def rmse_test_cv(model):

    rmse=np.sqrt(-cross_val_score(model,x_test,true_p,cv=5,scoring=scorer))

    return rmse
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,Ridge,Lasso
lreg=LinearRegression()

lreg.fit(x_train,y_train)

print('rmse value of train=',rmse_train_cv(lreg).mean())

print('rmse value of test=',rmse_test_cv(lreg).mean())
train_pred=lreg.predict(x_train)

test_pred=lreg.predict(x_test)
plt.scatter(x=train_pred,y=train_pred-y_train,c='blue',marker='s',label='train data')

plt.scatter(x=test_pred,y=test_pred-true_p,c='green',marker='s',label='test data')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.title('linear regression')

plt.plot([10.0,13.5],[0.0,0.0],c='red')

plt.show()

plt.scatter(x=train_pred,y=y_train,c='blue',marker='s',label='train data')

plt.scatter(x=test_pred,y=true_p,c='green',marker='s',label='test data')

plt.xlabel('predicted values')

plt.ylabel('actual values')

plt.title('linear regression')

plt.plot([11.0,13.2],[11.0,13.2],c='red')

plt.show()
r2_score(test_pred,true_p)
x_train,x_test,y_train,true_p=train_test_split(train,y,random_state=121,test_size=0.2,shuffle=True)

print('x_train,x_test,y_train,true_p\n')

print(x_train.shape,x_test.shape,y_train.shape,true_p.shape)
alphas=[0.001,0.003,0.007,0.01,0.03,0.06,0.09,0.2,0.6,1,3,6,10,30]

ridge=RidgeCV(alphas)

ridge.fit(x_train,y_train)

alpha=ridge.alpha_

print('best alpha:',alpha)
print('optimising alpha')

alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4]

ridge=RidgeCV(alphas,cv=10)

ridge.fit(x_train,y_train)

best_alpha=ridge.alpha_

print('optimised alpha:',best_alpha)
print('rmse value of train data in ridge regression:',rmse_train_cv(ridge).mean())

print('rmse value of test data in ridge regression:',rmse_test_cv(ridge).mean())
train_pred=ridge.predict(x_train)

test_pred=ridge.predict(x_test)
plt.scatter(x=train_pred,y=train_pred-y_train,c='blue',marker='s',label='train data')

plt.scatter(x=test_pred,y=test_pred-true_p,c='green',marker='s',label='test data')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.title('ridge regression')

plt.plot([10.0,13.5],[0.0,0.0],c='red')

plt.show()

plt.scatter(x=train_pred,y=y_train,c='blue',marker='s',label='train data')

plt.scatter(x=test_pred,y=true_p,c='green',marker='s',label='test data')

plt.xlabel('predicted values')

plt.ylabel('actual values')

plt.title('ridge regression')

plt.plot([11.0,13.2],[11.0,13.2],c='red')

plt.show()
r2_score(test_pred,true_p)
lasso=LassoCV(alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1],cv=10,max_iter=10000)

lasso.fit(x_train,y_train)

alpha=lasso.alpha_

print('best alpha:',alpha)
print('optimising alpha')

lasso=LassoCV(alphas=[alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4],cv=10,max_iter=10000)

lasso.fit(x_train,y_train)

best_alpha=lasso.alpha_

print('optimised alpha:',best_alpha)
print('rmse value of train data in lasso regression:',rmse_train_cv(lasso).mean())

print('rmse value of test data in lasso regression:',rmse_test_cv(lasso).mean())
train_pred=lasso.predict(x_train)

test_pred=lasso.predict(x_test)
plt.scatter(x=train_pred,y=train_pred-y_train,c='blue',marker='s',label='train data')

plt.scatter(x=test_pred,y=test_pred-true_p,c='green',marker='s',label='test data')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.title('lasso regression')

plt.plot([10.0,13.5],[0.0,0.0],c='red')

plt.show()

plt.scatter(x=train_pred,y=y_train,c='blue',marker='s',label='train data')

plt.scatter(x=test_pred,y=true_p,c='green',marker='s',label='test data')

plt.xlabel('predicted values')

plt.ylabel('actual values')

plt.title('lasso regression')

plt.plot([11.0,13.2],[11.0,13.2],c='red')

plt.show()
r2_score(test_pred,true_p)
final_pred=lreg.predict(test)

sub = pd.DataFrame()

sub['Id'] = test_id

sub['SalePrice'] = final_pred

sub.to_csv('submission.csv',index=False)