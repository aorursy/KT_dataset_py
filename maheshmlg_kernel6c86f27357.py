import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as stats

from scipy.stats import norm, skew

import os 

print(os.listdir('../input/house-prices-advanced-regression-techniques'))
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.head())

print('**'* 50)

print(test.head())
print(train.head().info())

print('**'* 50)

print(test.info())
#1) saleprice distribution plot 

plt.figure(figsize=(10,4))

sns.distplot(train['SalePrice'] , fit=norm);#from import seaborn as sns

(mu,sigma)=norm.fit(train['SalePrice'])#to compute mu, sigma

print('\n mu = {:.2f} sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['normal dist. ($\mu =$ {:.2f} $\sigma =$ {:.2f})\n'.format(mu, sigma)])

plt.ylabel('Frequency')

plt.xlabel('SalePrice')

plt.show()
#2) QQ-plot Scatter plot





fig=plt.figure(figsize=(10,4))

res=stats.stats.probplot(train['SalePrice'],plot=plt)

plt.show()
plt.figure(figsize=(30,30))

sns.heatmap(train.corr(),cmap='coolwarm',annot=True)

plt.show()
sns.lmplot(x='1stFlrSF',y='SalePrice',data=train)

plt.show()
sns.lmplot(x='GarageCars',y='GarageArea',data=train)

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot('GarageCars','SalePrice',data=train)

sns.lmplot(x='GarageCars',y='SalePrice',data=train)

plt.show()
sns.lmplot(x='OverallQual',y='SalePrice',data=train)

sns.boxplot(x='OverallQual',y='SalePrice',data=train)

plt.show()
sns.lmplot(x='GarageArea',y='SalePrice',data=train)

plt.show()
plt.figure(figsize=(8,4))

sns.barplot(x='FullBath', y='SalePrice',data=train)

plt.show()
total=train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
train=train.drop((missing_data[missing_data['Total']>81].index),1)
train.isnull().sum().sort_values(ascending=False).head(20)
total_test=test.isnull().sum().sort_values(ascending=False)

percent_test=(test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total_test,percent_test],axis=1,keys=['Total','Percent'])

missing_data.head(25)
test=test.drop((missing_data[missing_data['Total']>78]).index,1)
test.isnull().sum().sort_values(ascending=False).head(20)
categorical_feature_mask = train.dtypes==object
categorical_cols = train.columns[categorical_feature_mask].tolist()
train[categorical_cols].head()
from sklearn.preprocessing import LabelEncoder

labelencoder =  LabelEncoder()

train[categorical_cols]=train[categorical_cols].apply(lambda x:labelencoder.fit_transform(x.astype(str)))
# Categorical boolean mask

categorical_feature_mask_test = test.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols_test = test.columns[categorical_feature_mask_test].tolist()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

test[categorical_cols_test] = test[categorical_cols_test].apply(lambda y: labelencoder.fit_transform(y.astype(str)))
test[categorical_cols].head()
train.isnull().sum().sort_values(ascending=False).head(15)
test.isnull().sum().sort_values(ascending=False).head(15)
train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
plt.figure(figsize=(16,10))

cols=train.corr().nlargest(15,'SalePrice')['SalePrice'].index #picking top 15 correlated 

cormat=np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hmap=sns.heatmap(cormat,cbar=True,annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
train=train[cols]
test=test[cols.drop('SalePrice')]
test.isnull().sum().sort_values(ascending=False).head(20)
test.head()
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(train.drop('SalePrice',axis=1),train['SalePrice'],test_size=0.3,random_state=101)
ytrain=ytrain.values.reshape(-1,1)

ytest= ytest.values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler

scx=StandardScaler()

scy=StandardScaler()

xtrain=scx.fit_transform(xtrain)

xtest=scx.fit_transform(xtest)

ytrain=scx.fit_transform(ytrain)

ytest=scy.fit_transform(ytest)

xtrain
from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(xtrain,ytrain)

print(lm)
print(lm.intercept_)
print(lm.coef_)
predictions=lm.predict(xtest)

predictions=predictions.reshape(-1,1)
plt.figure(figsize=(16,8))

plt.scatter(ytest,predictions)

plt.xlabel('y test')

plt.ylabel('predicted values')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(ytest,label='ytest')

plt.plot(predictions,label='predicted values')

plt.show()
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(ytest,predictions))

print('MSE:',metrics.mean_squared_error(ytest,predictions))

print('RMSE:',np.sqrt(metrics.mean_squared_error(ytest,predictions)))

from sklearn import ensemble

from sklearn.utils import shuffle

from sklearn.metrics import mean_squared_error,r2_score
params={'n_estimators':500,'max_depth':4,'min_samples_split':2,'learning_rate':0.01,'loss':'ls'}

clf=ensemble.GradientBoostingRegressor(**params) #** is used to unpack params dictionary

clf.fit(xtrain,ytrain)
clfpred=clf.predict(xtest)

clfpred=clfpred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(ytest, clfpred))

print('MSE:', metrics.mean_squared_error(ytest, clfpred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, clfpred)))
plt.figure(figsize=(15,8))

plt.scatter(ytest,clfpred, c= 'brown')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()



plt.figure(figsize=(16,8))

plt.plot(ytest,label ='Test')

plt.plot(clfpred, label = 'predict')

plt.show()
from sklearn.tree import DecisionTreeRegressor

dtree=DecisionTreeRegressor(random_state=100)

dtree.fit(xtrain,ytrain)
dtreepred=dtree.predict(xtest)

dtreepred=dtreepred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(ytest, dtreepred))

print('MSE:', metrics.mean_squared_error(ytest, dtreepred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, dtreepred)))
plt.figure(figsize=(15,8))

plt.scatter(ytest,dtreepred,c='green')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
from sklearn.svm import SVR

svr=SVR(kernel='rbf')

svr.fit(xtrain,ytrain)
svrpred=svr.predict(xtest)

svrpred=svrpred.reshape(-1,1)
print('MAE: ',metrics.mean_absolute_error(ytest,svrpred))

print('MSE: ',metrics.mean_squared_error(ytest,svrpred))

print('RMSE: ',np.sqrt(metrics.mean_squared_error(ytest,svrpred)))
plt.figure(figsize=(16,8))

plt.scatter(ytest,svrpred,c='red')

plt.xlabel('ytest')

plt.ylabel('predicted y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(ytest,label='Test')

plt.plot(svrpred,label='Predict')

plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor(n_estimators=500,random_state=0)

rfr.fit(xtrain,ytrain)
rfrpred=rfr.predict(xtest)

rfrpred=rfrpred.reshape(-1,1)




print('MAE:', metrics.mean_absolute_error(ytest, rfrpred))

print('MSE:', metrics.mean_squared_error(ytest, rfrpred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, rfrpred)))



plt.figure(figsize=(15,8))

plt.scatter(ytest,rfrpred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(ytest,label ='Test')

plt.plot(rfrpred, label = 'predict')

plt.show()
import lightgbm as lgb

lgbmodel=lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=750,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgbmodel.fit(xtrain,ytrain)
lgbpred=lgbmodel.predict(xtest)

lgbpred=lgbpred.reshape(-1,1)
print('MAE:', metrics.mean_absolute_error(ytest, lgbpred))

print('MSE:', metrics.mean_squared_error(ytest, lgbpred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, lgbpred)))
plt.figure(figsize=(16,8))

plt.scatter(ytest,lgbpred, c='orange')

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
plt.figure(figsize=(16,8))

plt.plot(ytest,label ='Test')

plt.plot(lgbpred, label = 'predict')

plt.show()
error_rates=[metrics.mean_squared_error(ytest,predictions),

             metrics.mean_squared_error(ytest,clfpred),

             metrics.mean_squared_error(ytest,dtreepred),

             metrics.mean_squared_error(ytest,svrpred),

             metrics.mean_squared_error(ytest,rfrpred),

             metrics.mean_squared_error(ytest,lgbpred)]

labels=['Linear Regression','Gradient boosting','Decision tree','Support vector regressor ','Random Forest','LightGBM']
plt.figure(figsize=(16,5))

plt.plot(error_rates)

for x,y,z in zip(range(len(labels)),error_rates,labels):



    label = "{:.2f}".format(y)

    label1= z



    plt.annotate(label, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,0), # distance from text to points (x,y)

                 ha='center') # horizontal alignment can be left, right or center

    plt.annotate(label1, # this is the text

                 (x,y), # this is the point to label

                 textcoords="offset points", # how to position the text

                 xytext=(0,20), # distance from text to points (x,y)

                 ha='center') # horizontal alignment can be left, right or center
a=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_id=a['Id']

a=pd.DataFrame(test_id,columns=['Id'])
test=scx.fit_transform(test)
test.shape
testpred=clf.predict(test)

testpred=testpred.reshape(-1,1)

testpred




testpred =scy.inverse_transform(testpred)







testpred = pd.DataFrame(testpred, columns=['SalePrice'])



testpred.head()
result = pd.concat([a,testpred], axis=1)

result
result.to_csv('submission.csv',index=False)