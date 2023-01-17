import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns ; sns.set()
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test.head()
sub = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub.head()
fig, ax =plt.subplots(1,2,figsize=(20,6))

sns.distplot(train['SalePrice'],ax=ax[0],color='#478865')

sns.distplot(sub['SalePrice'], ax=ax[1],color='#478865')

ax[0].title.set_text('Train')

ax[1].title.set_text('Test')
data = pd.concat([train,test])
plt.figure(figsize=(20,6))

sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap='summer')
data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

data.head()
cat_data = data.select_dtypes(include='object')

cat_data.head()
plt.figure(figsize=(8,6))

sns.heatmap(cat_data.isnull(), cbar=False, yticklabels=False, cmap='summer')
cat_data.isnull().sum()
group1 = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']

cat_data[group1] = cat_data[group1].fillna('NA')
group2 = ['MSZoning','Utilities','Exterior1st','Exterior2nd','MasVnrType','Electrical','KitchenQual','Functional','SaleType']

cat_data[group2] = cat_data[group2].fillna(cat_data.mode().iloc[0])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

labels = ['ExterQual','ExterCond','HeatingQC','KitchenQual','BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond']

for i in labels:

    cat_data[i] = labelencoder.fit_transform(cat_data[i])
cat_data = pd.get_dummies(cat_data,columns=cat_data.select_dtypes(include='object').columns)

cat_data
num_data = data.select_dtypes(exclude='object')

num_data
plt.figure(figsize=(8,6))

sns.heatmap(num_data.isnull(), cbar=False, yticklabels=False, cmap='summer')
num_data.isnull().sum()
num_data['LotFrontage'].fillna(data['LotFrontage'].mean(),inplace=True)

num_data['GarageYrBlt'].fillna(data['GarageYrBlt'].mean(),inplace=True)

num_data['MasVnrArea'].fillna(data['MasVnrArea'].mean(),inplace=True)

num_data = num_data.fillna(0)
data = pd.concat([num_data,cat_data],axis=1)

data
data_train = data[data['Id']<1461]

data_test = data[data['Id']>1460]
X_train = data_train.drop('SalePrice',axis=1)

Y_train = data_train['SalePrice']

X_test = data_test.drop('SalePrice',axis=1)

Y_test = sub['SalePrice']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.33,random_state=0)
from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(random_state=0)

rfr.fit(x_train, y_train)

s1 = rfr.score(x_test,y_test)

pred1 = rfr.predict(x_test)

rmse1 = np.sqrt(metrics.mean_squared_error(y_test,pred1))
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=3000,learning_rate=0.01)

xgb.fit(x_train,y_train)

s2 = xgb.score(x_test,y_test)

pred2 = xgb.predict(x_test)

rmse2 = np.sqrt(metrics.mean_squared_error(y_test,pred2))
from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(n_estimators=3000)

lgbm.fit(x_train,y_train)

s3 = lgbm.score(x_test,y_test)

pred3 = lgbm.predict(x_test)

rmse3 = np.sqrt(metrics.mean_squared_error(y_test,pred3))
fig, ax =plt.subplots(1,2,figsize=(20,6))

sns.barplot(x=['RF','XGB','LGBM'], y=[s1,s2,s3], palette='viridis',ax=ax[0])

sns.barplot(x=['RF','XGB','LGBM'], y=[rmse1,rmse2,rmse3],palette='viridis',ax=ax[1])

ax[0].title.set_text('Score')

ax[0].set(ylim=(0.8,0.9))

ax[1].title.set_text('RMSE')

ax[1].set(ylim=(25000,32000))
pred_final = rfr.predict(X_test)*0.3+xgb.predict(X_test)*0.3+lgbm.predict(X_test)*0.4
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

submission['SalePrice'] = pred_final

submission.to_csv("submission.csv", index=False)