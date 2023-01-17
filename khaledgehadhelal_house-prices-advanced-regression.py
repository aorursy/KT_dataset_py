import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train_data_path='../input/house-prices-advanced-regression-techniques/train.csv'
test_data_path='../input/house-prices-advanced-regression-techniques/test.csv'
train_data=pd.read_csv(train_data_path)
test_data=pd.read_csv(test_data_path)

train_data.head()
train_data['MSZoning'].value_counts()
# show the missing values
sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False)
train_data.shape
train_data.info()
#histogram
sns.distplot(train_data['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())
var='Id'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='MSSubClass'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='MSZoning'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='LotFrontage'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='LotArea'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='GrLivArea'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='OverallQual'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='TotalBsmtSF'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='CentralAir'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='GarageArea'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='MoSold'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='YrSold'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
var='Heating'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
data.plot.scatter(x=var,y='SalePrice')
sns.set()
cols=['Heating','YrSold','MoSold','GarageArea','CentralAir','OverallQual','GrLivArea','LotArea','LotFrontage','MSZoning','MSSubClass','Id','SalePrice']
sns.pairplot(train_data[cols], size = 2.5)
plt.show();
var='GarageArea'
data=pd.concat([train_data['SalePrice'],train_data[var]],axis=1)
f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=var,y='SalePrice',data=data)
fig.axis(ymin=0, ymax=800000);

var = 'YearBuilt'
data = pd.concat([train_data['SalePrice'],train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#correlation matrix
corrmat=train_data.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True);

#missing data
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#dealing with missing data
train_data = train_data.drop(['Id','PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType'],axis=1)
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
train_data.isnull().sum().max() 
#just checking that there's no missing data missing.
#standardizing data
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'],train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'],train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
sns.distplot(train_data['SalePrice'],fit=norm)
fig=plt.figure()
res=stats.probplot(train_data['SalePrice'],plot=plt)
train_data['SalePrice']=np.log(train_data['SalePrice'])
sns.distplot(train_data['SalePrice'],fit=norm)
fig=plt.figure()
res=stats.probplot(train_data['SalePrice'],plot=plt)
sns.distplot(train_data['LotArea'],fit=norm)
fig=plt.figure()
res=stats.probplot(train_data['LotArea'],plot=plt)
train_data['LotArea']=np.log(train_data['LotArea'])
sns.distplot(train_data['LotArea'],fit=norm)
fig=plt.figure()
res=stats.probplot(train_data['LotArea'],plot=plt)
sns.distplot(train_data['YearBuilt'],fit=norm)
fig=plt.figure()
res=stats.probplot(train_data['YearBuilt'],plot=plt)
train_data['YearBuilt']=np.log(train_data['YearBuilt'])
sns.distplot(train_data['YearBuilt'],fit=norm)
fig=plt.figure()
res=stats.probplot(train_data['YearBuilt'],plot=plt)
train_data=pd.get_dummies(train_data)
#missing data
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train_data.head()
from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(train_data , test_data , test_size = 0.10,random_state =2)

from sklearn.impute import SimpleImputer
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


imp=SimpleImputer()
x_train=imp.fit_transform(x_train)
x_test=imp.fit_transform(x_test)
y_train=imp.fit_transform(x_train)
y_test=imp.fit_transform(x_test)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

reg.score(x_test,y_test)
