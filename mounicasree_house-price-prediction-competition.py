# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_columns', 81)

pd.set_option('display.max_rows', 90)
house_price_train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col='Id')
house_price_train.head()
house_price_test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col='Id')

house_price_test.head()
import matplotlib

import matplotlib.pyplot as plt



matplotlib.style.use('ggplot')





house_price_train['SalePrice'].hist(alpha=0.5, figsize=(10, 6))
target = np.log(house_price_train['SalePrice'])
target.hist(alpha=0.5, figsize=(10,6))
house_price_train.dtypes
numeric_train_features = house_price_train.select_dtypes(include=[np.number])

numeric_train_features.dtypes
import seaborn as sns

fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(numeric_train_features.corr()[['SalePrice']].sort_values('SalePrice').tail(16),

 vmax=1, vmin=-1, cmap='YlGnBu', annot=True, ax=ax);

ax.invert_yaxis()
fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(numeric_train_features.corr()[['SalePrice']].sort_values('SalePrice').head(16),

 vmax=1, vmin=-1, cmap='YlGnBu', annot=True, ax=ax);

ax.invert_yaxis()
numeric_train_features['OverallQual'].value_counts()
plt.boxplot(numeric_train_features['OverallQual'],patch_artist=True)
numeric_train_features[numeric_train_features['OverallQual']==1]
sns.regplot(numeric_train_features['OverallQual'],numeric_train_features['SalePrice'])
numeric_train_features['GrLivArea'].value_counts()
plt.boxplot(numeric_train_features['GrLivArea'],patch_artist=True)
sns.regplot(numeric_train_features['GrLivArea'],numeric_train_features['SalePrice'])
house_price_train.drop(house_price_train[(house_price_train['GrLivArea']>4000) & (house_price_train['SalePrice']<300000)].index,inplace=True)
sns.regplot(house_price_train['GrLivArea'],house_price_train['SalePrice'])
numeric_train_features['GarageCars'].value_counts()
plt.boxplot(numeric_train_features['GarageCars'])
sns.regplot(numeric_train_features['GarageCars'],numeric_train_features['SalePrice'])
numeric_train_features['GarageArea'].value_counts()
sns.regplot(numeric_train_features['GarageArea'],numeric_train_features['SalePrice'])
house_price_train.drop(house_price_train[(house_price_train['GarageArea']>1200) & (house_price_train['SalePrice']<300000)].index, inplace=True)
sns.regplot(house_price_train['GarageArea'],house_price_train['SalePrice'])
numeric_train_features['TotalBsmtSF'].value_counts()
plt.boxplot(numeric_train_features['TotalBsmtSF'])
sns.regplot(numeric_train_features['TotalBsmtSF'],numeric_train_features['SalePrice'])
house_price_train.drop(house_price_train[(house_price_train['TotalBsmtSF'] > 6000)].index , inplace=True)
sns.regplot(house_price_train['TotalBsmtSF'],house_price_train['SalePrice'])
sns.regplot(numeric_train_features['1stFlrSF'],numeric_train_features['SalePrice'])
plt.boxplot(numeric_train_features['1stFlrSF'])
house_price_train.drop(house_price_train [(house_price_train['1stFlrSF']>4000)].index, inplace=True)
sns.regplot(house_price_train['1stFlrSF'],house_price_train['SalePrice'])
sns.regplot(numeric_train_features['FullBath'],numeric_train_features['SalePrice'])
plt.boxplot(numeric_train_features['FullBath'])
sns.regplot(numeric_train_features['TotRmsAbvGrd'],numeric_train_features['SalePrice'])
plt.boxplot(numeric_train_features['TotRmsAbvGrd'])
house_price_train.drop(house_price_train[(house_price_train['TotRmsAbvGrd']==14)].index,inplace=True)
plt.boxplot(house_price_train['TotRmsAbvGrd'])
sns.regplot(numeric_train_features['YearBuilt'],numeric_train_features['SalePrice'])
plt.boxplot(numeric_train_features['YearBuilt'])
sns.regplot(numeric_train_features['YearRemodAdd'],numeric_train_features['SalePrice'])
plt.boxplot(numeric_train_features['YearRemodAdd'])
sns.regplot(numeric_train_features['GarageYrBlt'],numeric_train_features['SalePrice'])
numeric_train_features['GarageYrBlt'].value_counts()
sns.boxplot(numeric_train_features['GarageYrBlt'])
sns.boxplot(numeric_train_features['MasVnrArea'])
sns.regplot(numeric_train_features['MasVnrArea'],numeric_train_features['SalePrice'])
sns.boxplot(numeric_train_features['Fireplaces'])
sns.regplot(numeric_train_features['Fireplaces'],numeric_train_features['SalePrice'])
house_price_train.shape
house_price_test.shape
house_price_train['logSalePrice'] = np.log(house_price_train['SalePrice']+1)
saleprice = house_price_train[['SalePrice','logSalePrice']]
saleprice.head()
house_price_train.drop(columns=['SalePrice','logSalePrice'],inplace=True)
house_price_train.shape
alldata = pd.concat((house_price_train,house_price_test))
alldata.shape
alldata.head()
null_data = pd.DataFrame(alldata.isnull().sum().sort_values(ascending=False))[:50]



null_data.columns = ['Null Count']

null_data.index.name = 'Feature'

(null_data/len(alldata))*100
alldata[['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities']].dtypes

house_price_train.mode(dropna=False)
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):

    alldata[col] = alldata[col].fillna('None')
for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 'Utilities'):

    alldata[col] = alldata[col].fillna(alldata[col].mode()[0])
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):

    alldata[col] = alldata[col].fillna(0)
neigh_group = alldata.groupby('Neighborhood')

for entry in neigh_group:

    print(entry)
alldata['LotFrontage'] = alldata.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
alldata['TotalSF'] = alldata['TotalBsmtSF']+alldata['1stFlrSF']+alldata['2ndFlrSF']
alldata['No2ndFlr']=(alldata['2ndFlrSF']==0)

alldata['NoBsmt']=(alldata['TotalBsmtSF']==0)
alldata['TotalBath'] = alldata['BsmtFullBath'] + alldata['FullBath'] + alldata['BsmtHalfBath'] + alldata['HalfBath']
alldata['YrBltAndRemod']=alldata['YearBuilt']+alldata['YearRemodAdd']
alldata=alldata.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating'])
alldata.dtypes
alldata['MSSubClass']=alldata['MSSubClass'].astype(str)

alldata['MoSold']=alldata['MoSold'].astype(str)

alldata['YrSold']=alldata['YrSold'].astype(str)
def onehot(col_list):

    global alldata

    for col in col_list:

        #col=col_list.pop(0)

        data_encoded=pd.get_dummies(alldata[col], prefix=col)

        alldata=pd.merge(alldata, data_encoded, on='Id')

        alldata=alldata.drop(columns=col)

    print(alldata.shape)
categorical_data=alldata.select_dtypes(exclude=[np.number, bool])



onehot(list(categorical_data))
def log_transform(col_list):

    transformed_col=[]

    for col in col_list:

        if alldata[col].skew() > 0.5:

            alldata[col]=np.log(alldata[col]+1)

            transformed_col.append(col)

        else:

            pass

    print(f"{len(transformed_col)} features had been tranformed")

    print(alldata.shape)
numeric=alldata.select_dtypes(include=np.number)

log_transform(list(numeric))
house_price_train=alldata[:1454]

house_price_test=alldata[1454:]


house_price_test.shape
house_price_train.shape
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler







from sklearn import linear_model, model_selection, ensemble, preprocessing

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor





#Evaluation Metrics

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
feature_names=list(alldata)

Xtrain=house_price_train[feature_names]

Xtest=house_price_test[feature_names]

Ytrain=saleprice['logSalePrice']
Xtest.shape
lr_model = LinearRegression(n_jobs=-1)

train_X, val_X, train_y, val_y = train_test_split(Xtrain, Ytrain, random_state=1)



lr_model.fit(train_X,train_y)







y_pred = lr_model.predict(val_X)

val_mae = mean_absolute_error(y_pred, val_y)

val_mae

lr_model.fit(Xtrain,Ytrain)
predictions = np.exp(lr_model.predict(Xtest))-1

train_X, val_X, train_y, val_y = train_test_split(Xtrain, Ytrain, random_state=1)

lr_ridge = make_pipeline(RobustScaler(), Ridge(random_state=42,alpha=0.002))

lr_ridge.fit(train_X,train_y)







y_pred = lr_ridge.predict(val_X)







val_mae = mean_absolute_error(y_pred, val_y)

print(val_mae)

lr_ridge.fit(Xtrain,Ytrain)
predictions = np.exp(lr_ridge.predict(Xtest))-1
train_X, val_X, train_y, val_y = train_test_split(Xtrain, Ytrain, random_state=1)

model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=5, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=20)



model_GBoost.fit(train_X,train_y)

y_pred = model_GBoost.predict(train_X)

mae=mean_absolute_error(y_pred, train_y)

print(mae)
model_GBoost.fit(Xtrain,Ytrain)
predictions = np.exp(model_GBoost.predict(Xtest))-1
Xtest=Xtest.reset_index()

output = pd.DataFrame({'Id': Xtest.Id,

                       'SalePrice': predictions})

output.to_csv('submission.csv', index=False)