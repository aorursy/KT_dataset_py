# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import liberies 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import scipy as stats 
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.info()
train.head()
# saleprice describiton
train.SalePrice.describe()
#histogram
sns.distplot(train['SalePrice'])
#skewness and kurtosis
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);
corr= train.corr()['SalePrice'].sort_values(ascending=False)
corr
# most correlated features
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.set()
cols = ['OverallQual' , 'GrLivArea' ,'GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(train[cols], size = 2.5)
plt.show();
sns.barplot(train.OverallQual,train.SalePrice)
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
figure = plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
sns.distplot(train['SalePrice'])

from scipy import stats
plt.subplot(1,3,2)
stats.probplot(train['SalePrice'] , plot=plt)

plt.subplot(1,3,3)
sns.boxplot(train['SalePrice'] , orient='v')
train['SalePrice'] = np.log1p(train['SalePrice'])

print('Skewenss of SalePrice :- ' , train['SalePrice'].skew())
print('Kuroises of SalePrice:- '  ,train['SalePrice'].kurt())
figure = plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
sns.distplot(train['SalePrice'])

from scipy import stats
plt.subplot(1,3,2)
stats.probplot(train['SalePrice'] , plot=plt)

plt.subplot(1,3,3)
sns.boxplot(train['SalePrice'] , orient='v')
numeric_data = train.select_dtypes(include=np.number).drop(['SalePrice'] , axis=1)
items = numeric_data.loc[ : ,['OverallQual','GrLivArea','GarageCars','TotalBsmtSF',
                              'FullBath','YearBuilt','YearRemodAdd'  ] ]
  #visualize these itemes using Boxplot
fig = plt.figure(figsize=(12,12))
for col in range(len(items.columns)) : 
    fig.add_subplot(3 , 3 , col+1)
    sns.boxplot(y=items.iloc[: , col])
plt.show()
# Visualize these items using multivariate analysis (SCatter plot)   
fig = plt.figure(figsize=(16,12))
for col in range(len(items.columns)):
    fig.add_subplot(3,3,col+1)
    sns.scatterplot(items.iloc[ : , col] , train['SalePrice'])
plt.show()
# Using Z-Score to identify outliers
from scipy import stats
z= np.abs(stats.zscore(items))
print(z)
threshold = 4
print(np.where(z > threshold))
# Remove outlier using z-score
train.shape
train = train[(z < threshold).all(axis=1)]
train.shape
train.corr()['SalePrice'].sort_values(ascending=False)[:10]
def missing_val(df):
    total = df.isnull().sum().sort_values(ascending=False)
    total_miss = total[total != 0]
    percent = round(total_miss / len(df)*100,2)
    return pd.concat((total_miss, percent) , axis =1 , keys=['Total Miss' , 'percent'])
missing_val(train)
missing_val(test)
dataset = pd.concat((train,test) , sort = False).reset_index(drop=True)
dataset = dataset.drop(columns =['SalePrice'] , axis =1 )
missing_val(dataset)
dataset.drop(['Id','Utilities','PoolQC','MiscFeature','Alley','Fence','GarageYrBlt'],axis=1 , inplace=True)
missing_val(dataset)
miss_mode =  ['MasVnrArea' , 'Electrical' , 'MSZoning' , 'SaleType','Exterior1st','Exterior2nd','KitchenQual']
for col in miss_mode:
    dataset[col]  = dataset[col].fillna(dataset[col].mode()[0])
    
missing_feat = ['GarageType','GarageCond','GarageQual','GarageFinish',
                'BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual',
                'FireplaceQu','MasVnrType']
for col in missing_feat:
    dataset[col]=dataset[col].fillna('None')

dataset['Functional'] = dataset['Functional'].fillna('Typ')
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].median())

miss_zero = ['BsmtHalfBath','BsmtFullBath','GarageArea','GarageCars','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFinSF1']
for col in miss_zero:
    dataset[col] = dataset[col].fillna(0)
missing_val(dataset)
dataset.dropna(inplace=True)
dataset.shape
dataset['MSSubClass']   = train['MSSubClass'].astype(str)
#dataset['YrSold']       = dataset['YrSold'].astype(str)
#dataset['MoSold']       = dataset['MoSold'].astype(str)
#dataset['YearBuilt']    = dataset['YearBuilt'].astype(str)
#dataset['YearRemodAdd'] = dataset['YearRemodAdd'].astype(str)
dataset['totalSF'] =( dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']  )
dataset['ageHouse'] = (dataset['YrSold'] - dataset['YearBuilt'] )

#check for duplicate rows 
duplicate= train[train.duplicated()]
print(duplicate) # there is no duplicate rows
dataset.shape
final_features = pd.get_dummies(dataset).reset_index(drop=True)
print(final_features.shape)
final_features.head()
final_features =final_features.loc[:,~final_features.columns.duplicated()]
final_features.shape
y= train['SalePrice']
X = final_features.iloc[: len(y) , :]
df_test  = final_features.iloc[len(y): , :]
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(np.array(y).reshape(-1,1))
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X , y)
y_pred = lr.predict(df_test)
print(y_pred)
from sklearn.model_selection import KFold , cross_val_score
#lr = LinearRegression()
cv = KFold(shuffle= True , random_state=2 , n_splits=10)
scores = cross_val_score(lr , X , y , cv =cv ,scoring = 'neg_mean_absolute_error' )
print(scores.mean())
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
ridge = Ridge(alpha = 400)
ridge.fit(X , y)
test_pred = ridge.predict(df_test)
print(test_pred)
import pickle
filename = 'Ridge_model.pkl'
pickle.dump(ridge , open(filename , 'wb') )
## create simple submission file 
pred = pd.DataFrame(test_pred)
sample_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
final_data = pd.concat([sample_df['Id'] , pred] , axis=1)
final_data.columns=['Id' , 'SalePrice']
final_data.to_csv('Ridge_model.csv' , index=False)
