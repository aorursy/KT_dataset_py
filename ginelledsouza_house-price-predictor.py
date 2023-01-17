# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
house_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
house_train.head()
house_test.head()
print('Shape of the taining set') 
print(house_train.shape)
print('\nShape of the testing set') 
print(house_test.shape)
print('Information about the taining set\n') 
house_train.info()
print('\nInformation about the testing set\n') 
house_test.info()
data = [house_train, house_test]

for dataset in data:
    percentage = round(((dataset.isnull().sum()*100)/(dataset.shape[0])),4).sort_values(ascending=False)
    print(percentage.head(20),'\n')
for dataset in data:
    dataset.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage'],axis=1,inplace=True)
for dataset in data:
    percentage = round(((dataset.isnull().sum()*100)/(dataset.shape[0])),4).sort_values(ascending=False)
    print(percentage.head(30),'\n')
values = ['GarageYrBlt','GarageFinish','GarageCond','GarageQual','GarageType','BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2',
          'MasVnrType','BsmtHalfBath','MSZoning','Functional','Utilities','BsmtFullBath','Exterior2nd','Exterior1st','KitchenQual','TotalBsmtSF',
          'GarageCars','SaleType','GarageArea','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','MasVnrArea','Electrical']

for dataset in data:
    for feature in values:
        mode_in = dataset[feature].mode()[0]
        #print(mode_in)
        dataset[feature] =  dataset[feature].fillna(mode_in) 
for dataset in data:
    percentage = round(((dataset.isnull().sum()*100)/(dataset.shape[0])),4).sort_values(ascending=False)
    print(percentage.head(30),'\n')
print("Columns in the training set: ",house_train.shape[1])
print("\nColumns in the testing set: ",house_test.shape[1])
correlation = house_train.corr()
correlation['SalePrice'].sort_values(ascending=False)[:11]
max_corr = correlation['SalePrice'].sort_values(ascending=False)[:11].index
house_train[max_corr].head()
print(house_train.groupby('OverallQual').mean()['SalePrice'])
sns.barplot(x='OverallQual',y='SalePrice',data=house_train)
print(house_train.groupby('OverallQual').count()['Id'])
sns.countplot('OverallQual',data=house_train)
sns.jointplot(x='GrLivArea',y='SalePrice',data=house_train,kind='reg')
house_train = house_train[house_train['GrLivArea']<4500]
sns.jointplot(x='GrLivArea',y='SalePrice',data=house_train,kind='reg')
sns.jointplot(x='TotalBsmtSF',y='SalePrice',data=house_train,kind='reg')
print(house_train.groupby('FullBath').count()['Id'])
sns.countplot('FullBath',data=house_train)
print(house_train.groupby('TotRmsAbvGrd').count()['Id'])
sns.countplot('TotRmsAbvGrd',data=house_train)
sns.distplot(house_train['YearBuilt'],bins=30,kde=False)
sns.jointplot(x='GarageArea',y='SalePrice',data=house_train,kind='reg')
house_train = house_train[house_train['GarageArea'] < 1200]
sns.jointplot(x='GarageArea',y='SalePrice',data=house_train,kind='reg')

sns.distplot(house_train['YearRemodAdd'],bins=30,kde=False)
sns.jointplot(x='MasVnrArea',y='SalePrice',data=house_train,kind='reg')
house_train = house_train[house_train['MasVnrArea']<1500]
sns.jointplot(x='MasVnrArea',y='SalePrice',data=house_train,kind='reg')
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
gar = house_train.groupby('GarageType').count()['Id']
gar.plot.bar()
plt.xlabel('Garage Type')
plt.ylabel('Total Count')
plt.title('Subplot 1: Count Of Garage')

plt.subplot(1,2,2)
gar = house_train.groupby('GarageType').mean()['SalePrice']
gar.plot.bar()
plt.xlabel('Garage Type')
plt.ylabel('Total Cost')
plt.title('Subplot 2: Cost Of Garage')
table1 = pd.pivot_table(house_train, values=['SalePrice'], index=['Street'],columns=['LotShape'],aggfunc=np.mean)
table1
ax = table1.plot.bar(figsize=(8,5))
ax.set_xlabel("Street and Lot Shape")
ax.set_ylabel("Sale Price")
plt.figure(figsize=(10,5))
plt.title('Frequency Of The Sales Price')
plt.xlabel('Sales Price')
sns.distplot(house_train['SalePrice'])
print(house_train['CentralAir'].value_counts())
sns.countplot('CentralAir',data = house_train)
sale = house_train.groupby('SaleCondition').mean()['SalePrice']
sale.plot.bar()
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
numeric_train = pd.DataFrame(house_train.select_dtypes(include=[np.number]))
numeric_test =  pd.DataFrame(house_test.select_dtypes(include=[np.number]))
new_data = [numeric_train, numeric_test]
for dataset in new_data:
    dataset.drop(['Id'],axis=1,inplace=True)
for dataset in new_data:
    for i in dataset.columns:
            dataset[i] = dataset[i].astype(int)
for dataset in new_data:
    dataset['YearBltAge'] = dataset['YrSold'] - dataset['YearBuilt']
    dataset['RemodAge'] = dataset['YrSold'] - dataset['YearRemodAdd']
for dataset in new_data:
    dataset.drop(['YrSold','YearBuilt'],axis=1,inplace=True)
numeric_train['SalePrice'] = numeric_train['SalePrice'].fillna(0)
numeric_train['SalePrice'] = numeric_train['SalePrice'].astype(int)
print("Shape of the training set")
print(numeric_train.shape)
print("\nShape of the testing set")
print(numeric_test.shape)
X = numeric_train.drop('SalePrice',axis=1)
y = np.log(numeric_train['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Import Packages 
from sklearn.linear_model import LinearRegression
#Object creation and fitting of training set
lrm = LinearRegression()
lrm.fit(X_train,y_train)
#Creation of a prediction variable
predictionslrm = lrm.predict(X_test)
#Create a prediction score
scorelrm = round((lrm.score(X_test, y_test)*100),2)
print ("Model Score:",scorelrm,"%")
#Import Packages 
from sklearn.linear_model import Ridge
#Object creation and fitting of training set
rrm = Ridge(alpha=100)
rrm.fit(X_train,y_train)
#Creation of a prediction variable
predictionrrm = rrm.predict(X_test)
#Create a prediction score
scorerrm = round((rrm.score(X_test, y_test)*100),2)
print ("Model Score:",scorerrm,"%")
data = [['Linear Regression',scorelrm],['Ridge Regression',scorerrm]]
final = pd.DataFrame(data,columns=['Algorithm','Precision'],index=[1,2])
final