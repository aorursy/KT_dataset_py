# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load the train and test dataset



data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
data.head()
data.info()
#find out the shape of train data

data.shape
#plot the missing value attributes

plt.figure(figsize=(12,9))

sns.heatmap(data.isnull())

plt.show()
#Print the null columns in the training dataset

data.columns[data.isnull().any()]
Is_null_data = data.isnull().sum() / len(data) * 100

Is_null_data = Is_null_data[Is_null_data > 0]

Is_null_data.sort_values(inplace = True, ascending = False)

print(Is_null_data)
Is_null_data
Is_null_data = Is_null_data.to_frame()

print(Is_null_data)
plt.figure(figsize = (15,6))

barplot_isnull = sns.barplot(x=Is_null_data.index,y=Is_null_data[0])

barplot_isnull.set_xticklabels(barplot_isnull.get_xticklabels(),rotation = 45)

plt.xlabel('missing value lables')

plt.ylabel('Percentge of missing values')

plt.title('Missing value VS Percentage of missing values')

plt.show()
#PoolQC has missing values ratio is 99% so there is fill by none

data.drop(['PoolQC'],axis = 1,inplace = True)
data.drop(['MiscFeature'],axis = 1,inplace = True)

data.drop(['Alley'],axis = 1,inplace = True) 

data.drop(['Fence'],axis = 1,inplace = True) 

data.drop(['FireplaceQu'],axis = 1,inplace = True)
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(

    lambda x: x.fillna(x.median()))
for col in ['GarageType','GarageFinish','GarageQual','GarageCond']:

    data[col] = data[col].fillna(data[col].mode()[0])
data.drop(['GarageYrBlt'],axis = 1, inplace = True)
data.shape
plt.figure(figsize=(12,6))

sns.heatmap(data.isnull())
remaining_null_values = data.isnull().sum() / len(data)* 100

remaining_null_values = remaining_null_values[remaining_null_values > 0]

print(remaining_null_values)
col =['MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure', 'BsmtFinType1','BsmtFinType2','Electrical']
data[col].info()
data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])

data['BsmtCond'] = data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])

data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mode()[0])
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0])

data['BsmtFinType2'] = data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])
data.dropna(inplace = True)
plt.figure(figsize= (12,6))

sns.heatmap(data.isnull())

plt.show()
print('The shape of the final training dataset is ',  data.shape)
data.columns
col = ('Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle','OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle','RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea','ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2','BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC','CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd','Functional', 'Fireplaces', 'GarageType', 'GarageFinish', 'GarageCars','GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch','PoolArea','MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition')
from sklearn.preprocessing import LabelEncoder

for cols in col:

    le = LabelEncoder()

    le.fit(list(data[cols].values))

    data[cols] = le.transform(list(data[cols].values))
X = data.iloc[:,0:74].values

y = data.iloc[:,-1].values
#split the dataset into train and test dataset

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state = 6)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)
y_hat = lr.predict(X_test)
#Accuracy of linear regression

print('Model Accuracy :' , lr.score(X_test,y_test)*100)
#Train the model for random forest regression

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000)
rf.fit(X_train,y_train)
print('Random Forest Accuracy :',rf.score(X_test,y_test)*100)
#Train the model on Gradient Boosting Regressor

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100,max_depth=4)
gbr.fit(X_train,y_train)
print('Accuracy of Gradient Boosting Regressor is :',gbr.score(X_test,y_test)*100)