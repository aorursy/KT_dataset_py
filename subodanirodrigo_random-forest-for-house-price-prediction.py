# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/train.csv")

data1=pd.read_csv("../input/test.csv")

print(data.shape)

print(data1.shape)
#check missing vales

missing = data.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
data.drop(['MiscFeature','PoolQC','Fence','FireplaceQu','Alley','LotFrontage'],axis=1,inplace=True)

print(data.shape)
print("important variables")

corr = data.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
#select important variables

train=data[['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']]

print(train.shape)
#check missing values for important variables

train.isnull().sum()
#print varible type

print(train['GarageYrBlt'].dtype)

print(train['MasVnrArea'].dtype)
#replace missing values

train['GarageYrBlt']=train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0])

train['MasVnrArea']=train['MasVnrArea'].fillna(train['MasVnrArea'].mode()[0])

train.isnull().sum()   
#check categorical variable (want to dummy code)

categorical_features =train.select_dtypes(include=['object']).columns

categorical_features
#check distribution of sales price

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(12,8))

sns.distplot(train['SalePrice'], color='b')

plt.title('Distribution of Sales Price', fontsize=18)

plt.show()
#log transformance

from scipy.stats import norm

from scipy import stats

train['SalePrice'] = np.log(train['SalePrice'])

train_y=sns.distplot(train['SalePrice'],fit=norm);

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#check correlation

corrmat =train.corr()

f, ax = plt.subplots(figsize=(55, 9))

sns.heatmap(corrmat, vmax=.8, square=True,annot=True,cmap='YlOrRd',linewidths=0.2,annot_kws={'size':10})

plt.title("Correlation",fontsize=20)
#detect outliers

from sklearn import preprocessing

normalized_data = preprocessing.StandardScaler().fit_transform(train)

outliers_rows, outliers_columns = np.where(np.abs(normalized_data)>5)#get more than 5 standered deviation

print (outliers_rows)# outliers  rows
#remove outliers

data1.drop(data1.index[[5,48,51,53,55,58,70,88,113,115,129,153,159,170,182,185,197,198,205,237,249,258,263,267,271,280,313,322,335,346,359,406,426,451,470,495,496,523,542,583,597,625,635,691,705,706,726,729,744,747,764,808,809,810,854,883,889,907,924,941,954,1009,1031,1068,1080,1161,1169,1170,1173,1181,1182,1197,1230,1253,1298,1299,1328,1346,1369,1386,1418,1423,1437,1440,1458]])
#select predictor variable

train_x=train[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']]

print(train_x.shape)

#select response variable

train_y=(train['SalePrice'])
#select variables from test dataset

test_x=data1[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF','BedroomAbvGr','ScreenPorch','PoolArea','MoSold','3SsnPorch','BsmtFinSF2','BsmtHalfBath','MiscVal','Id','LowQualFinSF','YrSold','OverallCond','MSSubClass','EnclosedPorch','KitchenAbvGr']]

print(test_x.shape)

test_x.isnull().sum()
#Replace outliers for testset

test_x['GarageCars']=test_x['GarageCars'].fillna(test_x['GarageCars'].mode()[0])

test_x['GarageArea']=test_x['GarageArea'].fillna(test_x['GarageArea'].mode()[0])

test_x['TotalBsmtSF']=test_x['TotalBsmtSF'].fillna(test_x['TotalBsmtSF'].mode()[0])

test_x['GarageYrBlt']=test_x['GarageYrBlt'].fillna(test_x['GarageYrBlt'].mode()[0])

test_x['MasVnrArea']=test_x['MasVnrArea'].fillna(test_x['MasVnrArea'].mode()[0])

test_x['BsmtFinSF1']=test_x['BsmtFinSF1'].fillna(test_x['BsmtFinSF1'].mode()[0])

test_x['BsmtUnfSF']=test_x['BsmtUnfSF'].fillna(test_x['BsmtUnfSF'].mode()[0])

test_x['BsmtFinSF2']=test_x['BsmtFinSF2'].fillna(test_x['BsmtFinSF2'].mode()[0])

test_x['BsmtHalfBath']=test_x['BsmtHalfBath'].fillna(test_x['BsmtHalfBath'].mode()[0])

test_x['BsmtFullBath']=test_x['BsmtFullBath'].fillna(test_x['BsmtFullBath'].mode()[0])

test_x.isnull().sum() 
# Import the model

from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(train_x,train_y)
#prediction and get results

output = rf.predict(test_x)

print(output)
#print actual values 

output = np.exp(output)

print(output)
# define the data frame for the results

saleprice = pd.DataFrame(output, columns=['SalePrice'])

results = pd.concat([data1['Id'],saleprice['SalePrice']],axis=1)

results.head()
# and write to output

results.to_csv('housepricing_submission.csv', index = False)