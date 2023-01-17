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
#import libraries

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error

from math import sqrt

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

import matplotlib.pyplot as plt

import xgboost as xgb
#set pandas display options for full row col display

pd.set_option('display.max_rows', None, 'display.max_columns', None)
#Load training and test data

train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#Check the type of columns

train_df.info()
#check correlation between columns

train_df.corr()
#seperate SalePrice Column from dataframe

y_train = train_df['SalePrice'].values

train_df = train_df.drop(columns=['SalePrice'])
#merge train and test df

X_df = train_df.append(test_df).reset_index(drop=True)

len(X_df)
#handle missing values

print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
#drop columns PoolQC, Fence, MiscFeature, Alley since they have large missing values

X_df = X_df.drop(columns=['PoolQC', 'Fence', 'MiscFeature', 'Alley'])
#make list of categorical columns and numeric columns

columns_dtypes = list(X_df.dtypes)

columns_names = list(X_df.columns)

no_of_col = X_df.shape[1]

categorical_columns = [columns_names[i] for i in range(no_of_col) if columns_dtypes[i]=='object']

numeric_columns = set(columns_names).difference(set(categorical_columns))

len(categorical_columns),len(numeric_columns)
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
#Since FireplaceQu is missing it can be marked as NA

X_df['FireplaceQu'][X_df['FireplaceQu'].isna()] = 'NA'
X_df.groupby('FireplaceQu')['FireplaceQu'].count()

X_df.head()
#handeling LotFrontage

#If we look at corr table we can se that LotFrontage and LotArea has corr of 0.40

#Hence we can build a regression model to handle those missing values

#find LotFrontage values which are not null

lot_frontage_x_train = X_df[X_df['LotFrontage'].notna()]['LotArea'].values

lot_frontage_y_train = X_df[X_df['LotFrontage'].notna()]['LotFrontage'].values

lot_frontage_x_test = X_df[X_df['LotFrontage'].isna()]['LotArea'].values

lot_frontage_reg = LinearRegression().fit(lot_frontage_x_train.reshape(-1,1), lot_frontage_y_train.reshape(-1,1))
lot_frontage_y_pred = lot_frontage_reg.predict(lot_frontage_x_test.reshape(-1,1))
xg_reg_lot = xgb.XGBRegressor(objective ='reg:linear').fit(lot_frontage_x_train.reshape(-1,1), lot_frontage_y_train.reshape(-1,1))
lot_frontage_y_pred =xg_reg_lot.predict(lot_frontage_x_test.reshape(-1,1))
# Visualising our results

plt.scatter(lot_frontage_x_train, lot_frontage_y_train, color = 'red')

plt.scatter(lot_frontage_x_test, lot_frontage_reg.predict(lot_frontage_x_test.reshape(-1,1)), color = 'green')

plt.plot(lot_frontage_x_train, lot_frontage_reg.predict(lot_frontage_x_train.reshape(-1,1)), color = 'blue')

plt.title('LotArea vs LotFrontage')

plt.xlabel('LotArea(Size in Square Feet)')

plt.ylabel('LotFrontage(Size in Feet)')

plt.show()
# Visualising our results

plt.scatter(lot_frontage_x_train, lot_frontage_y_train, color = 'red')

plt.scatter(lot_frontage_x_test, xg_reg_lot.predict(lot_frontage_x_test.reshape(-1,1)), color = 'green')

plt.plot(lot_frontage_x_train, xg_reg_lot.predict(lot_frontage_x_train.reshape(-1,1)), color = 'blue')

plt.title('LotArea vs LotFrontage')

plt.xlabel('LotArea(Size in Square Feet)')

plt.ylabel('LotFrontage(Size in Feet)')

plt.show()
#This seems a decent fit let check it's r2_score

r2_score(lot_frontage_y_train,lot_frontage_reg.predict(lot_frontage_x_train.reshape(-1,1)))
#This seems a decent fit let check it's r2_score

r2_score(lot_frontage_y_train,xg_reg_lot.predict(lot_frontage_x_train.reshape(-1,1)))
#This is not a good score but better than a assigned mean or median values

med = np.median(lot_frontage_y_train)

mean = np.mean(lot_frontage_y_train)

y_med = np.full((1,len(lot_frontage_y_train)), med)

y_mean = np.full((1,len(lot_frontage_y_train)), mean)

r2_score(lot_frontage_y_train, y_med[0]), r2_score(lot_frontage_y_train, y_mean[0]) 

#Now assgin this predicted values to missing values

X_df[X_df['LotFrontage'].isna()]['LotFrontage'] = lot_frontage_y_pred
X_df.loc[X_df['LotFrontage'].isna(),'LotFrontage'] = lot_frontage_y_pred
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
#Handeling MSZoning missing values

#Check what types of values it has

X_df.groupby('MSZoning')['MSZoning'].count()
#RL is the most frequent one hence assign RL to missing values

X_df['MSZoning'] = X_df['MSZoning'].fillna('RL')
#Handeling Utilities missing values

#Check what types of values it has

X_df.groupby('Utilities')['Utilities'].count()
#AllPub is the most frequent one hence assign AllPub to missing values

X_df['Utilities'] = X_df['Utilities'].fillna('AllPub')
X_df.groupby('Utilities')['Utilities'].count()
#Handeling Exterior1st missing values

#Check what types of values it has

X_df.groupby('Exterior1st')['Exterior1st'].count()
#VinylSd is the most frequent one hence assign VinylSd to missing values

X_df['Exterior1st'] = X_df['Exterior1st'].fillna('VinylSd')
#Handeling Exterior2nd missing values

#Check what types of values it has

X_df.groupby('Exterior2nd')['Exterior2nd'].count()
#VinylSd is the most frequent one hence assign VinylSd to missing values

X_df['Exterior2nd'] = X_df['Exterior2nd'].fillna('VinylSd')
#Handeling MasVnrtype and MasVnrArea

X_df[X_df['MasVnrType'].isna()]
X_df[X_df['MasVnrArea'].isna()]
#Both columns have missing values at same index hence can be filled with None and 0

X_df['MasVnrType'] = X_df['MasVnrType'].fillna('None')

X_df['MasVnrArea'] = X_df['MasVnrArea'].fillna(0.0)
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
#Handeling Exterior1st missing values

#Check what types of values it has

X_df.groupby('BsmtQual')['BsmtQual'].count()
X_df[X_df['BsmtQual'].isna()]
#If we notice in above display that for all below columns 

# BsmtQual         81

# BsmtCond         82

# BsmtExposure     82

# BsmtFinType1     79

# BsmtFinType2     80

#the values are not present 

#at same index means there is no Basement hence we can assign the NA and 0 to those columns

X_df['BsmtQual'] = X_df['BsmtQual'].fillna('NA')

X_df['BsmtCond'] = X_df['BsmtCond'].fillna('NA')

X_df['BsmtExposure'] = X_df['BsmtExposure'].fillna('NA')

X_df['BsmtFinType1'] = X_df['BsmtFinType1'].fillna('NA')

X_df['BsmtFinType2'] = X_df['BsmtFinType2'].fillna('NA')

print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
X_df[X_df['BsmtFinSF1'].isna()]
X_df['BsmtFinSF1'] = X_df['BsmtFinSF1'].fillna(0.0)
X_df['TotalBsmtSF'] = X_df['TotalBsmtSF'].fillna(0.0)
X_df[X_df['BsmtFinSF2'].isna()]
X_df['BsmtFinSF2'] = X_df['BsmtFinSF2'].fillna(0.0)
X_df[X_df['BsmtUnfSF'].isna()]
X_df['BsmtUnfSF'] = X_df['BsmtUnfSF'].fillna(0.0)
X_df[X_df['BsmtFullBath'].isna()]
X_df['BsmtFullBath'] = X_df['BsmtFullBath'].fillna(0.0)

#BsmtHalfBath can a;lso be filled with 0 from above display

X_df['BsmtHalfBath'] = X_df['BsmtHalfBath'].fillna(0.0)
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
X_df.groupby('Electrical')['Electrical'].count()
#SBrkr is the most used hence we fill this

X_df['Electrical'] = X_df['Electrical'].fillna('SBrkr')
X_df[X_df['KitchenQual'].isna()]
X_df.groupby(['KitchenAbvGr', 'KitchenQual'])['KitchenAbvGr'].count()
#The maximum probability that the missing value can be is TA (Typical Average), hence assign TA

X_df['KitchenQual'] = X_df['KitchenQual'].fillna('TA')
X_df.groupby('Functional')['Functional'].count()
X_df['Functional'] = X_df['Functional'].fillna('Typ')
#Handelling 

# GarageType      157

# GarageYrBlt     159

# GarageFinish    159

# GarageQual      159

# GarageCond      159

#lets check which rows has missing value in these columns

X_df[X_df['GarageCond'].isna()]
#Since there is No garage we can easily fill these values with 'NA' except from the column GarageYrBlt

#If we notice closely that This columns Almost has similar year The House was built

#We can check the corr b/w YearBuilt and GarageYrBlt it is 0.825667 which is way higher 

#And causes the multicolliniarity problem hence we need to remove one of these columns

#We can remove GarageYrBlt since this is the only column which has missing data

X_df = X_df.drop(columns=['GarageYrBlt'])
numeric_columns = list(numeric_columns.difference(['GarageYrBlt']))
#Fill rest columns with NA

X_df['GarageType'] = X_df['GarageType'].fillna('NA')

X_df['GarageFinish'] = X_df['GarageFinish'].fillna('NA')

X_df['GarageQual'] = X_df['GarageQual'].fillna('NA')

X_df['GarageCond'] = X_df['GarageCond'].fillna('NA')
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
X_df[X_df['GarageCars'].isna()]
#Since others columns are NA means there is no garage then GarageCars and GarageArea can be filled with 0.0

X_df['GarageArea'] = X_df['GarageArea'].fillna(0.0)

X_df['GarageCars'] = X_df['GarageCars'].fillna(0.0)
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
X_df[X_df['SaleType'].isna()]
X_df.groupby('SaleType')['SaleType'].count()
X_df.groupby(['SaleCondition', 'SaleType'])['SaleType'].count()
#From above analysis we can see that whenever the SaleCondition is Normal the chances are that SaleType is going to be WD

X_df['SaleType'] = X_df['SaleType'].fillna('WD')
print(X_df.isnull().sum()[X_df.isnull().sum()!=0])
#Hence all missing values are handelled and now we can proceed to data transformation part

X_df.corr().unstack().sort_values(ascending=False).drop_duplicates()
#From above results it is visible that 6 cols are highly corelated that can cause multicollinearity problem

#Hence we will  remove 3 of them from our dataset

multi_collinear_columns = ['GarageArea','TotRmsAbvGrd', '1stFlrSF']

numeric_columns = list(set(numeric_columns).difference(set(multi_collinear_columns)))

categorical_columns = list(set(categorical_columns).difference(set(multi_collinear_columns)))

X_df = X_df.drop(columns=multi_collinear_columns)
X_df.head(2)
numeric_columns
X_df.groupby('MSSubClass')['MSSubClass'].count()
#MSSubClass, OverallQual, OverallCond columns cantains numeric values but is of categorical type hence remove it from numeric columns list and add to other one

numeric_columns = list(set(numeric_columns).difference(set(['MSSubClass', 'OverallQual','OverallCond'])))

categorical_columns = categorical_columns + ['MSSubClass', 'OverallQual', 'OverallCond']
numeric_columns.sort()

categorical_columns.sort()
numeric_columns
X_df[categorical_columns] = X_df[categorical_columns].apply(LabelEncoder().fit_transform)
X_df.head()
X_df = pd.get_dummies(X_df, columns=categorical_columns)
X_df.head()
X_df.shape
#divide the X_df into train and test data

x_train = X_df[:len(train_df)]

x_test = X_df[len(train_df):]

x_train.shape, x_test.shape
x_train = x_train.drop(columns=['Id'])

x_train.tail()
x_test_Id = x_test['Id']

x_test = x_test.drop(columns=['Id'])

print(type(x_test_Id))

x_test.tail()
#split train data into train : val ratio of 90 : 10

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.99, random_state=42)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
lr = LinearRegression(n_jobs=-1).fit(x_train, y_train)
lr.score(x_train,y_train), np.sqrt(np.mean((np.log(y_train)-np.log(lr.predict(x_train)))**2))
lr.score(x_val,y_val), np.sqrt(np.mean((np.log(y_val)-np.log(lr.predict(x_val)))**2))
lr.predict(x_test)
test_pred = pd.DataFrame(data={'Id': x_test_Id.T, 'SalePrice':lr.predict(x_test)})
test_pred.head()
xg_reg = xgb.XGBRegressor(objective ='reg:linear')
xg_reg.fit(x_train,y_train)
np.sqrt(np.mean((np.log(y_train)-np.log(xg_reg.predict(x_train)))**2))
np.sqrt(np.mean((np.log(y_val)-np.log(xg_reg.predict(x_val)))**2))
test_pred = pd.DataFrame(data={'Id': x_test_Id.T, 'SalePrice':xg_reg.predict(x_test)})
test_pred.to_csv('Submission.csv', index=False)
# X_scaler = StandardScaler()

# x_train = pd.DataFrame(X_scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)

# x_val = pd.DataFrame(X_scaler.transform(x_val), index=x_val.index, columns=x_val.columns)

# x_test = pd.DataFrame(X_scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
# x_train.head()
# x_val.head()
# x_test.head()
# Y_scaler = StandardScaler()

# y_train = Y_scaler.fit_transform(y_train.reshape(-1,1))
# y_train
# r2_score(Y_scaler.inverse_transform(y_train.reshape(-1,1)), 

#          Y_scaler.inverse_transform(lr.predict(x_train)))
# svr = SVR(kernel='rbf').fit(x_train, y_train)
# r2_score(Y_scaler.inverse_transform(y_train.reshape(-1,1)), 

#          Y_scaler.inverse_transform(svr.predict(x_train)))
# r2_score(y_val, 

#          Y_scaler.inverse_transform(svr.predict(x_val)))