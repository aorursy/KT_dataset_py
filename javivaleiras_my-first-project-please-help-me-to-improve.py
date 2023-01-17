# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path_train = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"

path_test = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"

train = pd.read_csv(path_train)

test = pd.read_csv(path_test)
print("Train: ",train.shape)

print("Test: ",test.shape)
numeric = train.select_dtypes(exclude='object')

print("\nNumber of numeric features : ",(len(numeric.axes[1])))

print("\n", numeric.axes[1])
# Isolate the numeric features and check his relevance



num_corr = numeric.corr()

table = num_corr['SalePrice'].sort_values(ascending=False).to_frame()

cm = sns.light_palette("green", as_cmap=True)

tb = table.style.background_gradient(cmap=cm)

tb

# Isolate the categorical data

categorical = train.select_dtypes(include='object')

categorical.head()
print("\nNumber of categorical features : ",(len(categorical.axes[1])))

print("\n", categorical.axes[1])
# I will drop features with 80% missing values

train_d = train.dropna(thresh=len(train)*0.8, axis=1)

droppedF = []

print("We dropped the next features: ")

for x in train.axes[1]:

    if(x not in train_d.axes[1]):

        droppedF.append(x)



print(droppedF)
# I will drop this features also in the test dataset

test_d = test.drop(droppedF,axis=1)

print(train_d.shape, test_d.shape)

sh_train = train_d.shape

# I will also mix both (test and train) to do all at the same time

c1 = pd.concat((train_d, test_d), sort=False).reset_index(drop=True)



print("Total size is :",c1.shape)
# Now I will detect and study what to do with missing values

c1_NA = (c1.isnull().sum() / len(c1)*100)

c1_NA = c1_NA.drop(c1_NA[c1_NA == 0].index).sort_values(ascending = False)

plt.figure(figsize=(14,7))

chart = sns.barplot(x=c1_NA.index , y = c1_NA)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

plt.ylabel("Missing Value Percentage")

plt.xlabel("Features")

plt.title("Missing Value´s Percentage by Feature")
# First of all split numerical NA and categorical NA data

NA=c1[c1_NA.index.to_list()]

catNA=NA.select_dtypes(include='object')

numNA=NA.select_dtypes(exclude='object')
# Start with the numerical ones

numNA
# Fill with 0 and Garage Year Built fill it with the median to not disturb the data. Lot frontage also with the median

Fillw0 = ['MasVnrArea','BsmtFullBath','BsmtFinSF1','BsmtFinSF1','TotalBsmtSF','GarageCars','GarageArea','BsmtFinSF2','BsmtHalfBath','BsmtUnfSF']

c1[Fillw0] = c1[Fillw0].fillna(0)

c1['GarageYrBlt'] = c1.GarageYrBlt.fillna(c1.GarageYrBlt.median())

c1['LotFrontage'] = c1.LotFrontage.fillna(c1.GarageYrBlt.median())

# Take a look to the cat features

catNA
# I will use the forward fill method to the ones with almost no null values.

# I will fill the rest with NA for the same reason I filled numerical ones with 0

f_forward = ['Electrical', 'SaleType', 'KitchenQual', 'Exterior1st',

             'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']

for col in c1[f_forward]:

    c1[col] = c1[col].fillna(method='ffill')

catNA.drop(f_forward,axis=1)



c1['has_garage'] = c1['GarageQual'].isnull().astype(int)

c1['has_Bsmt'] = c1['BsmtCond'].isnull().astype(int)

c1['has_MasVnr'] = c1['MasVnrType'].isnull().astype(int)



for col in catNA:

    c1[col] = c1[col].fillna("NA")
# Now we will encode the categorical features. For that I will use one hot encoding and I will also add has_bathroom, has_garage, and has_MasVnr



cb=pd.get_dummies(c1)

print("the shape of the original dataset",c1.shape)

print("the shape of the encoded dataset",cb.shape)

print("We have ",cb.shape[1]- c1.shape[1], 'new encoded features')

# Split again train and test

train = cb[:sh_train[0]] #sh_train is the shape of train_d

test = cb[sh_train[0]:]

test = test.drop('SalePrice',axis=1)# Remove SalePrice from the test

print(train.shape, test.shape)
#Detect and remove the outliers in the most significant features

fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)







ax1 = plt.subplot2grid((3,2),(0,1))

plt.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)







ax1 = plt.subplot2grid((3,2),(1,0))

plt.scatter(x = train['MasVnrArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('MasVnrArea', fontsize=13)







ax1 = plt.subplot2grid((3,2),(1,1))

plt.scatter(x = train['GarageArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)









ax1 = plt.subplot2grid((3,2),(2,0))

plt.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)







ax1 = plt.subplot2grid((3,2),(2,1))

plt.scatter(x = train['1stFlrSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('1stFlrSF', fontsize=13)

plt.show()

 



 



#Remove the detected outliers

train = train.drop(train[(train['GrLivArea'] > 4000)&(train['SalePrice'] < 250000)].index)

train = train.drop(train[(train['OverallQual'] == 10)&(train['SalePrice'] < 210000)].index)

train = train.drop(train[(train['MasVnrArea'] > 1400)&(train['SalePrice'] < 300000)].index)

train = train.drop(train[(train['GarageArea'] > 1200)&(train['SalePrice'] < 300000)].index)

train = train.drop(train[(train['TotalBsmtSF'] > 5000)&(train['SalePrice'] < 250000)].index)

train = train.drop(train[(train['1stFlrSF'] > 4000)&(train['SalePrice'] < 250000)].index)



fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot2grid((3,2),(0,0))

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)







ax1 = plt.subplot2grid((3,2),(0,1))

plt.scatter(x = train['OverallQual'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('OverallQual', fontsize=13)







ax1 = plt.subplot2grid((3,2),(1,0))

plt.scatter(x = train['MasVnrArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('MasVnrArea', fontsize=13)







ax1 = plt.subplot2grid((3,2),(1,1))

plt.scatter(x = train['GarageArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GarageArea', fontsize=13)









ax1 = plt.subplot2grid((3,2),(2,0))

plt.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)







ax1 = plt.subplot2grid((3,2),(2,1))

plt.scatter(x = train['1stFlrSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('1stFlrSF', fontsize=13)

plt.show()

 
# Study and correct (if it is necessary) the skewness and kurtosis.

print("Skewness:", train['SalePrice'].skew())

print("Kurtosis: ",train['SalePrice'].kurt())



plt.hist(train.SalePrice, bins=10, color='mediumpurple',alpha=0.5)

plt.show()
# Apply log1p to correct this  distribution

train["SalePrice"] = np.log1p(train["SalePrice"])

print("Skewness:", train['SalePrice'].skew())

print("Kurtosis: ",train['SalePrice'].kurt())



plt.hist(train.SalePrice, bins=10, color='mediumpurple',alpha=0.5)

plt.show()
# Finally we will apply Linear Regression and XGBoost

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



# First plit X and y

y = train.SalePrice

x = train.drop('SalePrice',axis=1)





# Split the them into train and test to evaluate both models

X_train, X_valid, y_train, y_valid = train_test_split(x, y,test_size = 0.2, random_state=0)



# Define the model

XGB = XGBRegressor(n_estimators=10,learning_rate=0.05)

lr = LinearRegression()

# Fit the model

XGB.fit(X_train,y_train) 

lr.fit(X_train,y_train) 

# Get predictions

pred_XGB = XGB.predict(X_valid)

pred_lr = lr.predict(X_valid)

# Calculate MAE

mae_XGB =  mean_absolute_error(pred_XGB,y_valid)

mae_lr =  mean_absolute_error(pred_lr,y_valid)

# Uncomment to print MAE

print("Mean Absolute Error XGB:" , mae_XGB)

print("Mean Absolute Error lr:" , mae_lr)

# Best Alpha for Ridge Regression

import sklearn.model_selection as ms

import sklearn.model_selection as GridSearchCV #Cross-Validation

from sklearn.linear_model import Ridge



ridge=Ridge()

parameters= {'alpha':[x for x in range(1,101)]}



ridge_reg=ms.GridSearchCV(ridge, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)

ridge_reg.fit(X_train,y_train)

print("The best value of Alpha is: ",ridge_reg.best_params_)



# Ridge Error

ridge_mod=Ridge(alpha=1)

ridge_mod.fit(X_train,y_train)

pred_rr=ridge_mod.predict(X_valid)

mae_rr =  mean_absolute_error(pred_rr,y_valid)



print("Mean absolute error Ridge Regression: ",mae_rr )

# Best alpha Lasso Regression

from sklearn.linear_model import Lasso



parameters= {'alpha':[0.0001,0.0009,0.001,0.002,0.003,0.01,0.1,1,10,100]}



lasso=Lasso()

lasso_reg=ms.GridSearchCV(lasso, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)

lasso_reg.fit(X_train,y_train)



print('The best value of Alpha is: ',lasso_reg.best_params_)
# Lasso Error

lasso_mod=Ridge(alpha=0.0001)

lasso_mod.fit(X_train,y_train)

pred_la=lasso_mod.predict(X_valid)

mae_la =  mean_absolute_error(pred_la,y_valid)



print("Mean absolute error Lasso Regression: ",mae_la)

# Based on that we will use Linear Regression

Ridge2 = Ridge(alpha=16)

# Fit the model

Ridge2.fit(x,y)

# Get predictions

pred_ridge2 = ridge2.predict(test)

final = np.expm1(pred_ridge2) # Undo the log1p

path_f = "/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv"

f = pd.read_csv(path_train)

output = pd.DataFrame({'Id': test.Id,

                       'SalePrice': final})

output.to_csv('sample_submission.csv', index=False)