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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn import model_selection

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.metrics import classification_report,confusion_matrix

import statsmodels.api as sm

from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import skew

%matplotlib inline



# Load the data

train = pd.read_csv("/kaggle/input/train.csv")

train.SalePrice = np.log1p(train.SalePrice)

print(train.columns)

test = pd.read_csv("/kaggle/input/test.csv")
#Check for null/nan values 

#Nanlist = ((train.isnull().sum()/(train.shape[0]))*100).sort_values()

#print(Nanlist[Nanlist!=0])

#print(Nanlist[Nanlist!=0].axes)

#print(train['MasVnrType'].value_counts(dropna=False))

#print(train['MasVnrArea'].value_counts(dropna=False))

#train['SalePrice'].describe()

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);


#saleprice correlation matrix

k = 20 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

plt.figure(figsize=(12,12))

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#Create Seperate Variable for Testing

traind = train



#overfit removal

overfit = []

for i in traind.columns:

        counts = traind[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(traind) * 100 > 99.9: # the threshold is set at 99.9%

            overfit.append(i)

overfit = list(overfit)

print(overfit)

traind.drop(overfit,axis=1,inplace=True)



#Replace or drop NaNs

LotFrontageMeans = dict(zip(traind.MSZoning.unique(),traind.groupby(['MSZoning']).mean()['LotFrontage']))

traind=traind.set_index('MSZoning')

traind['LotFrontage'].fillna(value=LotFrontageMeans,axis=0, inplace=True)

traind=traind.reset_index()

traind["GarageQual"].fillna(value="NA", inplace=True)

traind["GarageFinish"].fillna(value="NA", inplace=True)

traind["GarageYrBlt"].fillna(value="NA", inplace=True)

traind["GarageType"].fillna(value="NA", inplace=True)

traind["GarageCond"].fillna(value="NA", inplace=True)

traind["BsmtQual"].fillna(value="NA", inplace=True)

traind["BsmtCond"].fillna(value="NA", inplace=True)

traind["BsmtFinType1"].fillna(value="NA", inplace=True)

traind["BsmtFinType2"].fillna(value="NA", inplace=True)

traind["BsmtExposure"].fillna(value="NA", inplace=True)

traind["Alley"].fillna(value="NA", inplace=True)

traind["PoolQC"].fillna(value="NA", inplace=True)

traind["MiscFeature"].fillna(value="NA", inplace=True)

traind["Fence"].fillna(value="NA", inplace=True)

traind["FireplaceQu"].fillna(value="NA", inplace=True)

traind["MasVnrType"].fillna(value="NA", inplace=True)

traind["Electrical"].fillna(traind['Electrical'].value_counts().idxmax(), inplace=True)

traind["MasVnrArea"].fillna(traind["MasVnrArea"].mean(skipna=True), inplace=True)



#Traind["Embarked"].fillna(Traind['Embarked'].value_counts().idxmax(), inplace=True)

Nanlist = ((traind.isnull().sum()/(traind.shape[0]))*100).sort_values()



#Dummy Generation

traind['GarageYrBlt'][traind['GarageYrBlt']=='NA']='0'

traind['GarageYrBlt']= traind['GarageYrBlt'].astype('int64')

dummycolumns = list(traind.drop(columns='Id',).select_dtypes(include=['object','category']).columns) 

dummycolumns.append('MSSubClass')

for i in (traind.columns):

    if (traind.dtypes[i]=='object'):

        traind[i] = traind[i].astype('category')

traind = pd.get_dummies(traind,columns=dummycolumns,drop_first=True)





# Log transform of the skewed numerical features to lessen impact of outliers

# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

skewness = traind.select_dtypes(include=['float64','int64']).apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = list(skewness.index)

print(type(skewed_features))

traind[skewed_features] = np.log1p(traind[skewed_features])

from sklearn.model_selection import train_test_split

X = traind.drop(columns=['SalePrice','Id'])

y = traind['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#1000 Tree

rfc1000 = RandomForestRegressor(n_estimators=1000)

rfc1000.fit(X_train,y_train)

predictions = rfc1000.predict(X_test)



# Performance metrics

errors = abs(predictions - y_test)

print('Metrics for Random Forest Trained on Original Data')

print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)

# Calculate and display accuracy

accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')



kfold = model_selection.KFold(n_splits=5, shuffle = True, random_state=7)

cv_mse_results = model_selection.cross_val_score(rfc1000, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

cv_rmse_results = np.sqrt(-cv_mse_results)

print(cv_rmse_results.mean())
#now for test data...

Nanlist = ((test.isnull().sum()/(test.shape[0]))*100).sort_values()

print(Nanlist[Nanlist!=0])

print(Nanlist[Nanlist!=0].axes)
testd=test





#overfit removal

testd.drop(overfit,axis=1,inplace=True)



#Replace or drop NaNs

testd["Alley"].fillna(value="NA", inplace=True)

testd["PoolQC"].fillna(value="NA", inplace=True)

testd["MiscFeature"].fillna(value="NA", inplace=True)

testd["Fence"].fillna(value="NA", inplace=True)

testd["FireplaceQu"].fillna(value="NA", inplace=True)

testd["Functional"].fillna(value="Typ", inplace=True)

BsmtHalfBathMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['BsmtHalfBath'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['BsmtHalfBath'].fillna(value=BsmtHalfBathMeans,axis=0, inplace=True)

testd=testd.reset_index()

BsmtFullBathMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['BsmtFullBath'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['BsmtFullBath'].fillna(value=BsmtFullBathMeans,axis=0, inplace=True)

testd=testd.reset_index()

BsmtFinType2Means = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['BsmtFinType2'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['BsmtFinType2'].fillna(value=BsmtFinType2Means,axis=0, inplace=True)

testd=testd.reset_index()

GarageCarsMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['GarageCars'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['GarageCars'].fillna(value=GarageCarsMeans,axis=0, inplace=True)

testd=testd.reset_index()

BsmtFinSF1Means = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['BsmtFinSF1'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['BsmtFinSF1'].fillna(value=BsmtFinSF1Means,axis=0, inplace=True)

testd=testd.reset_index()

BsmtFinSF2Means = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['BsmtFinSF2'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['BsmtFinSF2'].fillna(value=BsmtFinSF2Means,axis=0, inplace=True)

testd=testd.reset_index()

KitchenQualMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['KitchenQual'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['KitchenQual'].fillna(value=KitchenQualMeans,axis=0, inplace=True)

testd=testd.reset_index()

Exterior1stMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['Exterior1st'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['Exterior1st'].fillna(value=Exterior1stMeans,axis=0, inplace=True)

testd=testd.reset_index()

Exterior2ndMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['Exterior2nd'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['Exterior2nd'].fillna(value=Exterior2ndMeans,axis=0, inplace=True)

testd=testd.reset_index()

SaleTypeMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['SaleType'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['SaleType'].fillna(value=SaleTypeMeans,axis=0, inplace=True)

testd=testd.reset_index()

BsmtUnfSFMeans = dict(zip(testd.MSSubClass.unique(),testd.groupby(['MSSubClass']).mean()['BsmtUnfSF']))

testd=testd.set_index('MSSubClass')

testd['BsmtUnfSF'].fillna(value=BsmtUnfSFMeans,axis=0, inplace=True)

testd=testd.reset_index()

GarageAreaMeans = dict(zip(testd.MSSubClass.unique(),testd.groupby(['MSSubClass']).mean()['GarageArea']))

testd=testd.set_index('MSSubClass')

testd['GarageArea'].fillna(value=GarageAreaMeans,axis=0, inplace=True)

testd=testd.reset_index()

TotalBsmtSFMeans = dict(zip(testd.MSSubClass.unique(),testd.groupby(['MSSubClass']).mean()['TotalBsmtSF']))

testd=testd.set_index('MSSubClass')

testd['TotalBsmtSF'].fillna(value=TotalBsmtSFMeans,axis=0, inplace=True)

testd=testd.reset_index()

MSZoningMeans = dict(zip(testd.MSSubClass.unique(),pd.DataFrame(list(testd.groupby(['MSSubClass'])['MSZoning'].value_counts(dropna=True).sort_values().groupby(level=0).tail(1).sort_index().to_dict()))[1]))

testd=testd.set_index('MSSubClass')

testd['MSZoning'].fillna(value=MSZoningMeans,axis=0, inplace=True)

testd=testd.reset_index()

LotFrontageMeans = dict(zip(testd.MSZoning.unique(),testd.groupby(['MSZoning']).mean()['LotFrontage']))

testd=testd.set_index('MSZoning')

testd['LotFrontage'].fillna(value=LotFrontageMeans,axis=0, inplace=True)

testd=testd.reset_index()

testd["GarageQual"].fillna(value="NA", inplace=True)

testd["GarageFinish"].fillna(value="NA", inplace=True)

testd["GarageYrBlt"].fillna(value="NA", inplace=True)

testd["GarageType"].fillna(value="NA", inplace=True)

testd["GarageCond"].fillna(value="NA", inplace=True)

testd["BsmtQual"].fillna(value="NA", inplace=True)

testd["BsmtCond"].fillna(value="NA", inplace=True)

testd["BsmtFinType1"].fillna(value="NA", inplace=True)

testd["BsmtFinType2"].fillna(value="NA", inplace=True)

testd["BsmtExposure"].fillna(value="NA", inplace=True)

testd["MasVnrType"].fillna(value="NA", inplace=True)

testd["Electrical"].fillna(testd['Electrical'].value_counts().idxmax(), inplace=True)

testd["MasVnrArea"].fillna(testd["MasVnrArea"].mean(skipna=True), inplace=True)



Nanlist = ((testd.isnull().sum()/(testd.shape[0]))*100).sort_values()

print(Nanlist[Nanlist!=0])



#Dummy Generation

dummycolumns = list(testd.drop(columns='Id').select_dtypes(include=['object','category']).columns) 

dummycolumns.append('MSSubClass')

for i in (testd.columns):

    if (testd.dtypes[i]=='object'):

        testd[i] = testd[i].astype('category')

testd = pd.get_dummies(testd,columns=dummycolumns,drop_first=True)







# Get missing columns in the training test

missing_cols = set(traind.drop(columns='SalePrice').columns) - set(testd.columns)

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    testd[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

testd = testd[traind.drop(columns='SalePrice').columns]



testd['SalePrice'] = np.expm1(rfc1000.predict(testd.drop(columns='Id')))

submission=testd[['Id','SalePrice']]

submission.to_csv("submission.csv", index=False)

submission.tail()