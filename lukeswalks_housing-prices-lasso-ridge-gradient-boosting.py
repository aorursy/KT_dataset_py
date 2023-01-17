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

import xgboost as xgb

from sklearn.linear_model import LinearRegression

from sklearn import model_selection

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

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

print(train['SalePrice'])

test = pd.read_csv("/kaggle/input/test.csv")
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



sns.pairplot(traind[['YearBuilt','OverallQual','SalePrice','LotFrontage','LotArea']])



#Traind["Embarked"].fillna(Traind['Embarked'].value_counts().idxmax(), inplace=True)

Nanlist = ((traind.isnull().sum()/(traind.shape[0]))*100).sort_values()



#Transform YearSold & YearRemodAdd

traind['YrSold'] = traind['YrSold']==traind['YearBuilt']

traind['YearRemodAdd'] = traind['YearRemodAdd']==traind['YearBuilt']



#Dummy Generation

traind['GarageYrBlt'][traind['GarageYrBlt']=='NA']='0'

traind['GarageYrBlt']= traind['GarageYrBlt'].astype('int64')

dummycolumns = list(traind.drop(columns='Id',).select_dtypes(include=['object','category','bool']).columns) 

dummycolumns.append('MSSubClass')

for i in (traind.columns):

    if (traind.dtypes[i]=='object'):

        traind[i] = traind[i].astype('category')

traind = pd.get_dummies(traind,columns=dummycolumns,drop_first=True)





# Log transform of the skewed numerical features to lessen impact of outliers

# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

#skewness = traind.select_dtypes(include=['float64','int64']).apply(lambda x: skew(x))

#skewness = skewness[abs(skewness) > 0.5]

#print(str(skewness.shape[0]) + " skewed numerical features to log transform")

#skewed_features = list(skewness.index)

#print(type(skewed_features))

traind[traind.drop(columns=['Id','SalePrice']).select_dtypes(include=['float64','int64']).columns] = np.log1p(traind[traind.drop(columns=['Id','SalePrice']).select_dtypes(include=['float64','int64']).columns])

from sklearn.model_selection import train_test_split

X = traind.drop(

    columns=['SalePrice','Id'])

y = traind['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#Modeling



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

CVR = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]



#Ridge Regression



MR=Ridge(alpha=10)

MR.fit(X_train, y_train)



CVR = pd.Series(CVR, index = alphas)

CVR.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")



print('Ridge:',CVR.min())



#lasso Regression



ML = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)



print('Lasso:',rmse_cv(ML).mean())



#Gradiant Boosting



dtrain = xgb.DMatrix(X_train, label = y_train)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)



#the params were tuned using xgb.cv



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(X_train, y_train)



xgb_preds = np.expm1(model_xgb.predict(X_test))



print('Gradiant:',rmse_cv(model_xgb).mean())
kfold = model_selection.KFold(n_splits=5, shuffle = True)

cv_mse_resultsMR = model_selection.cross_val_score(MR, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

cv_rmse_resultsMR = np.sqrt(-cv_mse_resultsMR)

print('Ridge:',cv_rmse_resultsMR.mean())

cv_mse_resultsML = model_selection.cross_val_score(MR, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

cv_rmse_resultsML = np.sqrt(-cv_mse_resultsML)

print('Lasso:',cv_rmse_resultsML.mean())
#Check Test Data NA

#Nanlist = ((test.isnull().sum()/(test.shape[0]))*100).sort_values()

#print(Nanlist[Nanlist!=0])

#print(Nanlist[Nanlist!=0].axes)



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



#Transform YearSold & YearRemodAdd

testd['YrSold'] = testd['YrSold']==testd['YearBuilt']

testd['YearRemodAdd'] = testd['YearRemodAdd']==testd['YearBuilt']



#Dummy Generation

dummycolumns = list(testd.drop(columns='Id').select_dtypes(include=['object','category','bool']).columns) 

dummycolumns.append('MSSubClass')

for i in (testd.columns):

    if (testd.dtypes[i]=='object'):

        testd[i] = testd[i].astype('category')

testd = pd.get_dummies(testd,columns=dummycolumns,drop_first=True)



testd[testd.drop(columns='Id').select_dtypes(include=['float64','int64']).columns] = np.log1p(testd[testd.drop(columns='Id').select_dtypes(include=['float64','int64']).columns])





# Get missing columns in the training test

missing_cols = set(traind.drop(columns='SalePrice').columns) - set(testd.columns)

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    testd[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

testd = testd[traind.drop(columns='SalePrice').columns]

print(np.expm1(ML.predict(testd.drop(columns=['Id']))))

testd['SalePrice'] = np.expm1(ML.predict(testd.drop(columns=['Id'])))

submission=testd[['Id','SalePrice']]

submission.to_csv("submission_LassoRegression.csv", index=False)

submission.tail()



testd['SalePrice'] = np.expm1(MR.predict(testd.drop(columns=['Id','SalePrice'])))

submission=testd[['Id','SalePrice']]

submission.to_csv("submission_RidgeRegression.csv", index=False)

submission.tail()



testd['SalePrice'] = np.expm1(model_xgb.predict(testd.drop(columns=['Id','SalePrice'])))

submission=testd[['Id','SalePrice']]

submission.to_csv("submission_XGBoostRegression.csv", index=False)

submission.tail()