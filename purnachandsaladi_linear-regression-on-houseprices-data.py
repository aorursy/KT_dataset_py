import numpy as np

import pandas as pd

import os

HP = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

HP.describe()
print(HP.columns)
HP.head()
HP.shape
#finding invalid values

HP.isnull().sum()
print(HP.LotFrontage.describe())
print(np.array(HP.LotFrontage).shape)

#print(np.array(HP.LotFrontage).reshape(-1,1).shape)
#replacing null values with mean using simple imputer

#from sklearn.impute import SimpleImputer

#SI = SimpleImputer(strategy='mean')

#HP.LotFrontage = SI.fit_transform(np.array(HP.LotFrontage).reshape(-1,1))

HP.LotFrontage.fillna(HP.LotFrontage.mean(),inplace=True)
HP.LotFrontage.isnull().sum()
HP.isnull().sum().sort_values(ascending=False)
print(HP.PoolQC .describe())

print(HP.MiscFeature.describe())

print(HP.Alley.describe())

print(HP.Fence.describe())

print(HP.FireplaceQu.describe())
numeric_features = HP.select_dtypes(include=[np.number])

print(numeric_features.dtypes)
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n')
print(HP.OverallQual.isnull().sum())    

print(HP.GrLivArea.isnull().sum())      

print(HP.GarageCars.isnull().sum())  

print(HP.TotalBsmtSF.isnull().sum())    

print(HP.FullBath.isnull().sum())      

print(HP.TotRmsAbvGrd.isnull().sum())  

print(HP.YearBuilt.isnull().sum())

print(HP.YearRemodAdd.isnull().sum())
# apply anova f-test on Neighbourhood and sale price(continous-category)

import statsmodels.api as sm

from statsmodels.formula.api import ols

ftest = ols("SalePrice ~ Neighborhood", data=HP).fit()

anova = sm.stats.anova_lm(ftest)

print(anova)
ContColumns = list(['GrLivArea','OverallQual','GarageCars','GarageArea','TotalBsmtSF',    

                    'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd'])

#defining dataframe for independent variables taken

Independent = HP[ContColumns]

print(Independent.shape)

Independent.describe()
# defining dataframe for dependent variable

Dependent = HP['SalePrice']

Dependent.describe()
# build linear model using sklearn

from sklearn.linear_model import LinearRegression

# initialize linear model

lm = LinearRegression(fit_intercept=True, normalize=False)

# apply linear model

model = lm.fit(Independent, Dependent)

print (lm.coef_)

print (lm.intercept_)
print (model.score(Independent, Dependent))
HPT= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print ( HPT.shape)
print(HPT.OverallQual.isnull().sum())    

print(HPT.GrLivArea.isnull().sum())      

print(HPT.GarageCars.isnull().sum())  

print(HPT.TotalBsmtSF.isnull().sum())    

print(HPT.FullBath.isnull().sum())      

print(HPT.TotRmsAbvGrd.isnull().sum())  

print(HPT.YearBuilt.isnull().sum())

print(HPT.YearRemodAdd.isnull().sum())
HPT.GarageCars.fillna(HPT.GarageCars.mean(),inplace=True)

HPT.GarageArea.fillna(HPT.GarageArea.mean(),inplace=True)

HPT.TotalBsmtSF.fillna(HPT.TotalBsmtSF.mean(),inplace=True)
FINAL=HPT[ContColumns]

predict_prices=model.predict(FINAL)

print(predict_prices)

predict_prices.shape

#print(np.array(predict_prices).reshape(-1,1).shape)
my_submission = pd.DataFrame({'Id': HPT.Id, 'SalePrice': predict_prices})

my_submission.to_csv('LR ON HOUSEPRICES USING CONT DATA.csv', index=False)