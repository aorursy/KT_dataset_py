# Importing libraries necessary for the study

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# reading the dataset

HouseAu = pd.read_csv('../input/train.csv')
# Head

HouseAu.head()
#transposing the Data table to view the variables better

HouseAu.head(12).transpose()
# summary of the dataset: 1460 rows, 81 columns

print(HouseAu.info())
# Let's check the dimensions of the dataframe

HouseAu.shape
#Cleaning up variable Alley (Replacing NA => No Alley Access)

HouseAu['Alley'].replace({np.nan:'No Alley Access'},inplace=True)

100*(HouseAu['Alley'].value_counts()/HouseAu['Alley'].count())
# As 93.8% of Alley is "No Alley access" it can be considered as a single value attribute and hence can be dropped

HouseAu=HouseAu.drop(['Alley'],axis=1)
#Cleaning up variable BsmtQual (Replacing NA => No Basement) to reduce the features

HouseAu['BsmtQual'].replace({np.nan:'No Basement'},inplace=True)

print(100*(HouseAu['BsmtQual'].value_counts()/HouseAu['BsmtQual'].count()))

# Three levels can be combined as "Others" (Fa, No BAsement, Ex)

HouseAu['BsmtQual'].replace({'Fa':'Others'},inplace=True)

HouseAu['BsmtQual'].replace({'Ex':'Others'},inplace=True)

HouseAu['BsmtQual'].replace({'No Basement':'Others'},inplace=True)

print(100*(HouseAu['BsmtQual'].value_counts()/HouseAu['BsmtQual'].count()))

#Cleaning up variable BsmtCond (Replacing NA => No Basement)

HouseAu['BsmtCond'].replace({np.nan:'No Basement'},inplace=True)

100*(HouseAu['BsmtCond'].value_counts()/HouseAu['BsmtCond'].count())

# Three levels of fair/good quality can be combined as OK 

HouseAu['BsmtCond'].replace({'Fa':'OK'},inplace=True)

HouseAu['BsmtCond'].replace({'TA':'OK'},inplace=True)

HouseAu['BsmtCond'].replace({'Gd':'OK'},inplace=True)

# Two levels of poor quality can be combined as NOK (Po, No Basement)

HouseAu['BsmtCond'].replace({'Po':'NOK'},inplace=True)

HouseAu['BsmtCond'].replace({'No Basement':'NOK'},inplace=True)

print(100*(HouseAu['BsmtCond'].value_counts()/HouseAu['BsmtCond'].count()))

#Can be considered as single value and can be dropped from dataset

HouseAu=HouseAu.drop(['BsmtCond'],axis=1)
#Cleaning up variable BsmtExposure (Replacing NA => No Basement)

HouseAu['BsmtExposure'].replace({np.nan:'No Basement'},inplace=True)

100*(HouseAu['BsmtExposure'].value_counts()/HouseAu['BsmtExposure'].count())
#Cleaning up variable BsmtFinType1 (Replacing NA => No Basement)

HouseAu['BsmtFinType1'].replace({np.nan:'No Basement'},inplace=True)

100*(HouseAu['BsmtFinType1'].value_counts()/HouseAu['BsmtFinType1'].count())
#Cleaning up variable BsmtFinType2 (Replacing NA => No Basement)

HouseAu['BsmtFinType2'].replace({np.nan:'No Basement'},inplace=True)

100*(HouseAu['BsmtFinType2'].value_counts()/HouseAu['BsmtFinType2'].count())
#Taking a deep dive into the Basement related attributes to understand the correlations

HouseAu_Basement=HouseAu[['BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]

HouseAu_Basement.head()

HouseAu_Basement.info()

##HouseAu['BsmtQual']=HouseAu['BsmtQual'].values.astype(np.int64)
# pairwise scatter plot to explore Basement attributes



plt.figure(figsize=(20, 10))

sns.pairplot(HouseAu_Basement)

plt.show()
#Dropping of correlated variables and keeping only TotalBsmtSF as this is the key one remaining are related to it.

HouseAu=HouseAu.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis=1)
#Plotting the categorical variables related to Basement to find which ones have correlation and can be dropped

plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)

sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,2)

sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,3)

sns.boxplot(x = 'BsmtFinType1',y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,4)

sns.boxplot(x = 'BsmtFinType2',y = 'SalePrice', data = HouseAu)
#Dropping variables BsmtFinType1 and BsmtFinType2 as two do not seem to have a strong influence on sale price

HouseAu=HouseAu.drop(['BsmtFinType1','BsmtFinType2'],axis=1)
#Cleaning up variable FireplaceQu (Replacing NA => No Fireplace)

HouseAu['FireplaceQu'].replace({np.nan:'No Fireplace'},inplace=True)

print(100*(HouseAu['FireplaceQu'].value_counts()/HouseAu['FireplaceQu'].count()))

#Imputing level values of FireplaceQu

HouseAu['FireplaceQu'].replace({'Fa':'OK Fireplace'},inplace=True)

HouseAu['FireplaceQu'].replace({'TA':'OK Fireplace'},inplace=True)

HouseAu['FireplaceQu'].replace({'Gd':'OK Fireplace'},inplace=True)

HouseAu['FireplaceQu'].replace({'Ex':'OK Fireplace'},inplace=True)

HouseAu['FireplaceQu'].replace({'Po':'OK Fireplace'},inplace=True)

print(100*(HouseAu['FireplaceQu'].value_counts()/HouseAu['FireplaceQu'].count()))
#Plotting the categorical variables related to FireplaceQu and checking correlation with SalePrice

plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)

sns.boxplot(x = 'FireplaceQu', y = 'SalePrice', data = HouseAu)

#Clearly Fireplace presence drives Sale price to some extent
#Cleaning up variable GarageType (Replacing NA => No Garage)

HouseAu['GarageType'].replace({np.nan:'No Garage'},inplace=True)

100*(HouseAu['GarageType'].value_counts()/HouseAu['GarageType'].count())
#Cleaning up variable GarageFinish (Replacing NA => No Garage)

HouseAu['GarageFinish'].replace({np.nan:'No Garage'},inplace=True)

100*(HouseAu['GarageFinish'].value_counts()/HouseAu['GarageFinish'].count())
#Cleaning up variable GarageQual (Replacing NA => No Garage)

HouseAu['GarageQual'].replace({np.nan:'No Garage'},inplace=True)

print(100*(HouseAu['GarageQual'].value_counts()/HouseAu['GarageQual'].count()))

#Imputing level values of GarageQual

HouseAu['GarageQual'].replace({'TA':'OK Garage'},inplace=True)

HouseAu['GarageQual'].replace({'Fa':'OK Garage'},inplace=True)

HouseAu['GarageQual'].replace({'Gd':'OK Garage'},inplace=True)

HouseAu['GarageQual'].replace({'Ex':'OK Garage'},inplace=True)

HouseAu['GarageQual'].replace({'Po':'No Garage'},inplace=True)

print(100*(HouseAu['GarageQual'].value_counts()/HouseAu['GarageQual'].count()))
#Cleaning up variable GarageCond (Replacing NA => No Garage)

HouseAu['GarageCond'].replace({np.nan:'No Garage'},inplace=True)

print(100*(HouseAu['GarageCond'].value_counts()/HouseAu['GarageCond'].count()))

#Imputing level values of GarageCond

HouseAu['GarageCond'].replace({'TA':'OK'},inplace=True)

HouseAu['GarageCond'].replace({'Fa':'OK'},inplace=True)

HouseAu['GarageCond'].replace({'Gd':'OK'},inplace=True)

HouseAu['GarageCond'].replace({'Ex':'OK'},inplace=True)

HouseAu['GarageCond'].replace({'Po':'No Garage'},inplace=True)

print(100*(HouseAu['GarageCond'].value_counts()/HouseAu['GarageCond'].count()))
#Plotting the categorical variables related to Garage and checking correlation with SalePrice

plt.figure(figsize=(20, 12))

plt.subplot(3,3,1)

sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,2)

sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,3)

sns.boxplot(x = 'GarageFinish', y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,4)

sns.boxplot(x = 'GarageType', y = 'SalePrice', data = HouseAu)

#GarageCond and GarageQual seem to be same in influence on SalePrice, one can be dropped

HouseAu=HouseAu.drop(['GarageQual'],axis=1)

#Garage type - CarPort, No Garage, Basement, 2Types can be combined as "Others"

#Imputing values to "Others"

HouseAu['GarageType'].replace({'CarPort':'No Garage'},inplace=True)

HouseAu['GarageType'].replace({'Basment':'No Garage'},inplace=True)

HouseAu['GarageType'].replace({'No Garage':'No Garage'},inplace=True)

HouseAu['GarageType'].replace({'2Types':'No Garage'},inplace=True)

print(100*(HouseAu['GarageType'].value_counts()/HouseAu['GarageType'].count()))
#Cleaning up variable PoolQC (Replacing NA => No Pool)

HouseAu['PoolQC'].replace({np.nan:'No Pool'},inplace=True)

print(100*(HouseAu['PoolQC'].value_counts()/HouseAu['PoolQC'].count()))

#Imputing level values of PoolQC

HouseAu['PoolQC'].replace({'Fa':'OK'},inplace=True)

HouseAu['PoolQC'].replace({'Gd':'OK'},inplace=True)

HouseAu['PoolQC'].replace({'Ex':'OK'},inplace=True)

print(100*(HouseAu['PoolQC'].value_counts()/HouseAu['PoolQC'].count()))
#Plotting the categorical variables related to PooQC to find which ones have correlation and can be dropped

plt.figure(figsize=(10,10))

plt.subplot(3,3,1)

sns.boxplot(x = 'PoolQC', y = 'SalePrice', data = HouseAu)

#PoolQC is only 0.4% of the houses so a small subset of data
#Sale Price is not strongly changing with Pool or No Pool, effects can be captured with Pool Area. Do dropping PoolQC

HouseAu=HouseAu.drop(['PoolQC'],axis=1)
#Cleaning up variable Fence (Replacing NA => No Fence)

HouseAu['Fence'].replace({np.nan:'No Fence'},inplace=True)

print(100*(HouseAu['Fence'].value_counts()/HouseAu['Fence'].count()))

#Imputing level values of Fence

HouseAu['Fence'].replace({'MnPrv':'Fence'},inplace=True)

HouseAu['Fence'].replace({'GdPrv':'Fence'},inplace=True)

HouseAu['Fence'].replace({'GdWo':'Fence'},inplace=True)

HouseAu['Fence'].replace({'MnWw':'Fence'},inplace=True)

print(100*(HouseAu['Fence'].value_counts()/HouseAu['Fence'].count()))
#Cleaning up variable MiscFeature (Replacing NA => No Fence)

HouseAu['MiscFeature'].replace({np.nan:'None'},inplace=True)

100*(HouseAu['MiscFeature'].value_counts()/HouseAu['MiscFeature'].count())
#Plotting the categorical variables related to MiscFeature to find which ones have correlation and can be dropped

plt.figure(figsize=(20,20))

plt.subplot(3,3,1)

sns.boxplot(x = 'MiscFeature', y = 'SalePrice', data = HouseAu)

#MiscFeature levels are a minor subset in the dataset but seem to have a good influence on sale price.
#Taking a deep dive into the Basement related attributes to understand the correlations

HouseAu_Porch=HouseAu[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']]

print(HouseAu_Porch.head())

print(HouseAu_Porch.info())

# pairwise scatter plot

plt.figure(figsize=(20, 10))

sns.pairplot(HouseAu_Porch)

plt.show()
#From the correlation pairplots, Out of four variables on Porch, we can capture key effects from Open Porch and Enclosed Porch

HouseAu=HouseAu.drop(['ScreenPorch','3SsnPorch'],axis=1)
print(100*(HouseAu['Neighborhood'].astype('category').value_counts()/HouseAu['Neighborhood'].count()))

#Imputing values of the minor category levels in Neighborhood

HouseAu['Neighborhood'].replace({'ClearCr':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'SWISU':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'StoneBr':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'Blmngtn':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'MeadowV':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'BrDale':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'Veenker':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'NPkVill':'Others'},inplace=True)

HouseAu['Neighborhood'].replace({'Blueste':'Others'},inplace=True)

print(100*(HouseAu['Neighborhood'].astype('category').value_counts()/HouseAu['Neighborhood'].count()))
#Binning of the Year built variable

#Creating bins to define the year periods - 1872-1925, 1925-1950,1950-1975, 1976-1990, 1991-2000,2001-2010

bins=[1872,1925,1950,1976,1991,2001,2010]

slot_names=['1872-1925','1925-1950','1950-1975','1976-1990','1991-2000','2001-2010']

HouseAu['YearBuilt']=pd.cut(HouseAu['YearBuilt'],bins,labels=slot_names,include_lowest=True)

print(100*(HouseAu['YearBuilt'].value_counts()/HouseAu['YearBuilt'].count()))
#Binning of the YearRemodAdd variable

#Creating bins to define the year periods - 1872-1925, 1925-1950,1950-1975, 1976-1990, 1991-2000,2001-2010

bins=[1872,1950,1976,1991,2001,2010]

slot_names=['1872-1950','1950-1975','1976-1990','1991-2000','2001-2010']

HouseAu['YearRemodAdd']=pd.cut(HouseAu['YearRemodAdd'],bins,labels=slot_names,include_lowest=True)

100*(HouseAu['YearRemodAdd'].value_counts()/HouseAu['YearRemodAdd'].count())
#Plotting the categorical variables related toYear Built and Year Remodified to find which ones have correlation and can be dropped

plt.figure(figsize=(20,15))

plt.subplot(3,3,1)

sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = HouseAu)

plt.subplot(3,3,2)

sns.boxplot(x = 'YearRemodAdd', y = 'SalePrice', data = HouseAu)
# percentage of missing values in each column

round(HouseAu.isnull().sum()/len(HouseAu.index), 2)*100
# missing values in rows

HouseAu.isnull().sum(axis=1)
#Converting the binned year columns as object datatype

HouseAu['YearBuilt']=HouseAu['YearBuilt'].values.astype(np.object)

HouseAu['YearRemodAdd']=HouseAu['YearRemodAdd'].values.astype(np.object)
#Cleaning up variable LotFrontage (Replacing NA => 0)

HouseAu['LotFrontage'].replace({np.nan:'0'},inplace=True)

HouseAu['LotFrontage']=HouseAu['LotFrontage'].values.astype(np.int64)

100*(HouseAu['LotFrontage'].value_counts()/HouseAu['LotFrontage'].count())

HouseAu.info()
#MasVnrArea: Masonry veneer area in square feet

100*(HouseAu['MasVnrType'].astype('category').value_counts()/HouseAu['MasVnrType'].count())
HouseAu=HouseAu.drop(['GarageYrBlt'],axis=1) # As it is same as Year Built
#Replacing missing value with Unknown

HouseAu['Electrical'].replace({np.nan:'Unknown'},inplace=True)

print(100*(HouseAu['Electrical'].value_counts()/HouseAu['Electrical'].count()))

#Imputing the minor category levels of Electrical

HouseAu['Electrical'].replace({'FuseA':'Other'},inplace=True)

HouseAu['Electrical'].replace({'FuseF':'Other'},inplace=True)

HouseAu['Electrical'].replace({'FuseP':'Other'},inplace=True)

HouseAu['Electrical'].replace({'Mix':'Other'},inplace=True)

HouseAu['Electrical'].replace({'Unknown':'Other'},inplace=True)

print(100*(HouseAu['Electrical'].value_counts()/HouseAu['Electrical'].count()))
# checking whether some rows have more than 1 missing values

len(HouseAu[HouseAu.isnull().sum(axis=1) > 1].index)
#NULL Rows in MasVnrType 

HouseAu=HouseAu.dropna(how='any',axis=0)
#Dropping column MasVnrarea and LotFrontage as these are not adding value

HouseAu=HouseAu.drop(['MasVnrArea','LotFrontage'],axis=1)
# percentage of missing values in each column

round(HouseAu.isnull().sum()/len(HouseAu.index), 2)*100
HouseAu.describe().transpose()
#finding uniqness in records we see there is no attribute column with a single value

HouseAu.nunique().sort_values(ascending =True)
HouseAu.shape
# all numeric (float and int) variables in the dataset

HouseAu_numeric = HouseAu.select_dtypes(include=['float64', 'int64'])

HouseAu_numeric.head()
# dropping ID column 

HouseAu_numeric = HouseAu_numeric.drop(['Id'], axis=1)

HouseAu_numeric.head()
# pairwise scatter plot

plt.figure(figsize=(20, 10))

sns.pairplot(HouseAu_numeric)

plt.show()
# correlation matrix

cor = HouseAu_numeric.corr()

cor
# plotting correlations on a heatmap

# figure size

plt.figure(figsize=(16,8))

# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()
#Target variable: sale price of house

sns.distplot(HouseAu['SalePrice'])

plt.show()

#Normally distributed SalePrice
# Predictor variable: LotArea --> Lot size in square feet

sns.distplot(HouseAu['LotArea'])

plt.show()

#Normally distributed --- LotArea
# Predictor variable: GrLivArea--> Above grade (ground) living area square feet

sns.distplot(HouseAu['GrLivArea'])

plt.show()

#Normally distributed but slightly bimodal
# Predictor variable: TotalBsmtSF ---> Total square feet of basement area

sns.distplot(HouseAu['TotalBsmtSF'])

plt.show()
# Predictor variable: 1stFlrSF: First Floor square feet

sns.distplot(HouseAu['1stFlrSF'])

plt.show()

#Normally distributed with bimodaldistribution
# Predictor variable: 2ndFlrSF: Second floor square feet

sns.distplot(HouseAu['2ndFlrSF'])

plt.show()

#Normally distributed 
HouseAu.info()
# split into X and y

X = HouseAu.loc[:, ['MSSubClass','MSZoning','LotArea','Street','LotShape','LandContour','Utilities',

                    'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',

                    'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',

                    'ExterQual','ExterCond','Foundation','BsmtQual','BsmtExposure','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical',

                    '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',

                    'BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu',

                    'GarageType','GarageFinish','GarageCars','GarageArea','GarageCond','PavedDrive',

                    'WoodDeckSF','OpenPorchSF','EnclosedPorch','PoolArea','Fence','MiscFeature',

                    'MiscVal','MoSold','YrSold','SaleType','SaleCondition']]



y = HouseAu['SalePrice']
# creating dummy variables for categorical variables



# subset all categorical variables

HouseAu_categorical = X.select_dtypes(include=['object'])

HouseAu_categorical.head()
# convert categorical variables into dummies

HouseAu_dummies = pd.get_dummies(HouseAu_categorical, drop_first=True)

HouseAu_dummies.head()
# drop categorical variables 

X = X.drop(list(HouseAu_categorical.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, HouseAu_dummies], axis=1)
X.shape
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
# split into train and test

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=100)
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}

# cross validation

folds = 5



lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# plotting mean test and train scores with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv.best_params_
alpha =100



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
#Extracting the coefficients and model equation from lasso regression

lasso.coef_
# lasso model parameters generation

model_parameters = list(lasso.coef_)

model_parameters.insert(0, lasso.intercept_)

model_parameters = [round(x, 1) for x in model_parameters]

cols = X.columns

cols = cols.insert(0, "constant")

print(list(zip(cols, model_parameters)))
# model with optimal alpha

# lasso regression

lm1 = Lasso(alpha=100)

#lm1 = Lasso(alpha=0.001)

lm1.fit(X_train, y_train)



from sklearn.metrics import r2_score

# predict

y_train_pred = lm1.predict(X_train)

#print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm1.predict(X_test)

#print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))



#New Code for R2

r_square_score=r2_score(y_test,y_test_pred)

print("R Square score:{}".format(r_square_score))
#AIC and BIC Calculation

n= 1016 # n is equal to total datapoints on which model is built

k= 100 # k is equal to number of predictor variables in model built

resid=np.subtract(y_test_pred,y_test)

rss=np.sum(np.power(resid,2))

print("RSS:{}".format(rss))

aic=n*np.log(rss/n)+2*k

print("AIC:{}".format(aic))

bic=n*np.log(rss/n)+k*np.log(n)

print("BIC:{}".format(bic))
#Predictor Variables from the Model built using Lasso Regression:

[('constant', 180762.1), ('MSSubClass', -2299.5), ('LotArea', 6976.5), ('OverallQual', 12762.5), ('OverallCond', 5932.6), ('TotalBsmtSF', 13352.4), ('1stFlrSF', 0.0), ('2ndFlrSF', 6658.1), ('LowQualFinSF', -2033.2), ('GrLivArea', 27001.7), ('BsmtFullBath', 6507.3), ('BsmtHalfBath', 0.0), ('FullBath', 1629.3), ('HalfBath', 1682.0), ('BedroomAbvGr', -4823.0), ('KitchenAbvGr', -4531.2), ('TotRmsAbvGrd', 1524.2), ('Fireplaces', 2298.4), ('GarageCars', 5535.4), ('GarageArea', 322.1), ('WoodDeckSF', 582.8), ('OpenPorchSF', -0.0), ('EnclosedPorch', -1227.0), ('PoolArea', 4833.0), ('MiscVal', 1988.5), ('MoSold', 0.0), ('YrSold', 208.3), ('MSZoning_FV', 4201.8), ('MSZoning_RH', -53.1), ('MSZoning_RL', 2131.6), ('MSZoning_RM', 0.0), ('Street_Pave', 2159.7), ('LotShape_IR2', -282.6), ('LotShape_IR3', 25.1), ('LotShape_Reg', -822.2), ('LandContour_HLS', 1103.4), ('LandContour_Low', -1905.3), ('LandContour_Lvl', 0.0), ('Utilities_NoSeWa', -192.7), ('LotConfig_CulDSac', 2034.0), ('LotConfig_FR2', -1105.6), ('LotConfig_FR3', -753.2), ('LotConfig_Inside', -551.2), ('LandSlope_Mod', 458.8), ('LandSlope_Sev', -4745.6), ('Neighborhood_CollgCr', -925.1), ('Neighborhood_Crawfor', 3167.2), ('Neighborhood_Edwards', -1462.2), ('Neighborhood_Gilbert', -1587.3), ('Neighborhood_IDOTRR', 348.6), ('Neighborhood_Mitchel', -2035.4), ('Neighborhood_NAmes', -2336.2), ('Neighborhood_NWAmes', -1841.2), ('Neighborhood_NoRidge', 6125.6), ('Neighborhood_NridgHt', 3541.9), ('Neighborhood_OldTown', -2038.7), ('Neighborhood_Others', 2368.6), ('Neighborhood_Sawyer', -0.0), ('Neighborhood_SawyerW', 602.9), ('Neighborhood_Somerst', -7.3), ('Neighborhood_Timber', -971.6), ('Condition1_Feedr', 1509.1), ('Condition1_Norm', 5238.4), ('Condition1_PosA', 1450.9), ('Condition1_PosN', 1925.5), ('Condition1_RRAe', -334.5), ('Condition1_RRAn', 1224.4), ('Condition1_RRNe', 0.0), ('Condition1_RRNn', 557.4), ('Condition2_Feedr', -0.0), ('Condition2_Norm', 0.0), ('Condition2_PosA', 1055.2), ('Condition2_PosN', -15909.2), ('Condition2_RRAe', -0.0), ('Condition2_RRAn', -0.0), ('Condition2_RRNn', 0.0), ('BldgType_2fmCon', 0.0), ('BldgType_Duplex', -936.7), ('BldgType_Twnhs', -2828.2), ('BldgType_TwnhsE', -2544.1), ('HouseStyle_1.5Unf', 982.8), ('HouseStyle_1Story', 3463.2), ('HouseStyle_2.5Fin', -2102.8), ('HouseStyle_2.5Unf', -588.6), ('HouseStyle_2Story', -765.1), ('HouseStyle_SFoyer', 54.0), ('HouseStyle_SLvl', 955.6), ('YearBuilt_1925-1950', 2674.2), ('YearBuilt_1950-1975', 1812.2), ('YearBuilt_1976-1990', 3304.6), ('YearBuilt_1991-2000', 6088.8), ('YearBuilt_2001-2010', 7061.1), ('YearRemodAdd_1950-1975', 1089.8), ('YearRemodAdd_1976-1990', 0.0), ('YearRemodAdd_1991-2000', 1295.7), ('YearRemodAdd_2001-2010', 1487.3), ('RoofStyle_Gable', -0.0), ('RoofStyle_Gambrel', 73.5), ('RoofStyle_Hip', 651.7), ('RoofStyle_Mansard', 183.3), ('RoofStyle_Shed', 2116.0), ('RoofMatl_CompShg', 0.0), ('RoofMatl_Membran', -0.0), ('RoofMatl_Metal', 1124.1), ('RoofMatl_Roll', -141.8), ('RoofMatl_Tar&Grv', -934.8), ('RoofMatl_WdShake', -427.2), ('RoofMatl_WdShngl', 63.9), ('Exterior1st_AsphShn', -0.0), ('Exterior1st_BrkComm', -161.8), ('Exterior1st_BrkFace', 2355.7), ('Exterior1st_CBlock', -30.6), ('Exterior1st_CemntBd', 0.0), ('Exterior1st_HdBoard', -574.3), ('Exterior1st_ImStucc', -0.0), ('Exterior1st_MetalSd', 0.0), ('Exterior1st_Plywood', -393.2), ('Exterior1st_Stone', -99.7), ('Exterior1st_Stucco', -549.7), ('Exterior1st_VinylSd', -0.0), ('Exterior1st_Wd Sdng', -0.0), ('Exterior1st_WdShing', 238.0), ('Exterior2nd_AsphShn', 544.7), ('Exterior2nd_Brk Cmn', 0.0), ('Exterior2nd_BrkFace', -785.7), ('Exterior2nd_CBlock', -4.5), ('Exterior2nd_CmentBd', 830.9), ('Exterior2nd_HdBoard', 0.0), ('Exterior2nd_ImStucc', 862.6), ('Exterior2nd_MetalSd', 593.2), ('Exterior2nd_Other', 359.6), ('Exterior2nd_Plywood', -897.3), ('Exterior2nd_Stone', 398.1), ('Exterior2nd_Stucco', -494.5), ('Exterior2nd_VinylSd', -0.0), ('Exterior2nd_Wd Sdng', -1708.3), ('Exterior2nd_Wd Shng', -919.7), ('MasVnrType_BrkFace', 0.0), ('MasVnrType_None', 0.0), ('MasVnrType_Stone', 2715.7), ('ExterQual_Fa', -723.7), ('ExterQual_Gd', -8167.3), ('ExterQual_TA', -8529.6), ('ExterCond_Fa', -267.7), ('ExterCond_Gd', -0.0), ('ExterCond_Po', -0.0), ('ExterCond_TA', 430.7), ('Foundation_CBlock', 2254.4), ('Foundation_PConc', 1657.6), ('Foundation_Slab', 389.7), ('Foundation_Stone', 335.5), ('Foundation_Wood', -1002.7), ('BsmtQual_Others', 6617.0), ('BsmtQual_TA', 1466.7), ('BsmtExposure_Gd', 4489.8), ('BsmtExposure_Mn', -1743.4), ('BsmtExposure_No', -4071.3), ('BsmtExposure_No Basement', 298.7), ('Heating_GasA', -0.0), ('Heating_GasW', 407.8), ('Heating_Grav', 315.7), ('Heating_OthW', -1628.0), ('Heating_Wall', 186.3), ('HeatingQC_Fa', -796.3), ('HeatingQC_Gd', -265.3), ('HeatingQC_Po', -189.7), ('HeatingQC_TA', -0.0), ('CentralAir_Y', 595.5), ('Electrical_SBrkr', -176.6), ('KitchenQual_Fa', -3268.2), ('KitchenQual_Gd', -10980.7), ('KitchenQual_TA', -10858.8), ('Functional_Maj2', -786.1), ('Functional_Min1', 2032.5), ('Functional_Min2', 861.8), ('Functional_Mod', -290.7), ('Functional_Sev', -854.9), ('Functional_Typ', 5678.9), ('FireplaceQu_OK Fireplace', -1838.5), ('GarageType_BuiltIn', 766.3), ('GarageType_Detchd', 232.7), ('GarageType_No Garage', -468.6), ('GarageFinish_No Garage', 3184.3), ('GarageFinish_RFn', -1373.5), ('GarageFinish_Unf', -1305.2), ('GarageCond_OK', 1108.2), ('PavedDrive_P', -990.9), ('PavedDrive_Y', 195.3), ('Fence_No Fence', 171.3), ('MiscFeature_None', 0.0), ('MiscFeature_Othr', 371.7), ('MiscFeature_Shed', -0.0), ('MiscFeature_TenC', -2114.1), ('SaleType_CWD', 521.8), ('SaleType_Con', 643.6), ('SaleType_ConLD', 376.3), ('SaleType_ConLI', 1153.8), ('SaleType_ConLw', 562.1), ('SaleType_New', 5113.1), ('SaleType_Oth', 756.3), ('SaleType_WD', 0.0), ('SaleCondition_AdjLand', 441.1), ('SaleCondition_Alloca', -574.3), ('SaleCondition_Family', 478.6), ('SaleCondition_Normal', 2011.1), ('SaleCondition_Partial', -0.0)]
# split into X and y, X being selected from predictor variables found in Lasso model

X = HouseAu.loc[:, ['MSSubClass','LotArea','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',

                    'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',

                    'ExterQual','Foundation','BsmtQual','BsmtExposure','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical',

                    '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',

                    'KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu',

                    'GarageType','GarageFinish','GarageCars','GarageCond','WoodDeckSF','EnclosedPorch','PoolArea',

                    'SaleType','SaleCondition']]



y = HouseAu['SalePrice']
# creating dummy variables for categorical variables



# subset all categorical variables

HouseAu_categorical = X.select_dtypes(include=['object'])

HouseAu_categorical.head()
# convert categorical variables into dummies

HouseAu_dummies = pd.get_dummies(HouseAu_categorical, drop_first=True)

HouseAu_dummies.head()
# drop categorical variables 

X = X.drop(list(HouseAu_categorical.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, HouseAu_dummies], axis=1)
X.shape
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv.best_params_
alpha = 10

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

#Predictor Variables from the Model built using Ridge Regression:

ridge.coef_
# ridge model parameters

model_parameters = list(ridge.coef_)

model_parameters.insert(0, ridge.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = X.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))

#Predictor Variables from the Model built using Ridge Regression:
# model with optimal alpha

# Ridge regression

lm2 = Ridge(alpha=10)

#lm2 = Ridge(alpha=0.001)

lm2.fit(X_train, y_train)



from sklearn.metrics import r2_score

# predict

y_train_pred = lm2.predict(X_train)

#print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm2.predict(X_test)

#print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))



#New Code for R2

r_square_score=r2_score(y_test,y_test_pred)

print("R Square score:{}".format(r_square_score))
#AIC and BIC Calculation

n= 1016 # n is equal to total datapoints on which model is built

k= 50 # k is equal to number of predictor variables in model built

resid=np.subtract(y_test_pred,y_test)

rss=np.sum(np.power(resid,2))

print("RSS:{}".format(rss))

aic=n*np.log(rss/n)+2*k

print("AIC:{}".format(aic))

bic=n*np.log(rss/n)+k*np.log(n)

print("BIC:{}".format(bic))