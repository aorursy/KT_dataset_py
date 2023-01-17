import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model, metrics

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')
# reading the dataset



housing= pd.read_csv('../input/house-prices-data/train.csv')



housing.head()
# summary of the dataset:



print(housing.info())
housing.describe()
housing.describe(percentiles=[.25, .5, .75, .90, .95, .99])
#SalePrice



plt.boxplot(housing['SalePrice'])

Q1 = housing['SalePrice'].quantile(0.1)

Q3 = housing['SalePrice'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['SalePrice'] >= Q1 - 1.5*IQR) & 

                      (housing['SalePrice'] <= Q3 + 1.5*IQR)]

housing.shape
# Lot Area

plt.boxplot(housing['LotArea'])

Q1 = housing['LotArea'].quantile(0.1)

Q3 = housing['LotArea'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['LotArea'] >= Q1 - 1.5*IQR) & 

                      (housing['LotArea'] <= Q3 + 1.5*IQR)]

housing.shape
# MiscVal



plt.boxplot(housing['MiscVal'])

Q1 = housing['MiscVal'].quantile(0.1)

Q3 = housing['MiscVal'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['MiscVal'] >= Q1 - 1.5*IQR) & 

                      (housing['MiscVal'] <= Q3 + 1.5*IQR)]

housing.shape
# LotFrontage

plt.boxplot(housing['LotFrontage'])

Q1 = housing['LotFrontage'].quantile(0.1)

Q3 = housing['LotFrontage'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['LotFrontage'] >= Q1 - 1.5*IQR) & 

                      (housing['LotFrontage'] <= Q3 + 1.5*IQR)]

housing.shape
# MasVnrArea

plt.boxplot(housing['MasVnrArea'])

Q1 = housing['MasVnrArea'].quantile(0.1)

Q3 = housing['MasVnrArea'].quantile(0.9)

IQR = Q3 - Q1

housing = housing[(housing['MasVnrArea'] >= Q1 - 1.5*IQR) & 

                      (housing['MasVnrArea'] <= Q3 + 1.5*IQR)]

housing.shape
# correlation matrix

cor = housing.corr()

cor
# Checking the percentage of missing values

missing = round(100*(housing.isnull().sum()/len(housing.Id)), 2)

missing.loc[missing > 0]
columns_with_missing_values = list(missing[missing >= 70].index)



len(columns_with_missing_values)
housing = housing.drop(columns_with_missing_values,axis=1)

housing.shape
#NA in FireplaceQu column means No Fireplace, so we will replace NA by it.

housing['FireplaceQu'].fillna('No Fireplace', inplace=True)
housing['MasVnrArea'].fillna(0, inplace=True) 
housing['LotFrontage'].fillna(0, inplace=True) 
#NA in GarageType, GarageFinish, GarageQual, GarageCond columns mean No Garage, so we will replace NA by it.



housing['GarageType'].fillna('No Garage', inplace=True) 

housing['GarageFinish'].fillna('No Garage', inplace=True) 

housing['GarageQual'].fillna('No Garage', inplace=True) 

housing['GarageCond'].fillna('No Garage', inplace=True)
#converting year to number of years

housing['YearBuilt'] = 2020 - housing['YearBuilt']

housing['YearRemodAdd'] = 2020 - housing['YearRemodAdd']

housing['GarageYrBlt'] = 2020 - housing['GarageYrBlt']

housing['YrSold'] = 2020 - housing['YrSold']
#converting from int type to object to treat the variables as categorical variables



housing['MSSubClass'] = housing['MSSubClass'].astype('object')

housing['OverallQual'] = housing['OverallQual'].astype('object')

housing['OverallCond'] = housing['OverallCond'].astype('object')

housing['BsmtFullBath'] = housing['BsmtFullBath'].astype('object')

housing['BsmtHalfBath'] = housing['BsmtHalfBath'].astype('object')

housing['FullBath'] = housing['FullBath'].astype('object')

housing['HalfBath'] = housing['HalfBath'].astype('object')

housing['BedroomAbvGr'] = housing['BedroomAbvGr'].astype('object')

housing['KitchenAbvGr'] = housing['KitchenAbvGr'].astype('object')

housing['TotRmsAbvGrd'] = housing['TotRmsAbvGrd'].astype('object')

housing['Fireplaces'] = housing['Fireplaces'].astype('object')

housing['GarageCars'] = housing['GarageCars'].astype('object')
housing.shape
housing.isnull().values.any()
housing.isnull().sum().sum()
housing.dropna(inplace=True)
housing.shape
# List of variables to map



varlist1 =  ['Street']



# Defining the map function

def binary_map(x):

    return x.map({'Pave': 1, "Grvl": 0})



# Applying the function to the Lead list

housing[varlist1] = housing[varlist1].apply(binary_map)
#CentralAir



varlist2 =  ['CentralAir']



# Defining the map function

def binary_map(x):

    return x.map({'Y': 1, "N": 0})



# Applying the function to the Lead list

housing[varlist2] = housing[varlist2].apply(binary_map)
#Utilities



varlist3 =  ['Utilities']



# Defining the map function

def binary_map(x):

    return x.map({'AllPub': 1, "NoSeWa": 0})



# Applying the function to the Lead list

housing[varlist3] = housing[varlist3].apply(binary_map)
housing['Utilities']
drop_columns= ['Id']

housing= housing.drop(drop_columns,axis=1)
np.isnan(housing.any())

# creating dummy variables for categorical variables



# subset all categorical variables

house_categorical = housing.select_dtypes(include=['object'])

house_categorical.head()
# convert into dummies

house_dummies = pd.get_dummies(house_categorical, drop_first=True)

house_dummies.head()
# drop categorical variables 

housing = housing.drop(list(house_categorical.columns), axis=1)
housing.head()
# concat dummy variables with X

housing = pd.concat([housing, house_dummies], axis=1)
housing.head()

from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['LotFrontage','YearBuilt','YearRemodAdd','LotArea', 'TotalBsmtSF', 'GrLivArea',

           'MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GarageYrBlt',

            'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MoSold','YrSold','SalePrice'

           ]



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
df_train.tail()
y_train = df_train.pop('SalePrice')

X_train = df_train
# Test Split-

y_test = df_test.pop('SalePrice')

X_test = df_test
X_train.isnull().values.any()
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 50

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 30)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#RFE Support columns

col = X_train.columns[rfe.support_]

col
#Not RFE support columns



X_train.columns[~rfe.support_]
# list of alphas to tune

params = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5,1.0, 5.0, 10.0, 100]}





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
#checking the value of optimum number of parameters

print(model_cv.best_params_)

print(model_cv.best_score_)
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
alpha = 5.0

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
#lets predict the R-squared value of test and train data

y_train_pred = ridge.predict(X_train)

print('RSquare- '+str(metrics.r2_score(y_true=y_train, y_pred=y_train_pred)))
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

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.xscale('log')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
#checking the value of optimum number of parameters

print(model_cv.best_params_)

print(model_cv.best_score_)
alpha = 0.001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train)
#lets predict the R-squared value of test and train data

y_train_pred = lasso.predict(X_train)

print('RSquare- ' +str(metrics.r2_score(y_true=y_train, y_pred=y_train_pred)))
alpha = 0.001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
results= pd.DataFrame.from_dict(model_cv.cv_results_)
results= results.sort_values(['param_alpha'])

alphas= np.array(params['alpha'])
results[['param_alpha','mean_train_score','mean_test_score']]