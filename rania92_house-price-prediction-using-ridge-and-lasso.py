import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')

# reading the dataset

house = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# head

house.head()
house.shape
test.shape
# summary of the dataset: 1460 rows, 81 columns

house.info()
house.describe()      #other atributes of the dataframe
# all numeric (float and int) variables in the dataset

house_numeric = house.select_dtypes(include=['float64', 'int64'])

house_numeric.head()
house_numeric.info()
# dropping the columns we want to treat as categorical variables

house_numeric = house_numeric.drop(['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 

                                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

                                   'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 

                                   'MoSold', 'YrSold'], axis=1)

house_numeric.head()
house_numeric.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# outlier treatment

plt.boxplot(house['PoolArea'])

Q1 = house['PoolArea'].quantile(0.1)

Q3 = house['PoolArea'].quantile(0.9)

IQR = Q3 - Q1

house = house[(house['PoolArea'] >= Q1 - 1.5*IQR) & 

                      (house['PoolArea'] <= Q3 + 1.5*IQR)]

house.shape
# outlier treatment

plt.boxplot(house['MiscVal'])

Q1 = house['MiscVal'].quantile(0.1)

Q3 = house['MiscVal'].quantile(0.9)

IQR = Q3 - Q1

house = house[(house['MiscVal'] >= Q1 - 1.5*IQR) & 

                      (house['MiscVal'] <= Q3 + 1.5*IQR)]

house.shape
# outlier treatment

plt.boxplot(house['ScreenPorch'])

Q1 = house['ScreenPorch'].quantile(0.1)

Q3 = house['ScreenPorch'].quantile(0.9)

IQR = Q3 - Q1

house = house[(house['ScreenPorch'] >= Q1 - 1.5*IQR) & 

                      (house['ScreenPorch'] <= Q3 + 1.5*IQR)]

house.shape
# outlier treatment

plt.boxplot(house['LotArea'])

Q1 = house['LotArea'].quantile(0.1)

Q3 = house['LotArea'].quantile(0.9)

IQR = Q3 - Q1

house = house[(house['LotArea'] >= Q1 - 1.5*IQR) & 

                      (house['LotArea'] <= Q3 + 1.5*IQR)]

house.shape
# outlier treatment

plt.boxplot(house['MasVnrArea'])

Q1 = house['MasVnrArea'].quantile(0.1)

Q3 = house['MasVnrArea'].quantile(0.9)

IQR = Q3 - Q1

house = house[(house['MasVnrArea'] >= Q1 - 1.5*IQR) & 

                      (house['MasVnrArea'] <= Q3 + 1.5*IQR)]

house.shape
# outlier treatment

plt.boxplot(house['SalePrice'])

Q1 = house['SalePrice'].quantile(0.1)

Q3 = house['SalePrice'].quantile(0.9)

IQR = Q3 - Q1

house = house[(house['SalePrice'] >= Q1 - 1.5*IQR) & 

                      (house['SalePrice'] <= Q3 + 1.5*IQR)]

house.shape
# correlation matrix

cor = house_numeric.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(18,10))



# heatmap

sns.heatmap(cor, annot=True)

plt.show()

# variable formats

house.info()
house.isnull().sum()  #checking the number of null values in the dataset
# Checking the percentage of missing values

round(100*(house.isnull().sum()/len(house.index)), 2)
house.shape
house = pd.concat((house,test))
#NA in Alley column means No Alley, so we will replace NA by it.

house['Alley'].fillna('No Alley', inplace=True)
house['MasVnrType'].fillna('None', inplace=True) 
#NA in FireplaceQu column means No Fireplace, so we will replace NA by it.

house['FireplaceQu'].fillna('No Fireplace', inplace=True)
#NA in PoolQC column means No Pool, so we will replace NA by it.

house['PoolQC'].fillna('No Pool', inplace=True) 
#NA in Fence column means No Fence, so we will replace NA by it.

house['Fence'].fillna('No Fence', inplace=True) 
house['MasVnrArea'].fillna(0, inplace=True) 
house['LotFrontage'].fillna(0, inplace=True) 
#NA in GarageType, GarageFinish, GarageQual, GarageCond columns mean No Garage, so we will replace NA by it.



house['GarageType'].fillna('No Garage', inplace=True) 

house['GarageFinish'].fillna('No Garage', inplace=True) 

house['GarageQual'].fillna('No Garage', inplace=True) 

house['GarageCond'].fillna('No Garage', inplace=True) 
# MiscFeature column has almost 99% null values so we will drop it

house= house.drop('MiscFeature', axis=1)
house.isnull().sum()
#converting year to number of years

house['YearBuilt'] = 2019 - house['YearBuilt']

house['YearRemodAdd'] = 2019 - house['YearRemodAdd']

house['GarageYrBlt'] = 2019 - house['GarageYrBlt']

house['YrSold'] = 2019 - house['YrSold']
#converting from int type to object to treat the variables as categorical variables

house['MSSubClass'] = house['MSSubClass'].astype('object')

house['OverallQual'] = house['OverallQual'].astype('object')

house['OverallCond'] = house['OverallCond'].astype('object')

house['BsmtFullBath'] = house['BsmtFullBath'].astype('object')

house['BsmtHalfBath'] = house['BsmtHalfBath'].astype('object')

house['FullBath'] = house['FullBath'].astype('object')

house['HalfBath'] = house['HalfBath'].astype('object')

house['BedroomAbvGr'] = house['BedroomAbvGr'].astype('object')

house['KitchenAbvGr'] = house['KitchenAbvGr'].astype('object')

house['TotRmsAbvGrd'] = house['TotRmsAbvGrd'].astype('object')

house['Fireplaces'] = house['Fireplaces'].astype('object')

house['GarageCars'] = house['GarageCars'].astype('object')
house.shape
final = house
# List of variables to map



varlist1 =  ['Street']



# Defining the map function

def binary_map(x):

    return x.map({'Pave': 1, "Grvl": 0})



# Applying the function to the Lead list

final[varlist1] = final[varlist1].apply(binary_map)
# List of variables to map



varlist2 =  ['Utilities']



# Defining the map function

def binary_map(x):

    return x.map({'AllPub': 1, "NoSeWa": 0})



# Applying the function to the Lead list

final[varlist2] = final[varlist2].apply(binary_map)
# List of variables to map



varlist3 =  ['CentralAir']



# Defining the map function

def binary_map(x):

    return x.map({'Y': 1, "N": 0})



# Applying the function to the Lead list

final[varlist3] = final[varlist3].apply(binary_map)
# split into X and y

X = final.drop([ 'Id'], axis=1)

# creating dummy variables for categorical variables



# subset all categorical variables

house_categorical = X.select_dtypes(include=['object'])

house_categorical.head()

# convert into dummies

house_dummies = pd.get_dummies(house_categorical, drop_first=True)

house_dummies.head()
# drop categorical variables 

final = final.drop(list(house_categorical.columns), axis=1)
# concat dummy variables with X

final = pd.concat([final, house_dummies], axis=1)
final.shape
test = final.tail(1459)
test.shape
X = final.head(1253)

y = np.log(X.SalePrice)

X = X.drop("SalePrice",1) # take out the target variable
test = test.fillna(test.interpolate())

X = X.fillna(X.interpolate())
test = test.drop("SalePrice",1) # take out the target variable
# scaling the features



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)
# scaling the features



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(test)
# split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=100)
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
#checking the value of optimum number of parameters

print(model_cv.best_params_)

print(model_cv.best_score_)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=1000]

cv_results
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

plt.figure(figsize=(16,5))



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
alpha = 4

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
#lets predict the R-squared value of test and train data

y_train_pred = ridge.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
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
#lets find out the R-squared value of the lasso model

model_cv1 = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      



# fit the model

model_cv1.fit(X_train, y_train) 
# cv results

cv_results1 = pd.DataFrame(model_cv1.cv_results_)

cv_results1
# plotting cv results

plt.figure(figsize=(16,4))



plt.plot(cv_results1["param_alpha"], cv_results1["mean_test_score"])

plt.plot(cv_results1["param_alpha"], cv_results1["mean_train_score"])

plt.xlabel('number of features')

plt.ylabel('r-squared')

plt.title("Optimal Number of Features")

plt.legend(['test score', 'train score'], loc='upper right')
#checking the value of optimum number of parameters

print(model_cv.best_params_)

print(model_cv.best_score_)
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.figure(figsize=(16,5))

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.show()
alpha = 0.0001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
#lets predict the R-squared value of test and train data

y_train_pred = lasso.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

alpha = 0.0001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
#lets predict the R-squared value of test and train data

y_test_pred = lasso.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

from sklearn.metrics import mean_squared_error

print ('RMSE is: \n', mean_squared_error(y_test, y_test_pred))
alpha = 0.0001



lasso = Lasso(alpha=alpha)



lasso.fit(X_train,y_train)

preds = lasso.predict(test)

final_predictions = np.exp(preds)
test.index = test.index + 1461
submission = pd.DataFrame({'Id': test.index ,'SalePrice': final_predictions })

submission.to_csv("submission.csv",index=False)
alpha = 4



ridge = Ridge(alpha=alpha)



ridge.fit(X_train,y_train)

preds1 = ridge.predict(test)

final_predictions1 = np.exp(preds1)
submission1 = pd.DataFrame({'Id': test.index ,'SalePrice': final_predictions1 })

submission1.to_csv("submission1.csv",index=False)