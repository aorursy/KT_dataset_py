import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Reading the dataset
train = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")
test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

houses = pd.concat([train,test], axis=0, ignore_index=True)
# Inspect the shape of the dataset
houses.shape
# Inspect the different columsn in the dataset
houses.columns
# Let's take a look at the first few rows
houses.head()
# Check the summary of the dataset
houses.describe()
# Summary of the dataset: 
print(houses.info())
# All numeric (float and int) variables in the dataset
houses_numeric = houses.select_dtypes(include=['float64', 'int64'])
houses_numeric.head()
# dropping the columns we want to treat as categorical variables
houses_numeric = houses_numeric.drop(['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
                                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                                   'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
                                   'MoSold', 'YrSold'], axis=1)
houses_numeric.head()
# Pairwise scatter plot

#plt.figure(figsize=(20, 10))
#sns.pairplot(houses_numeric)
#plt.show()
# Correlation matrix
cor = houses_numeric.corr()
cor
# Figure size
plt.figure(figsize=(18,10))

# Heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()
# Check the number of missing values in each column

houses.isnull().sum()
# Let's now check the percentage of missing values in each column

round(100*(houses.isnull().sum()/len(houses.index)), 2)
##Repalcing NA values with meaningful values as NA means the facility is not present

#NA in Alley column means No Alley
houses['Alley'].fillna('No Alley', inplace=True)

houses['MasVnrType'].fillna('None', inplace=True) 

#NA in FireplaceQu column means No Fireplace
houses['FireplaceQu'].fillna('No Fireplace', inplace=True)

#NA in PoolQC column means No Pool
houses['PoolQC'].fillna('No Pool', inplace=True) 

#NA in Fence column means No Fence
houses['Fence'].fillna('No Fence', inplace=True) 
houses['MasVnrArea'].fillna(0, inplace=True) 
houses['LotFrontage'].fillna(0, inplace=True) 
houses['GarageType'].fillna('No Garage', inplace=True) 
houses['GarageFinish'].fillna('No Garage', inplace=True) 
houses['GarageQual'].fillna('No Garage', inplace=True) 
houses['GarageCond'].fillna('No Garage', inplace=True) 
# Dropping "FireplaceQu" having 47% missing values
houses.drop(['MiscFeature'], axis = 1, inplace = True)
# Get the value counts of all the columns

#for column in houses:
#    if round(100*(houses[column].astype('category').value_counts()/len(houses.index)), 2)[0] >90:
 #       print('___________________________________________________')
## Dropping highly skewed features
houses.drop(['LandSlope','LowQualFinSF', 'BsmtHalfBath', 
         'ScreenPorch', 'PoolArea', 'MiscVal'], axis = 1, inplace = True)
# Check the number of null values again
houses.isnull().sum()
# Fill the empty values with median values
houses= houses.apply(lambda x: x.fillna(x.value_counts().index[0]))
# Check the number of null values again
houses.isnull().sum()
print(len(houses.index))
print(len(houses.index)/1460)
# Let's look at the dataset again

houses.head()
houses.drop(['Id','MSZoning'], 1, inplace = True)
houses.head(20)
#converting from int type to object to treat the variables as categorical variables

houses['MSSubClass'] = houses['MSSubClass'].astype('object')
houses['OverallQual'] = houses['OverallQual'].astype('object')
houses['OverallCond'] = houses['OverallCond'].astype('object')
houses['BsmtFullBath'] = houses['BsmtFullBath'].astype('object')
#houses['BsmtHalfBath'] = houses['BsmtHalfBath'].astype('object')
houses['FullBath'] = houses['FullBath'].astype('object')
houses['HalfBath'] = houses['HalfBath'].astype('object')
houses['BedroomAbvGr'] = houses['BedroomAbvGr'].astype('object')
houses['KitchenAbvGr'] = houses['KitchenAbvGr'].astype('object')
houses['TotRmsAbvGrd'] = houses['TotRmsAbvGrd'].astype('object')
houses['Fireplaces'] = houses['Fireplaces'].astype('object')
houses['GarageCars'] = houses['GarageCars'].astype('object')
# subset all categorical variables
house_categorical = houses.select_dtypes(include=['object'])
house_categorical.head()
# convert into dummies
house_dummies = pd.get_dummies(house_categorical, drop_first=True)
house_dummies.head()
# drop categorical variables 
houses = houses.drop(list(house_categorical.columns), axis=1)
houses = pd.concat([houses, house_dummies], axis=1)
houses.shape
CurrentDate = 2020

houses['Age of House'] = CurrentDate - houses['YearBuilt']
houses.drop(['YearBuilt'], 1, inplace = True)

houses['last remodeled age'] = CurrentDate - houses['YearRemodAdd']
houses.drop(['YearRemodAdd'], 1, inplace = True)

houses['Last deal of the house'] = CurrentDate - houses['YrSold']
houses.drop(['YrSold'], 1, inplace = True)

houses['Garage Age'] = CurrentDate - houses['GarageYrBlt']
houses.drop(['GarageYrBlt'], 1, inplace = True)
houses.shape
houses.head()
train_data = houses.iloc[:train.shape[0]]
train_data['SalePrice'].tail()
train_data
test_data = houses.iloc[train.shape[0]:]
test_data =test_data.drop(['SalePrice'], 1)
#test_data['SalePrice'].head()
# Put all the feature variables in X

X = train_data.drop(['SalePrice'], 1)
X.head()
# Put the target variable in y

y = train_data['SalePrice']

y.head()
# target variable: price of car
sns.distplot(train_data['SalePrice'])
plt.show()
y = np.log1p(y)
# target variable: price of car
sns.distplot(y)
plt.show()
houses_numeric.head()
# Import the StandardScaler()
from sklearn.preprocessing import StandardScaler

# Create a scaling object
scaler = StandardScaler()

# Create a list of the variables that you need to scale
varlist = ['LotFrontage','LotArea','MasVnrArea','TotalBsmtSF',
       '1stFlrSF','2ndFlrSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF']

# Scale these variables using 'fit_transform'
X[varlist] = scaler.fit_transform(X[varlist])
test_data[varlist] = scaler.transform(test_data[varlist])
test_data.head()
X.head()
# split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)
X_train
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
print(model_cv.best_estimator_)
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
print(model_cv.best_estimator_)
print(model_cv.best_params_)
print(model_cv.best_score_)
alpha = 10
ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)
ridge.coef_
#lets predict the R-squared value of test and train data
from sklearn import metrics

y_train_pred = ridge.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
type(ridge.coef_)
coeffs = np.squeeze (np.asarray(ridge.coef_))
print(coeffs)
X_train.columns
ridge_features = pd.Series(coeffs, index = X_train.columns)
ridge_features.abs().sort_values(ascending=False)
alpha = 10

ridge = Ridge(alpha=alpha)

ridge.fit(X_train,y_train)
preds1 = ridge.predict(test_data)
final_predictions_ridge = np.exp(preds1)
ridge_values = pd.DataFrame({'Id': test_data.index , 
                             'SalePrice_predicted':final_predictions_ridge })
ridge_values.head()
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
#find out the R-squared value of the lasso model
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
# plotting CV results
plt.figure(figsize=(16,4))

plt.plot(cv_results1["param_alpha"], cv_results1["mean_test_score"])
plt.plot(cv_results1["param_alpha"], cv_results1["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper right')
#checking the value of optimum number of parameters
print(model_cv.best_estimator_)
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
#predicting the R-squared value of test and train data
y_train_pred = lasso.predict(X_train)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
alpha = 0.0001

lasso = Lasso(alpha=alpha)

lasso.fit(X_train,y_train)
preds = lasso.predict(test_data)
final_predictions_lasso =np.exp(preds)
lasso.coef_
coeffs1 = np.squeeze (np.asarray(lasso.coef_))
print(coeffs1)
lasso_features = pd.Series(coeffs1, index = X_train.columns)
lasso_features.abs().sort_values(ascending=False)
lasso_values = pd.DataFrame({'Id': test_data.index ,#'SalePrice_actual': np.exp(y_test), 
                             'SalePrice_predicted':final_predictions_lasso })
lasso_values.head(8)
sub = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
sub.head()
sub['SalePrice'] = np.exp(preds)
sub.head()
sub.to_csv('submission.csv')
