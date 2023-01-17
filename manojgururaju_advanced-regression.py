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
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import os

# hide warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
# Reading the dataset

data = pd.read_csv("../input/house-data/train.csv")
data.head()
data.shape
data.describe()
data.info()
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# plotting the correlation
plt.figure(figsize=(30,15))

# heatmap
sns.heatmap(numeric_data.corr(), annot=True)
plt.show()
numeric_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# capping the outlier
for column in numeric_data.columns:
    q1 = numeric_data[column].quantile(0.1)
    q3 = numeric_data[column].quantile(0.9)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    numeric_data.loc[numeric_data[column] > upper_limit, column] = upper_limit
    numeric_data.loc[numeric_data[column] < lower_limit, column] = lower_limit

numeric_data.describe(percentiles=[.25, .5, .75, .90, .95, .99])
percent_missing = data.isnull().sum() * 100 / len(data)
missing_value = pd.DataFrame({'column_name': data.columns,
                                 'percent_missing': percent_missing})
missing_value.iloc[missing_value.percent_missing.to_numpy().nonzero()]

# replacing missing data with approprtiate values

# replacing LotFrontage na values to mean
mean = data['LotFrontage'].mean()
data['LotFrontage'].fillna(mean, inplace=True) 

# replacing MasVnrArea na values to 0 
mean = data['MasVnrArea'].mean()
data['MasVnrArea'].fillna(mean, inplace=True) 

# replacing MasVnrType na values to None
data['MasVnrType'].fillna('None', inplace=True)

# replacing feature with na values refering speacial reference
data['Alley'].fillna('No Alley', inplace=True)
data['FireplaceQu'].fillna('No Fireplace', inplace=True)
data['PoolQC'].fillna('No Pool', inplace=True) 
data['Fence'].fillna('No Fence', inplace=True) 
data['GarageType'].fillna('No Garage', inplace=True) 
data['GarageFinish'].fillna('No Garage', inplace=True) 
data['GarageQual'].fillna('No Garage', inplace=True) 
data['GarageCond'].fillna('No Garage', inplace=True) 
# year data to number of years to 2020
data['YearBuilt'] = 2020 - data['YearBuilt']
data['YearRemodAdd'] = 2020 - data['YearRemodAdd']
data['GarageYrBlt'] = 2020 - data['GarageYrBlt']
data['YrSold'] = 2020 - data['YrSold']
data['MiscFeature'].replace({np.nan:'None'},inplace=True)
list(data.columns)
# As there multiple basement related feature there is possiblity to eliminate some
# pairwise scatter plot to explore Basement attributes

basement_col=data[['BsmtQual','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']]
plt.figure(figsize=(20, 10))
sns.heatmap(basement_col.corr(), annot=True)
plt.show()
# Dropping of correlated variables and keeping only TotalBsmtSF

data=data.drop(['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF'],axis=1)
# Plotting the categorical variables related to Basement to find which ones have correlation and can be dropped

plt.figure(figsize=(20, 12))
plt.subplot(3,3,1)
sns.boxplot(x = 'BsmtQual', y = 'SalePrice', data = data)
plt.subplot(3,3,2)
sns.boxplot(x = 'BsmtExposure', y = 'SalePrice', data = data)
plt.subplot(3,3,3)
sns.boxplot(x = 'BsmtFinType1',y = 'SalePrice', data = data)
plt.subplot(3,3,4)
sns.boxplot(x = 'BsmtFinType2',y = 'SalePrice', data = data)
# dropping variables as they are not influence sale

data=data.drop(['BsmtFinType1','BsmtFinType2'],axis=1)
# Corelation between Pool and Sales

plt.figure(figsize=(15, 12))
plt.subplot(3,3,1)
sns.boxplot(x = 'PoolQC', y = 'SalePrice', data = data)
plt.subplot(3,3,2)
sns.boxplot(x = 'PoolArea', y = 'SalePrice', data = data)
# Pool with excelent condition will imporve the price of the house to high and Pool Area can be dropped

data=data.drop(['PoolQC'],axis=1)
# Corelation between Garage condition and qualities [ sounds same] and Sales

plt.figure(figsize=(15, 12))
plt.subplot(3,3,1)
sns.boxplot(x = 'GarageCond', y = 'SalePrice', data = data)
plt.subplot(3,3,2)
sns.boxplot(x = 'GarageQual', y = 'SalePrice', data = data)
# Looks like garage condition and qualities has similar influence for some extent, we can keep GarageQual 

data=data.drop(['GarageQual'],axis=1)
# correlation between porchs

porch=data[['OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']]

plt.figure(figsize=(15, 10))
sns.pairplot(porch)
plt.show()
data.info()
# Correlation between all numeric data

plt.figure(figsize=(20, 12))
sns.heatmap(data.select_dtypes(include=['float64', 'int64']).corr(), cmap="YlGnBu", annot=True)
plt.show()
data = data.drop('Id', axis=1)
X = data[['MSSubClass','MSZoning','LotArea','Street','LotShape','LandContour','Utilities',
                    'LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',
                    'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                    'ExterQual','ExterCond','Foundation','BsmtQual','BsmtExposure','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical',
                    '1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                    'BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu',
                    'GarageType','GarageFinish','GarageCars','GarageArea','GarageCond','PavedDrive',
                    'WoodDeckSF','OpenPorchSF','EnclosedPorch','PoolArea','Fence',
                    'MiscVal','MoSold','YrSold','SaleType','SaleCondition', 'MiscFeature']]

y = data['SalePrice']
X.head()
category = X.select_dtypes(include=['object'])
category.head()
# categorical variables into dummies
category_dumm = pd.get_dummies(category, drop_first=True)
category_dumm.head()
# dropping original category colums

X = X.drop(list(category.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, category_dumm], axis=1)
# scaling the features
from sklearn.preprocessing import scale

cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns
X.shape
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
data.head()
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# list of alpha for tuning the model
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
# cross validation
folds = 5

lasso = Lasso()

model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_cv.fit(X_train, y_train) 
# comparision of models with various alpha

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=200]
cv_results.head(20)
# plotting mean test and train scores with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'])
plt.show()
model_cv.best_params_
# lasso regression
lm1 = Lasso(alpha=500)
lm1.fit(X_train, y_train)

# predict
y_train_pred = lm1.predict(X_train)
y_test_pred = lm1.predict(X_test)


# r square calculation
r_square_score=r2_score(y_test,y_test_pred)
print("R Square score:", r_square_score)
# metrics for evaluation

n= X_train.shape[0] # number of data points
k= X_train.shape[1] # number of predictor variables in model built
resid=np.subtract(y_test_pred,y_test)
rss=np.sum(np.square(resid))
print("RSS:{}".format(rss))
aic=n*np.log(rss/n)+2*k
print("AIC:{}".format(aic))
bic=n*np.log(rss/n)+k*np.log(n)
print("BIC:{}".format(bic))
# split into X and y 

X = data[['MSSubClass','LotArea','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual',
                    'OverallCond','YearBuilt','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                    'ExterQual','Foundation','BsmtQual','BsmtExposure','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical',
                    '2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath',
                    'KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu',
                    'GarageType','GarageFinish','GarageCars','GarageCond','WoodDeckSF','EnclosedPorch','PoolArea',
                    'SaleType','SaleCondition']]

y = data['SalePrice']
category = X.select_dtypes(include=['object'])
category.head()
# categorical variables into dummies

category_dumm = pd.get_dummies(category, drop_first=True)
category_dumm.head()
# dropping original category colums

X = X.drop(list(category.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, category_dumm], axis=1)
# scaling the features
from sklearn.preprocessing import scale

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
# plotting mean test and train scores with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'])
plt.show()
model_cv.best_params_
# Ridge regression

lm1 = Ridge(alpha=500)
lm1.fit(X_train, y_train)

# predict
y_train_pred = lm1.predict(X_train)
y_test_pred = lm1.predict(X_test)


# r square calculation
r_square_score=r2_score(y_test,y_test_pred)
print("R Square score:", r_square_score)
model_parameter = list(lm1.coef_)
model_parameter.insert(0,lm1.intercept_)
model_parameter = [round(x,3) for x in model_parameter]
col = X_train.columns
col.insert(0,'Constant')
ridge_coef = list(zip(col,model_parameter))
# metrics for evaluation

n= X_train.shape[0] # number of data points
k= X_train.shape[1] # number of predictor variables in model built
resid=np.subtract(y_test_pred,y_test)
rss=np.sum(np.square(resid))
print("RSS:{}".format(rss))
aic=n*np.log(rss/n)+2*k
print("AIC:{}".format(aic))
bic=n*np.log(rss/n)+k*np.log(n)
print("BIC:{}".format(bic))