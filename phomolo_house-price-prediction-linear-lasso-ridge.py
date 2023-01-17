#Import all libraries

import numpy as np 

import pandas as pd 

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from scipy import stats

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

%matplotlib inline

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn import linear_model

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn import tree

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline
#Read and load Data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# View head of train

train.head()
# View head of test

test.head()
# View train columns

train.columns
# View test columns

test.columns
# Shape of train

train.shape
# Shape of test

test.shape
#Info on our target variable

train.SalePrice.describe()
#Plot Histogram for 'SalePrice'

print ("Skew is:", train.SalePrice.skew())

print("Kurtosis : %f" % train['SalePrice'].kurt())

sns.distplot(train['SalePrice'], color = 'green')

plt.show()
#Skewness and Kurtosis - After log-transformation

target = np.log1p(train.SalePrice)

print("Skewness : %f" % target.skew())

print("Kurtosis : %f" % target.kurt())

sns.distplot(target, color = 'green')

plt.show()
# Displaying of all numerical features

numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
# Displaying of all categorical features

categorical_features = train.select_dtypes(include=[np.object])

categorical_features.dtypes
#Finding the correlations of in numeric features

corr = numeric_features.corr()

print (corr['SalePrice'].sort_values(ascending=False))
# Most correlated variables to clean outliers

corrmat = train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Compare 'SalePrice' and Overall Quality

quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

quality_pivot.plot(kind='bar', color='green', alpha = 0.5)

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
# SalePrice vs OverallQual - Graph on right shows most sought after houses are quality of 5

plt.figure(figsize= (15,6))

plt.subplot(121)

sns.boxenplot(train['OverallQual'], train['SalePrice'],palette="RdYlGn")

plt.subplot(122)

train['OverallQual'].value_counts().plot(kind="bar",color = 'green', alpha = 0.5)
# Houses with 6 rooms above ground are most in demand - SalePrice is also competitive mostly cheaper than 5 rooms

plt.figure(figsize= (15,6))

plt.subplot(121)

sns.boxenplot(train['TotRmsAbvGrd'], train['SalePrice'], palette="RdYlGn")

plt.subplot(122)

train['TotRmsAbvGrd'].value_counts().plot(kind="bar",color = 'green', alpha = 0.5)
# Two garage houses are the most prevalent, interestingly four garages SalePrice drops considerably

plt.figure(figsize= (15,6))

plt.subplot(121)

sns.boxenplot(train['GarageCars'], train['SalePrice'],palette="RdYlGn")

plt.subplot(122)

train['GarageCars'].value_counts().plot(kind="bar",color = 'green', alpha = 0.5)
# Houses with 1 or 2 full bathrooms seems to be the most prevalent, 3 full bathrooms only available in more expensive homes

plt.figure(figsize= (15,7))

plt.subplot(121)

sns.boxenplot(train['FullBath'], train['SalePrice'],palette='YlOrRd')

plt.subplot(122)

train['FullBath'].value_counts().plot(kind="bar",color = 'green', alpha = 0.5)
#Analyse SalePrice/GrLiveArea

data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)

data.plot.scatter(x ='GrLivArea', y= 'SalePrice', ylim = (0,800000), c= 'green', alpha = 0.5)
# Remove outliers from GrLivArea

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice']< 4000000)].index)
# After outliers have been removed

data = pd.concat([train['SalePrice'], train['GrLivArea']], axis = 1)

data.plot.scatter(x ='GrLivArea', y= 'SalePrice', ylim = (0,800000), c= 'red', alpha = 0.5)
# Plot all figures where possible outliers remain

fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(20, 10))

axes = np.ravel(axes)

col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']

for i, c in zip(range(5), col_name):

    train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, color='green', alpha = 0.5)



# Delete outliers for affected columns

train = train[train['GrLivArea'] < 4000]

train = train[train['LotArea'] < 100000]

train = train[train['TotalBsmtSF'] < 3000]

train = train[train['1stFlrSF'] < 2500]

train = train[train['BsmtFinSF1'] < 2000]



# Loop to   

for i, c in zip(range(5,10), col_name):

    train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, color='red', alpha = 0.5)
# Distribution is skewed to the right - Does not follow a normal distribution

# QQ plot confirms

sns.distplot(train['SalePrice'], fit = norm, color='green')

fig = plt.figure()

res = stats.probplot(train['SalePrice'],plot = plt)
#Log transformation - log(1+x) to account for smaller rounding errors

train['SalePrice'] = np.log1p(train['SalePrice'])
# Distribution is now more fitted to a normal distribution after log-transformation

sns.distplot(train['SalePrice'], fit = norm, color = 'green')

fig = plt.figure()

res = stats.probplot(train['SalePrice'],plot = plt)
# Show total amount of missing data

total = train.isnull().sum().sort_values(ascending = False)



# Show percentage of missing data

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)



# Concatenate new columns in dataframe

missing_data = pd.concat([total,percent], axis = 1, keys = ['Total', 'Percent'])

missing_data.head(20)
# Visualising missing data

f, ax = plt.subplots(figsize=(10,6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.head(15).index, y=missing_data.head(15)['Percent'], palette="ch:2.5,-.2,dark=.3")

plt.xlabel('Features')

plt.ylabel('Missing values %')

plt.title('Percent of Missing Data by Features')

plt.savefig('missingdata.png')

plt.show()
# View of categorical variables details - train dataset

categoricals = train.select_dtypes(exclude=[np.number])

categoricals.describe()
# View of categorical variables details - test dataset

cate = test.select_dtypes(exclude=[np.number])

cate.describe()
# Run for loop for view of all categorical variables and unique identifiers

for c in categoricals.columns:

    print('{:<14}'.format(c), train[c].unique())
# Label Encoding categorical features

columns = ('MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition')



# Apply LabelEncoder to all columns above

for i in columns:

    lbl = LabelEncoder() 

    lbl.fit(list(train[i].values)) 

    train[i] = lbl.transform(list(train[i].values))

    lbl.fit(list(test[i].values)) 

    test[i] = lbl.transform(list(test[i].values))
# Remove features that are not of any use to value the target variable

train=train.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating','PoolArea','PoolQC','MiscVal','MiscFeature'])

test=test.drop(columns=['Street','Utilities','Condition2','RoofMatl','Heating','PoolArea','PoolQC','MiscVal','MiscFeature'])
# Remove any columns up to 0.70 missing values in both training and test dataset

train = train.dropna(thresh=0.70*len(train), axis=1)

test = test.dropna(thresh=0.70*len(test), axis=1)
#We have to remove any missing data to make our model more robust - free from errors in the modelling phase

data = train.select_dtypes(include=[np.number]).interpolate().dropna()
#Setting up x and y variables  - ready for modelling

y = train.SalePrice

X = data.drop(['SalePrice', 'Id'], axis=1)
#Firstly setting up of our train and test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state =1)
#Linear Regression

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



#Create linear regression 

lm = linear_model.LinearRegression()



#Train the model using the training sets

lm.fit(X_train, y_train)



# Make predictions using the testing set

y_pred = lm.predict(X_test)

print('The accuracy of the Linear Regression is',r2_score(y_test,y_pred))

print ('RMSE is: ', mean_squared_error(y_test, y_pred))
# Ready for submission

submission = pd.DataFrame()

submission['Id'] = test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = lm.predict(feats)

lm_predictions = np.exp(predictions)
# Print results and save results to csv

print ("Original predictions are: \n", predictions[:5], "\n")

print ("Final predictions are: \n", lm_predictions[:5])

submission['SalePrice'] = lm_predictions

submission.head()

submission.to_csv('Linear Regression.csv', index=False)
# Lambda values for lasso fit

lambda_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3, 1]



# Train lasso

def train_lasso(X, Y, alpha):

    lasso = Lasso(alpha=alpha, max_iter=1000)

    lasso = lasso.fit(X, Y)

    return lasso



lasso_models = []



# Iterate lambda_values to train

for alpha in lambda_values:

    l = train_lasso(X_train, y_train, alpha)

    lasso_models.append(l)

    

# Return score for each lambda iteration    

for i, alpha in enumerate(lambda_values):

    print('Lambda value: ',alpha)

    y_pred_lasso = lasso_models[i].predict(X_test)

    print('The accuracy of the Linear Regression is',r2_score(y_test,y_pred_lasso))

    print ('RMSE is: ', mean_squared_error(y_test, y_pred_lasso))

    print('\n')
# Ready for submission

submission = pd.DataFrame()

submission['Id'] = test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

lasso_predictions = lasso_models[1].predict(feats)

final_lasso = np.exp(lasso_predictions)
# Print results and save results to csv

print ("Original predictions are: \n", lasso_predictions[:5], "\n")

print ("Final lasso predictions are: \n", final_lasso[:5])

submission['SalePrice'] = final_lasso

submission.head()

submission.to_csv('Lasso_Regression.csv', index=False)
#Ridge Regression

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



#Create Ridge regression 

ridge = Ridge()



#Train the model using the training sets

ridge.fit(X_train, y_train)

b = float(ridge.intercept_)

coeff = pd.DataFrame(ridge.coef_, X.columns, columns=['Coefficient'])

print("Intercept:", float(b))



# Make predictions using the testing set - Ridge Regression

test_ridge = ridge.predict(X_test)

print('The accuracy of the Ridge Regression is', r2_score(y_test, test_ridge))

print ('RMSE is: ', mean_squared_error(y_test, test_ridge))
# Ready for submission

submission = pd.DataFrame()

submission['Id'] = test.Id

feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

predictions = ridge.predict(feats)

final_ridge = np.exp(predictions)
# Print results and save results to csv

print ("Original predictions are: \n", predictions[:5], "\n")

print ("Final predictions are: \n", final_ridge[:5])

submission['SalePrice'] = final_ridge

submission.head()

submission.to_csv('Ridge_Regression.csv', index=False)