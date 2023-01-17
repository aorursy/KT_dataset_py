import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import scipy.stats as stats
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
# 2.1 Check number of observations and columns
print(train.shape)
print(test.shape)
# Train has one more column the predictor variable SalePrice than the test data
print(set(train.columns)-set(test.columns))
# 2.2 Check the number of observations, data types and NAN for each variable
print(train.info())
print('-----------------------'*3)
print(test.info())
# 2.3 Check the descriptive info for 37 numeric variables, exclude Id
train.drop(['Id'], axis=1).describe()
# Check how many numeric columns and how many object columns, 38 numeric columns including ID
# 43 character columns
print(train.select_dtypes(exclude=['object']).shape)
print(train.select_dtypes(include=['object']).shape)
# 2.4 Check the first 5 observations and last five observations of train and head
#print(train.head(5))
print(train[:5])
print('/n')
print(train.tail(5))
#print(train[-5:])
plt.figure(figsize=(13,10))
sns.heatmap(train.corr(), vmax=0.8)
train.drop(['GarageYrBlt','TotRmsAbvGrd','TotalBsmtSF'], axis=1, inplace=True)
test.drop(['GarageYrBlt','TotRmsAbvGrd','TotalBsmtSF'], axis=1, inplace=True)
# Top 10 numeric features positively correlated with SalePrice
train.corr()['SalePrice'].sort_values(ascending=False).head(11)
train.corr()['SalePrice'].abs().sort_values(ascending=False).head(11)
top_corr_features=train.corr()['SalePrice'].abs().sort_values(ascending=False).head(11).index
top_corr_features
# Box-plot to check relationship between SalePrice and OverallQual
plt.figure(figsize=(10,7))
sns.boxplot(x='OverallQual', y='SalePrice', data=train)
# Scatterplot to check the relationship between SalePrice and GrLivArea
plt.scatter(x='GrLivArea', y='SalePrice', data=train, color='r', marker='*')
train['GrLivArea'].sort_values(ascending=False).head(2)
train.index[[523, 1298]]
print(train.shape)
train.drop(train.index[[523, 1298]], inplace=True)
print(train.shape)
print(top_corr_features)
box_feature=['SalePrice','OverallQual','GarageCars','FullBath', 'YearBuilt','YearRemodAdd','Fireplaces']
scatter_feature=['SalePrice', 'GrLivArea','1stFlrSF','GarageArea']
# Use sns.pairplot to check the relationship between the SalePrice and top 10 correlated features
sns.pairplot(train[scatter_feature])
sns.pairplot(train[box_feature], kind='scatter', diag_kind='hist')
train.isnull().sum().sort_values(ascending=False)
# Check the NAN values as percentage
train_nan_pct=(train.isnull().sum())/(train.isnull().count())
train_nan_pct=train_nan_pct[train_nan_pct>0]
train_nan_pct.sort_values(ascending=False)
train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1, inplace=True)
train['GarageQual'].value_counts()
train_impute_index=train_nan_pct[train_nan_pct<0.3].index
train_impute_index
train_impute_mode=['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
train_impute_median=['LotFrontage', 'MasVnrArea']
# Impute character or qualitative feature with mode
for feature in train_impute_mode:
    train[feature].fillna(train[feature].mode()[0], inplace=True)
    test[feature].fillna(test[feature].mode()[0], inplace=True)
# Impute numeric feature with median
for feature in train_impute_median:
    train[feature].fillna(train[feature].median(), inplace=True)
    test[feature].fillna(test[feature].median(), inplace=True)
# There are no nan values in train
train.isnull().sum().sort_values(ascending=False).head(5)
test_only_nan=test.isnull().sum().sort_values(ascending=False)
test_only_nan=test_only_nan[test_only_nan>0]
print(test_only_nan.index)
test_impute_mode=['MSZoning', 'BsmtFullBath', 'Utilities','BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 'Exterior1st', 'GarageCars', 'KitchenQual']
test_impute_median=['BsmtFinSF2','GarageArea', 'BsmtFinSF1','BsmtUnfSF' ]
# Impute test character feature with mode
for feature in test_impute_mode:
    test[feature].fillna(test[feature].mode()[0], inplace=True)
for feature in test_impute_median:
    test[feature].fillna(test[feature].median(), inplace=True)
#Impute test numeric feature with median
# Now there are no NAN values in both train and test data
test.isnull().sum().sort_values(ascending=False).head(5)
# Store the test data ID for competition purpose
TestId=test['Id']
total_features=pd.concat((train.drop(['Id','SalePrice'], axis=1), test.drop(['Id'], axis=1)))
total_features=pd.get_dummies(total_features, drop_first=True)
train_features=total_features[0:train.shape[0]]
test_features=total_features[train.shape[0]:]
sns.distplot(train['SalePrice'])
# The response variable is right-skewed, we will log1p() transform y
train['Log SalePrice']=np.log1p(train['SalePrice'])
sns.distplot(train['Log SalePrice'])
# natural log one plus the array log(y+1) is more symmetric 
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.kdeplot(train['SalePrice'], legend=True)
plt.subplot(1,2,2)
sns.kdeplot(train['Log SalePrice'], legend=True)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train_features, train[['SalePrice']], test_size=0.3, random_state=100)
# Import Ridge regression from sklearn
from sklearn.linear_model import Ridge
# Evaluate model performance using root mean square error
from sklearn.metrics import mean_squared_error
rmse=[]
# check the below alpha values for Ridge Regression
alpha=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
rmse.argmin()
# Adjust alpha based on previous result
alpha=np.arange(8,14, 0.5)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())
# Adjust alpha based on previous result
alpha=np.arange(10.5, 11.6, 0.1)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())
# Use alpha=11.1 to predict the test data
ridge=Ridge(alpha=11.1)
# Use all training data to fit the model
ridge.fit(train_features, train[['SalePrice']])
predicted=ridge.predict(test_features)
submission=pd.DataFrame()
submission['Id']=TestId
submission['SalePrice']=predicted
submission.to_csv('submission.csv', index=False)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train_features, train[['Log SalePrice']], test_size=0.3, random_state=100)
# Import Ridge regression from sklearn
from sklearn.linear_model import Ridge
# Evaluate model performance using root mean square error
from sklearn.metrics import mean_squared_error
rmse=[]
# check the below alpha values for Ridge Regression
alpha=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())
print(rmse.min())
# Adjust alpha based on previous result
alpha=np.arange(8,14, 0.5)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print(rmse.argmin())
print(rmse.min())
# Adjust alpha based on previous result
alpha=np.arange(10.5, 11.5, 0.1)
rmse=[]

for alph in alpha:
    ridge=Ridge(alpha=alph, copy_X=True, fit_intercept=True)
    ridge.fit(X_train, y_train)
    predict=ridge.predict(X_val)
    rmse.append(np.sqrt(mean_squared_error(predict, y_val)))
print(rmse)
plt.scatter(alpha, rmse)
rmse=pd.Series(rmse, index=alpha)
print('Minimum RMSE at alpah: ', rmse.argmin())
print('Minimum RMSE is: ', rmse.min())
# Use alpha=11 to predict the test data
ridge=Ridge(alpha=11)
# Use all training data to fit the model
ridge.fit(train_features, train[['Log SalePrice']])
predicted_log_price=ridge.predict(test_features)
# Transform back the log(SalePrice+1) to SalePrice
Test_price=np.exp(list(predicted_log_price))-1
Test_price
submission=pd.DataFrame()
submission['Id']=TestId
submission['SalePrice']=Test_price
submission.to_csv('submission.csv', index=False)