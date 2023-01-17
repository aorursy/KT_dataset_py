#Importing Libraries
import os
import pandas as pd # data processing
import numpy as np # linear algebra
import matplotlib.pyplot as plt # plotting library
import seaborn as sns # plotting library
# Load files

train = pd.read_csv("../input/home-data-for-ml-course/train.csv")
test = pd.read_csv("../input/home-data-for-ml-course/test.csv")

# train = pd.read_csv('train.csv') # training dataset
# test = pd.read_csv('test.csv')   # same dataset, without target `SalePrice`.

train_id = train['Id']
test_id = test['Id']

train.shape, test.shape #Just checking the shape
corrmat = train.corr()
plt.figure(figsize = (12, 10))
sns.heatmap(corrmat, square=True)
plt.title('Correlation between features');
plt.show()
#Plot of the second Heatmap with less features
corrMatrix=train[["SalePrice","OverallQual","GrLivArea","GarageCars",
                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",
                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

corrmat = train.corr()
plt.figure(figsize = (12, 10))
sns.heatmap(corrMatrix, square=True)
plt.title('Correlation between less features');
plt.show()
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
#The first five features are the most positively correlated with SalePrice,
#while the next five are the most negatively correlated.
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
#Plot of numerical values (Tried to visualise here but didn't really help out)
%matplotlib inline  
corrmat.hist(bins = 20 , figsize = (20,20))
plt.show()
#Plot of the features that have the greatest impact on the price (Proud of this one)
plt.figure(figsize = (12, 10))
corrmat['SalePrice'].sort_values(ascending = False).plot(kind = 'bar')
plt.show()
#Summarizing the data of OverallQual and SalePrice by using pivot_table, 
OverQuality_pivot = train.pivot_table(index='OverallQual',
                  values='SalePrice', aggfunc=np.median)
OverQuality_pivot
#Plot of the relationship between OverallQual and SalePrice,
OverQuality_pivot.plot(kind='bar', figsize = (12, 10))
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
from scipy import stats
from scipy.stats import norm, skew
#Plot of the price distribution
plt.figure(figsize = (12, 10))
sns.distplot(train_salePrice , fit = norm)
plt.show()
#Using the numpy fuction log1p which  applies log(1+x) to all elements of the column
plt.figure(figsize = (12, 10))
sns.distplot(np.log1p(train_salePrice) , fit = norm)
plt.show()
#Plot of GrLivArea feature to remove any outliers
plt.figure(figsize = (12, 10))
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.show()
#Plot of GrLivArea removing the outliers
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 700000)].index)
plt.figure(figsize = (12, 10))
plt.scatter(train['GrLivArea'], train['SalePrice'])
plt.show()
#Joining train and test datasets
train_shape = train.shape
test_shape = test.shape
train_salePrice = train['SalePrice']
dataset = pd.concat((train, test)).reset_index(drop = True)
#Plot of columns that have NaN values
#Cleaning the data set to have a stronger analysis by removing NaN values

nan_columns = [column for column in dataset.columns if dataset[column].isnull().values.any()]
plt.figure(figsize = (12, 10))
dataset[nan_columns].isnull().sum().sort_values(ascending = False).plot(kind = 'bar')
plt.show()
def dummy_variable(dataset, column, variable):
    tmp = pd.get_dummies(dataset[column])
    tmp = tmp.drop(variable, axis = 1)
    for col in tmp.columns:
        dataset[column , '_' , col] = tmp[col]
    return dataset.drop(column, axis = 1)
#Categorical variable (PoolQC)
dataset['PoolQC'] = dataset['PoolQC'].fillna(0)
d = {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}
dataset['PoolQC'] = dataset['PoolQC'].replace(d)
#The pd.dummy_variable() for MiscFeature and removing NA column
dataset['MiscFeature'] = dataset['MiscFeature'].fillna('NA')
dataset = dummy_variable(dataset, 'MiscFeature', 'NA')
#The dummy_variable() for Alley and removing NA column
dataset['Alley'] = dataset['Alley'].fillna('NA')
dataset = dummy_variable(dataset, 'Alley', 'NA')
#The dummy_variable() for Fence and removing NA column
dataset['Fence'] = dataset['Fence'].fillna('NA')
dataset = dummy_variable(dataset, 'Fence', 'NA')
#Categorical variable (FireplaceQu)
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(0)
d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
dataset['FireplaceQu'] = dataset['FireplaceQu'].replace(d)
dataset['LotFrontage'] = dataset.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#Categorical variable (GarageQual & GarageCond)
dataset['GarageQual'] = dataset['GarageQual'].fillna(0)
dataset['GarageCond'] = dataset['GarageCond'].fillna(0)
d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
dataset['GarageQual'] = dataset['GarageQual'].replace(d)
dataset['GarageCond'] = dataset['GarageCond'].replace(d)
#Categorical variable (GarageFinish)
dataset['GarageFinish'] = dataset['GarageFinish'].fillna(0)
d = {'Unf': 1, 'RFn': 2, 'Fin': 3}
dataset['GarageFinish'] = dataset['GarageFinish'].replace(d)
#Replacing NaN values with 0
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0)
#The dummy_variable() for GarageType and removing NA column
dataset['GarageType'] = dataset['GarageType'].fillna('NA')
dataset = dummy_variable(dataset, 'GarageType', 'NA')
#Categorical variable (BsmtExposure)
dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(0)
d = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
dataset['BsmtExposure'] = dataset['BsmtExposure'].replace(d)
#Categorical variable (BsmtCond & BsmtQual)
dataset['BsmtCond'] = dataset['BsmtCond'].fillna(0)
dataset['BsmtQual'] = dataset['BsmtQual'].fillna(0)
d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
dataset['BsmtCond'] = dataset['BsmtCond'].replace(d)
dataset['BsmtQual'] = dataset['BsmtQual'].replace(d)
#Categorical variable (BsmtFinType2 & BsmtFinType1)
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(0)
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(0)
d = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
dataset['BsmtFinType2'] = dataset['BsmtFinType2'].replace(d)
dataset['BsmtFinType1'] = dataset['BsmtFinType1'].replace(d)
#The dummy_variable() for MasVnrType and removing NA column
dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
dataset = dummy_variable(dataset, 'MasVnrType', 'None')
#Replacing NaN values with 0 for MasVnrArea
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
#Replacing NaN values with the most frequent element because there are only 4 NaN values and then
#using pd.get_dummies() and remove column
dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].value_counts().index[0])
dataset = dummy_variable(dataset, 'MSZoning', 'C (all)')
#Replacing NaN values with 0 for BsmtFullBath & BsmtHalfBath
dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
#Removing Utilities column because all the elements are 'AllPub' and there are only 2 NaN values
dataset = dataset.drop('Utilities', axis = 1)
#Replacing NaN values with the most frequent element because there are only 2 NaN values and then
#using pd.get_dummies() and remove column
dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].value_counts().index[0])
dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].value_counts().index[0])
dataset = dummy_variable(dataset, 'Functional', 'Sev')
dataset = dummy_variable(dataset, 'Electrical', 'Mix')
#Replacing NaN values with 0 for BsmtFullBath & BsmtHalfBath
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(0)
#Replacing NaN values with the most frequent element because there are only 2 NaN values and then
#using pd.get_dummies() and remove column
dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].value_counts().index[0])
dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].value_counts().index[0])
dataset = dummy_variable(dataset, 'Exterior1st', 'AsbShng')
dataset = dummy_variable(dataset, 'Exterior2nd', 'AsphShn')
#Replacing NaN values with 0 for BsmtFullBath & BsmtHalfBath
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)
dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0)
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(0)
#categorical variable (convertible in ordinal variable)
dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].value_counts().index[0])
d = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
dataset['KitchenQual'] = dataset['KitchenQual'].replace(d)
#Replacing NaN values with the most frequent element because there are only 1 NaN values and then
#using pd.get_dummies() and remove Oth column
dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].value_counts().index[0])
dataset = dummy_variable(dataset, 'SaleType', 'Oth')
#Replacing NaN values with 0 for GarageArea
dataset['GarageArea'] = dataset['GarageArea'].fillna(0)
#Creating a new column which contains the sum of TotalBsmtSF, 1stFlrSF, 2ndFlrSF
dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
#Having only the non-numeric columns
dataset.select_dtypes(exclude = [np.number]).columns
#And converting each non-numeric column to a column that contains values of 0 or 1
for column in dataset.select_dtypes(exclude = [np.number]).columns:  
    dataset = dummy_variable(dataset, column, dataset[column].value_counts().index[-1])
#deleting Id and SalePrice columns 
dataset = dataset.drop(['Id', 'SalePrice'], axis = 1)
#Now we can re-split the dataset
train = dataset[ : train_shape[0]]
test = dataset[train_shape[0] : ]
#check shapes of train and test (all good !)
train.shape, test.shape
#Tools 
from sklearn.metrics import mean_squared_error # Mean squared error regression loss
from sklearn.model_selection import GridSearchCV # GridSearchCV implements a “fit” and a “score” method
from sklearn.model_selection import KFold # statistical method used to estimate the skill of machine learning models.
from sklearn.model_selection import cross_val_score # cross_val_score estimates the expected accuracy of your model on out-of-training data
from sklearn.preprocessing import RobustScaler # This Scaler removes the median and scales the data according to the quantile range, standardization of a dataset is a common requirement for many machine learning estimators.
from sklearn.pipeline import Pipeline # Sequentially apply a list of transforms and a final estimator


# Regression model
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge 
from sklearn.linear_model import RidgeCV 
from sklearn.linear_model import ElasticNet 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 

# Boosting
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.ensemble import AdaBoostRegressor 
from xgboost import XGBRegressor # (The mighty model???) 
X = train
y = np.log1p(train_salePrice.values)
scoring = 'neg_mean_squared_error'
# Testing various regression algorithms.
# Creating an empty list and appending each models
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('RIDGE', Ridge()))
models.append(('RIDGECV', RidgeCV()))
models.append(('EN', ElasticNet()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('SVR', SVR()))
#Testing various regression algorithms

results = []
names = []
m, model_name = float('-inf'), ''
for name, model in models:
    kfold = KFold()
    cross_val_res = cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    print('Model:', name, '\tMean:', cross_val_res.mean(), '\tStd:', cross_val_res.std())
    results.append(cross_val_res)
    names.append(name)
    if cross_val_res.mean() > m:
        m = cross_val_res.mean()
        model_name = name
print('\nBest model:', '\t' + model_name)
#Plot of the statistical distribution of the chosen algorithms
# We need to scale, we can't compare the algorithms here
fig = plt.figure(figsize = (12, 10))
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#testing various regression algorithms with the feature scaling
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', RobustScaler()), ('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', RobustScaler()), ('LASSO', Lasso())])))
pipelines.append(('ScaledRIDGE', Pipeline([('Scaler', RobustScaler()), ('RIDGE', Ridge())])))
pipelines.append(('ScaledRIDGECV', Pipeline([('Scaler', RobustScaler()), ('RIDGECV', RidgeCV())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', RobustScaler()), ('EN', ElasticNet())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', RobustScaler()), ('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', RobustScaler()), ('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', RobustScaler()), ('SVR', SVR())])))
results = []
names = []
m, model_name = float('-inf'), ''
for name, model in pipelines:
    kfold = KFold()
    cross_val_res = cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    print('Model:', name, '\tMean:', cross_val_res.mean(), '\tStd:',cross_val_res.std())
    results.append(cross_val_res)
    names.append(name)
    if cross_val_res.mean() > m:
        m = cross_val_res.mean()
        model_name = name
print('\nBest model:', '\t' + model_name)
#Plot of the statistical distribution of the chosen algorithms
fig = plt.figure(figsize = (12, 10))
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#RidgeCV tuning (finding the best alphas parameter) The number of sequential trees to be modeled
alpha_values = []
value = 27.0
while value <= 30.0:
    alpha_values.append([value])
    value += 0.1
param_grid = dict(alphas = alpha_values)
kfold = KFold()
grid = GridSearchCV(estimator = RidgeCV(), param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(X, y)
print('Best score: %f using %s\n' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))
    
best_alpha = grid_result.best_params_['alphas']
#testing various boosting algorithms
# create an empty list and appending each models
models = []
models.append(('AB', AdaBoostRegressor()))
models.append(('GBM', GradientBoostingRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('XGB', XGBRegressor()))
models.append(('ET', ExtraTreesRegressor()))
results = []
names = []
m, model_name = float('-inf'), ''
for name, model in models:
    kfold = KFold()
    cross_val_res = cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    print('Model:', name, '\tMean:', cross_val_res.mean(), '\tStd:',cross_val_res.std())
    results.append(cross_val_res)
    names.append(name)
    if cross_val_res.mean() > m:
        m = cross_val_res.mean()
        model_name = name
print('\nBest model:', '\t' + model_name)
#Plot of the statistical distribution of the chosen algorithms
fig = plt.figure(figsize = (12, 10))
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#Testing various boosting algorithms with the feature scaling
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', RobustScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', RobustScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', RobustScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledXGB', Pipeline([('Scaler', RobustScaler()),('XGB', XGBRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', RobustScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
m, model_name = float('-inf'), ''
for name, model in ensembles:
    kfold = KFold()
    cross_val_res = cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    print('Model:', name, '\tMean:', cross_val_res.mean(), '\tStd:',cross_val_res.std())
    results.append(cross_val_res)
    names.append(name)
    if cross_val_res.mean() > m:
        m = cross_val_res.mean()
        model_name = name
print('\nBest model:', '\t' + model_name)
# Plot of the statistical distribution of the chosen algorithms 
# (We're trying to have the mean that is closer to 0)
fig = plt.figure(figsize = (12, 10))
fig.suptitle('Algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#GBM tuning (finding the best n_estimators parameter) The number of sequential trees to be modeled

param_grid = dict(n_estimators = np.array([350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400]))
model = GradientBoostingRegressor()
kfold = KFold()
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(X, y)
print('Best score: %f using %s\n' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))
    
best_n_estimators = grid_result.best_params_['n_estimators']
#Ensembling RidgeCV (70%) and Gradient Boosting (30%) without feature scaling
print('Predicting house prices with:\n')
print('RidgeCV(alphas = ' + str(best_alpha) + ')')
print('GradientBoostingRegressor(n_estimators = '+ str(best_n_estimators) + ')')

ridge_cv = RidgeCV(alphas = best_alpha)
ridge_cv_predictions = ridge_cv.fit(X, y).predict(test)

gbm = GradientBoostingRegressor(n_estimators = best_n_estimators)
gbm_predictions = gbm.fit(X, y).predict(test)

predictions = np.expm1(ridge_cv_predictions) * 0.7 + np.expm1(gbm_predictions) * 0.3
# Prepare submission CSV to send to Kaggle
submission = pd.DataFrame()
submission['Id'] = test_id
submission['SalePrice'] = predictions
submission.to_csv('submission.csv', index = False)
submission