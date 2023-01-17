# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
!cat /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
import seaborn as sns
sns.distplot(train.SalePrice)
train.columns
import matplotlib.pyplot as plt
plt.figure(figsize = (20,10))
plt.title('Missing values')
sns.heatmap(train.isna() == False, cbar = False)
plt.figure(figsize = (20,15))
sns.barplot(y = train.corr().SalePrice.index, x = train.corr().SalePrice)
plt.xlabel('Correlation')
plt.show()
categories = list(train.select_dtypes(include = 'object').columns)
categories.append('YrSold')
numerical = list(train.select_dtypes(exclude = 'object').columns)
numerical.remove('YrSold')
numerical.remove('SalePrice')
len(categories)
outdoor = ['LotFrontage', 'LotArea', 'Street','Alley','LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Exterior1st', 'Exterior2nd', 'ExterQual','ExterCond' , 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal' ]
indoor = ['BldgType','1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu']
location = ['MSZoning', 'Neighborhood', 'Condition1','Condition2']
materials = ['MSSubClass', 'Utilities', 'HouseStyle', 'OverallQual', 'OverallCond','YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'MasVnrType','MasVnrArea' , 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical' ]
otherindoor = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond']
sale = ['MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
plt.figure(figsize = (20,10))
plt.subplot(3,2,1)
cols = outdoor
cols.append('SalePrice')
sns.barplot(y = train[cols ].corr().SalePrice.index, x = train[cols].corr().SalePrice)
plt.subplot(3,2,2)
cols = indoor
cols.append('SalePrice')
sns.barplot(y = train[cols ].corr().SalePrice.index, x = train[cols].corr().SalePrice)
plt.subplot(3,2,3)
cols = materials
cols.append('SalePrice')
sns.barplot(y = train[cols ].corr().SalePrice.index, x = train[cols].corr().SalePrice)
plt.subplot(3,2,4)
cols = otherindoor
cols.append('SalePrice')
sns.barplot(y = train[cols ].corr().SalePrice.index, x = train[cols].corr().SalePrice)
print(train.YrSold.unique())
sns.barplot(x = train.YrSold, y = train.SalePrice)
plt.figure()
sns.scatterplot(x = train['GrLivArea'], y = train.SalePrice)
plt.figure()
sns.scatterplot(x = train['LotArea'], y = train.SalePrice)
plt.figure(figsize = (10,5))
sns.barplot(x = train.groupby('Neighborhood').mean().SalePrice, y = train.groupby('Neighborhood').mean().index)
plt.figure(figsize = (10,5))
sns.barplot(x = train.groupby('HouseStyle').mean().SalePrice, y = train.groupby('HouseStyle').mean().index)
from sklearn.model_selection import train_test_split
X = train.drop('SalePrice', axis = 1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)
X_train
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
numerical_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),OneHotEncoder(handle_unknown = 'ignore', sparse = False) )
prepro = make_column_transformer((numerical_pipeline, numerical), (categorical_pipeline, categories), sparse_threshold = 0)
pipeline = make_pipeline(prepro,DecisionTreeRegressor() )
pipeline.fit(X_train, y_train)
prepro.fit_transform(X_train, y_train)
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import learning_curve, cross_val_score

def FE_evaluation(prepro):
    X_pre = prepro.fit_transform(X_train, y_train)
    model = DecisionTreeRegressor()
    train_sizes, train_scores, val_scores = learning_curve(model, X_pre, y_train, train_sizes = np.linspace(0.1,1,10) , shuffle = True, random_state = 0)
    
    
   
    return cross_val_score(model, X_pre, y_train, cv =3).mean(axis = 0)
FE_evaluation(prepro)
def divide(features):
    cat = list(train[features].select_dtypes(include = 'object').columns)
    num = list(train[features].select_dtypes(exclude = 'object').columns)
    
    
    return cat, num
def tryfeatures(features):
    cat, num = divide(features)
    prepro = make_column_transformer((numerical_pipeline, num), (categorical_pipeline, cat))
    return FE_evaluation(prepro)

missing_values = train.drop('SalePrice', axis = 1).isna().sum()/train.shape[0]
missing_values.sort_values(ascending = False, inplace = True)
for tr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4,]:
    new_columns = list(missing_values[missing_values < tr].index)
    print (tryfeatures(new_columns))
scores = []
def SelectBestCorr(threshold):
    corr = train.corr()[train.corr().SalePrice > threshold]
    corr_features = list(corr.SalePrice.index.drop('SalePrice'))
    catcorr = train.drop(numerical, axis = 1).copy()
    catcorr = pd.get_dummies(catcorr) 
    catcorr = catcorr.corr(method = 'pearson')
    cat_threshold = catcorr[abs(catcorr.SalePrice) > threshold]
    new_cats = list(set(list(cat_threshold.index.str.split('_').str[0])))
    new_cats.remove('SalePrice')
    return new_cats, corr_features
SelectBestCorr(0.5)
num, cat = SelectBestCorr(0.5)
missing_values = train.drop('SalePrice', axis = 1).isna().sum()/train.shape[0]
missing_values.sort_values(ascending = False, inplace = True)
new_columns = list(missing_values[missing_values < 0.05].index)
features = list(set(new_columns) & set( num + cat))
tryfeatures(num+cat), tryfeatures(new_columns), tryfeatures(features)
X_train, X_test, y_train, y_test = train_test_split(train[features], train['SalePrice'], random_state = 1)
to_pred = test[features]
to_pred
from sklearn.preprocessing import PolynomialFeatures
cat, num = divide(features)
numerical_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler() )
categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),OneHotEncoder(handle_unknown = 'ignore', sparse = False) )
prepro = make_column_transformer((numerical_pipeline, num), (categorical_pipeline, cat), sparse_threshold = 0)
pipeline = make_pipeline(prepro,DecisionTreeRegressor() )
pipeline.fit(X_train, y_train)
pipeline
def FE_evaluation2(prepro):
    X_pre = prepro.fit_transform(X_train, y_train)
    model = DecisionTreeRegressor()
    return cross_val_score(model, X_pre, y_train, cv =3).mean(axis = 0)
FE_evaluation2(prepro)

def evaluation(model):
    #X_pre = prepro.fit_transform(X_train, y_train)
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, train_sizes = np.linspace(0.1,1,10) )
    plt.figure()
    plt.plot(train_sizes, train_scores.mean(axis = 1), label = 'train')
    plt.plot(train_sizes, val_scores.mean(axis = 1), label = 'val')
    return val_scores.mean(axis=1)[-1] 

evaluation(pipeline)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso
cat, num = divide(features)
numerical_pipeline = make_pipeline(SimpleImputer(strategy = 'mean'), StandardScaler() )
categorical_pipeline = make_pipeline(SimpleImputer(strategy = 'most_frequent'),OneHotEncoder(handle_unknown = 'ignore', sparse = False) )
prepro = make_column_transformer((numerical_pipeline, num), (categorical_pipeline, cat), sparse_threshold = 0)
pipeline1 = make_pipeline(prepro,RandomForestRegressor(n_estimators = 150) )
pipeline2 = make_pipeline(prepro,AdaBoostRegressor() )
pipeline3 = make_pipeline(prepro, LinearRegression())
pipeline4 = make_pipeline(prepro, PolynomialFeatures(2), LinearRegression())
pipeline5 = make_pipeline(prepro, Ridge())
pipeline6 = make_pipeline(prepro, Lasso())
models = [pipeline1, pipeline5, pipeline6]
for model in models :
    print (evaluation(model))

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(RandomForestRegressor(), param_grid = {'n_estimators' : [50, 100, 150]})
grid.fit(prepro.fit_transform(X_train), y_train)
grid.best_params_
pipeline1.fit(X_train, y_train)
pipeline1.score(X_test, y_test)

submission = pd.Series(data =pipeline1.predict(to_pred), index = test['Id'] , name = 'SalePrice')
submission.to_csv('/kaggle/working/apparts.csv')
submission