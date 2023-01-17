# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Set up code checking
import os
if not os.path.exists("../input/train.csv"):
    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  
    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
home_data.tail()
home_data.shape
features = [i for i in home_data.columns if (i!="Id") & (i!="SalePrice")]
target = home_data["SalePrice"]
num_feature =[]
object_feature =[]
total_missing = 0
total_cells = len(features)* len(home_data)
for c in features:
    missing_count = home_data[c].isnull().sum()
    total_missing = total_missing + missing_count
    
    if (home_data[c].dtypes !="object"):
            num_feature.append(c)
    else:
            object_feature.append(c)

    if missing_count >0:
        print(c,"",home_data[c].dtype,"",missing_count)
# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print ("percent of missing data: {:.2f}%".format(percent_missing))
print("numerical features: ",num_feature)
print("non-numerical features: ",object_feature)
import math
import seaborn as sns
import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt
rows = math.ceil(len(num_feature)/4)
fig = plt.figure(figsize=(12, 18))
for i,f in enumerate(num_feature):
    fig.add_subplot(rows, 4, i+1)
    sns.boxplot(y=home_data[f])
plt.tight_layout()
plt.show()
# check Skew of target
target = home_data.SalePrice
plt.figure()
sns.distplot(target)
plt.title('Distribution of SalePrice')
plt.show()
import numpy as np
# log transform target to make it Gaussian-alike
log_target = np.log(target + 1)
plt.figure()
sns.distplot(log_target)
plt.title('Distribution of SalePrice')
plt.show()

# data skew check
fig = plt.figure(figsize=(12,18))
for i,f in enumerate(num_feature):
    fig.add_subplot(rows,4,i+1)
    sns.distplot(home_data[f].dropna(),kde= False)
    #plt.xlabel(home_data[f])

plt.tight_layout()
plt.show()
# check the correlation of target and each feature
fig = plt.figure(figsize=(12,20))

for i,f in enumerate(num_feature):
    fig.add_subplot(rows, 4, i+1)
    sns.scatterplot(home_data[f], target)
    
plt.tight_layout()
plt.show()
#Heatmap with annotation of correlation values
home_data = home_data.drop('Id', axis=1).copy()
correlation = home_data.corr()

f, ax = plt.subplots(figsize=(14,12))
plt.title('Correlation of numerical attributes', size=16)
sns.heatmap(correlation)
plt.show()
# log transform
home_data['SalePrice'] = np.log(home_data['SalePrice']+1)
home_data = home_data.rename(columns={'SalePrice': 'SalePrice_log'})
def fill_up_missing(data):
    # fill up the unknown object type as "None"
    for f in data.select_dtypes(include='object').columns:
        data.fillna({f:"None"},inplace=True)
    # fill up the unknown float type as 0
    for f in data.select_dtypes(exclude='object').columns:
        data.fillna({f:0},inplace=True)
    missing_values_count  = data.isnull().sum()
    # check the columns which have missing values
    print("missing_values_count: ",missing_values_count[missing_values_count!=0])
    return data
correlation['SalePrice_log'].sort_values(ascending=False).tail(10)
# Feature selection
# GrLivArea and TotRmsAbvGrd 82% correlation, drop column TotRmsAbvGrd
# GarageYrBlt and YearBuilt 82% correlation, drop column GarageYrBlt
# TotalBsmtSF and 1stFlrSF 82% correlation, drop column TotalBsmtSF
# GarageArea and GarageCars 82% correlation, drop column GarageArea
from datetime import date
def feature_selection(data):
    data = data.drop('Id',axis=1)
    #drop_feature = ['TotRmsAbvGrd','GarageYrBlt','TotalBsmtSF','GarageArea','GarageCars','MiscVal', 'MSSubClass', 'MoSold', 'YrSold','BsmtHalfBath','OverallCond','EnclosedPorch','KitchenAbvGr','LowQualFinSF']
    drop_feature = ['MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd']
    data = data.drop(drop_feature, 1)
    
    data['YearRemodAdd'] = date.today().year -data['YearRemodAdd']
    data['YearBuilt']= date.today().year - data['YearBuilt']
    return data
home_data_copy = feature_selection(home_data)
test_data_copy = feature_selection(test_data)
home_data_copy = fill_up_missing(home_data_copy)
test_data_copy = fill_up_missing(test_data_copy)
# Remove outliers based on observations on scatter plots against SalePrice:
home_data_copy = home_data_copy.drop(home_data_copy['LotFrontage']
                                     [home_data_copy['LotFrontage']>200].index)
home_data_copy = home_data_copy.drop(home_data_copy['LotArea']
                                     [home_data_copy['LotArea']>100000].index)
home_data_copy = home_data_copy.drop(home_data_copy['BsmtFinSF1']
                                     [home_data_copy['BsmtFinSF1']>4000].index)
home_data_copy = home_data_copy.drop(home_data_copy['TotalBsmtSF']
                                     [home_data_copy['TotalBsmtSF']>6000].index)
home_data_copy = home_data_copy.drop(home_data_copy['1stFlrSF']
                                     [home_data_copy['1stFlrSF']>4000].index)
home_data_copy = home_data_copy.drop(home_data_copy.GrLivArea
                                     [(home_data_copy['GrLivArea']>4000) & 
                                      (target<300000)].index)
home_data_copy = home_data_copy.drop(home_data_copy.LowQualFinSF
                                     [home_data_copy['LowQualFinSF']>550].index)

# predited values are log(SalePrice)
def inv_y(transformed_y):
    return np.exp(transformed_y)-1
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer

# try "yeo-johnson",'box-cox'
numeric_transformer = Pipeline(steps=[
   ('imputer', SimpleImputer()),
   ('scaler', StandardScaler()),
   ('power',PowerTransformer(method='yeo-johnson'))])

# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numeric_features = home_data_copy.select_dtypes(exclude='object').drop('SalePrice_log',axis=1).columns
categorical_features = home_data_copy.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

X = home_data_copy.drop('SalePrice_log', axis=1)
#X = pd.get_dummies(X)

# Create target object and call it y
y = home_data_copy.SalePrice_log

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Select ML model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
regression_model_name =['random forest','XGBoost','Linear Regression','Lasso','Ridge','ElasticNet',
                        'KNN Regression','Gradient Boosting Regression','Ada Boost Regression','Support Vector Regression']
regression_models =[RandomForestRegressor(),XGBRegressor(n_estimators=1000, learning_rate=0.05),LinearRegression(),Lasso(alpha=0.0005, random_state=5),Ridge(alpha=0.002, random_state=5),
                  ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7),KNeighborsRegressor(),GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, 
                                      max_depth=4, random_state=5),AdaBoostRegressor(n_estimators=300, learning_rate=0.05, random_state=5),SVR(kernel='linear')]

n_folds = 10

for i in range(len(regression_model_name)):
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regression', regression_models[i])])
    pipe.fit(train_X,train_y)
    val_predictions = pipe.predict(val_X)
    val_mae = mean_absolute_error(inv_y(val_predictions), inv_y(val_y))
    print("Validation MAE for {:}: {:,.0f}".format(regression_model_name[i],val_mae))
# grid search for hyperparameter tuning
# lasso = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('regression', Lasso(random_state=5))])
# param_grid = [{'regression__alpha': np.arange(0.005,0.02,0.003)}]
# CV = GridSearchCV(lasso, param_grid, n_jobs= 1,scoring='neg_mean_squared_error',cv=5)
# CV.fit(train_X, train_y)  
# print(CV.best_params_)    
lasso = Pipeline(steps=[('preprocessor', preprocessor),
                      ('regression', Lasso(alpha=0.0001,random_state=5))])
lasso.fit(train_X,train_y)
val_predictions = lasso.predict(val_X)
val_mae = mean_absolute_error(inv_y(val_predictions), inv_y(val_y))
print("Validation MAE for lasso: {:,.0f}".format(val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data
# fit rf_model_on_full_data on all data from the training data
lasso.fit(X,y)
# make predictions which we will submit. 
test_preds = lasso.predict(test_data_copy)
# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': inv_y(test_preds)})
output.to_csv('submission.csv', index=False)