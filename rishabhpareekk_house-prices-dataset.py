# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read train and test test data
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# view data samples for train data
train_data.sample(10)
# Check the information of datasets
print('*' * 20 ,'Traning Data ',"*"*20)
print(train_data.info())
print('*' * 20 ,'Test Data ',"*"*20)
print(test_data.info())
pd.isnull(train_data).sum()
for i in train_data.columns:
    print(i,'-->',train_data[i].isnull().sum())
sns.heatmap(train_data.isnull(),yticklabels=False, cmap='plasma')
# columns which have missing values
train_data.columns[train_data.isnull().any()]
missing_data_list = train_data.columns[train_data.isnull().any()]
list(missing_data_list)
for i in missing_data_list:
    print(i,'--->',round(train_data[i].isna().sum()/len(train_data) *100,2),'%' )
pd.isnull(test_data).sum()
sns.heatmap(test_data.isnull(),yticklabels=False, cmap='plasma')
# columns which contains missing values in test data
test_data.columns[test_data.isnull().any()]
# percentage of of missing values
missing_col = list(test_data.columns[test_data.isnull().any()])
for i in missing_col:
    print(i,'-->',round(test_data[i].isna().sum()/len(train_data) *100,2),'%')
# Descriptive mesure of train_data
train_data.describe(include='all')
train_data.describe()
# for Test_data 
test_data.describe(include ='all')
test_data.describe()
columns = []
for i in train_data.columns:
    if train_data[i].isna().sum()/len(train_data)*100 >=10:
        columns.append(i)
print(columns)# List of columns which has >=10% missing data       
# Droping columns in train data
train_data = train_data.drop(columns=columns,axis = 1)
train_data.sample(5)
train_data.info()
feature_data = train_data.drop(columns=['SalePrice'])
target_data = train_data.SalePrice
feature_data.info()
# select int data in to one variable
float_int_data = feature_data.select_dtypes(include = ['int','float'])
# select object data into one variable
cat_data = feature_data.select_dtypes(include = ['object'])
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
# creating sub pipelines
float_int_pipeline = make_pipeline(SimpleImputer(strategy='median'),MinMaxScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OrdinalEncoder())
from sklearn.compose import make_column_transformer
preprocessor = make_column_transformer(
               (cat_pipeline,cat_data.columns),
               (float_int_pipeline,float_int_data.columns)
               )
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
pipeline = make_pipeline(preprocessor, LinearRegression())
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(feature_data, target_data)
pipeline.fit(feature_data, target_data)
pipeline.score(testX,testY)
pipeline = make_pipeline(preprocessor, RandomForestClassifier())
pipeline.fit(feature_data, target_data)
pipeline.score(testX, testY)
from sklearn.metrics import mean_squared_error
y_pred = pipeline.predict(testX)
mean_squared_error(testY, y_pred)
y_pred = pipeline.predict(test_data)
submission = pd.DataFrame({'Id': test_data.index,'SalePrice': y_pred})
submission.to_csv("house_prices_submission.csv", index=False)
