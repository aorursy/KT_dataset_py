# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# data analysis and wrangling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import time

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

!pip install py7zr
import py7zr
import zipfile

archive = py7zr.SevenZipFile('/kaggle/input/mercari-price-suggestion-challenge/train.tsv.7z', mode='r')
archive.extractall(path='/kaggle/temp/')
archive.close()

with zipfile.ZipFile('/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/temp/')
for dirname, _, filenames in os.walk('/kaggle/temp'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/temp/train.tsv', sep='\t')
train.info()
A = train[['name', 'price']].groupby(['name'], as_index=False).mean().sort_values(by='price', ascending=False)
B = train[['item_condition_id', 'price']].groupby(['item_condition_id'], as_index=False).mean().sort_values(by='price', ascending=False)
C = train[['category_name', 'price']].groupby(['category_name'], as_index=False).mean().sort_values(by='price', ascending=False)
D = train[['brand_name', 'price']].groupby(['brand_name'], as_index=False).mean().sort_values(by='price', ascending=False)
E = train[['shipping', 'price']].groupby(['shipping'], as_index=False).mean().sort_values(by='price', ascending=False)
F = train[['category_name', 'price']].groupby(['category_name'], as_index=False).mean().sort_values(by='price', ascending=False)
for i in (A, B, C, D, E, F):
    print(i)
X = train.drop(['train_id', 'price'], axis=1)
y = train['price']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train = X
y_train = y
print(X_train.shape)
print(X_train.columns.values)
print(train.shape)
global brand_uniq
brand_uniq = X_train.loc[X_train.brand_name.notnull(), ['brand_name']]
brand_uniq = brand_uniq['brand_name'].unique()
print(brand_uniq)
brand_null_index = X_train[X_train.brand_name.isnull()].index
print(brand_null_index)
def add_brandname1(name):
    global brand_uniq
    for i in brand_uniq:
        if i in name:
            return(i)
    return('')

def add_brandname2(df, brand_null_index):
    df.loc[brand_null_index, 'brand_name'] = df.name[brand_null_index].apply(add_brandname1)
    return(df)
start = time.time()
X_train = add_brandname2(X_train, brand_null_index)
X_train.brand_name
print("brand replace time :", time.time() - start)
plt.hist(y_train)
y_train = np.log1p(y_train)
plt.hist(y_train)
print(sum(X_train.name.isnull())) #0
print(sum(X_train.item_condition_id.isnull())) #0
print(sum(X_train.category_name.isnull())) #5718
print(sum(X_train.brand_name.isnull())) #0
print(sum(X_train.shipping.isnull())) #0
print(sum(X_train.item_description.isnull())) #3
X_train.category_name = X_train.category_name.fillna('Others')
print(sum(X_train.category_name.isnull()))
object_cate = X_train.category_name.value_counts()[X_train.category_name.value_counts() > 5000].index.values
X_train.category_name[~X_train.category_name.isin(object_cate)] = 'Others'
X_train.brand_name[X_train.brand_name == ''] = 'Others'
object_brand = X_train.brand_name.value_counts()[X_train.brand_name.value_counts() > 5000].index.values
X_train.brand_name[~X_train.brand_name.isin(object_brand)] = 'Others'
feature = X_train.drop(['name', 'item_description'], axis = 1)
print(feature.info())
print(X_train.info())
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

# Converting string labels into numbers.
le1.fit(list(feature.brand_name))
le2.fit(list(feature.category_name))

brand_name=le1.transform(list(feature.brand_name))
category_name=le2.transform(list(feature.category_name))
feature_processed = zip(brand_name, category_name, X_train.item_condition_id, X_train.shipping)
feature_processed = list(feature_processed)
print(feature_processed[:10])
from sklearn.linear_model import BayesianRidge, LinearRegression
#from sklearn.svm import SVR
#training SVR was too slow because of dataset's size
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

#start = time.time()
#clf = BayesianRidge(compute_score=True)
#clf.fit(feature_processed, y_train)
#print("clf time :", time.time() - start)

#start = time.time()
#ols = LinearRegression()
#ols.fit(feature_processed, y_train)
#print("ols time :", time.time() - start)
start = time.time()
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(feature_processed, y_train)
print("rfr time :", time.time() - start)
#start = time.time()
#mlp = MLPRegressor(random_state=1, max_iter=500)
#mlp.fit(feature_processed, y_train)
#print("mlp time :", time.time() - start)
#start = time.time()
#reg = GradientBoostingRegressor(random_state=0)
#reg.fit(feature_processed, y_train)
#print("reg time :", time.time() - start)
"""
MSE_clf = mean_squared_error(np.log1p(y_test), predicted_clf) #1.630
print(MSE_clf**0.5)
MSE_ols = mean_squared_error(np.log1p(y_test), predicted_ols) #1.630
print(MSE_ols**0.5)
MSE_rfr = mean_squared_error(np.log1p(y_test), predicted_rfr) #1.646 n = 10
print(MSE_rfr**0.5)
MSE_mlp = mean_squared_error(np.log1p(y_test), predicted_mlp) #1.510 (iter = 5), 1.576 (iter = 10)
print(MSE_mlp**0.5)
"""
#Same processing to test data
test = pd.read_csv('/kaggle/temp/test_stg2.tsv', sep='\t')
test.info()
X_test = test.drop(['test_id'], axis=1)

start = time.time()
brand_null_index = X_test[X_test.brand_name.isnull()].index
X_test = add_brandname2(X_test, brand_null_index)
print("brand replace time :", time.time() - start)
X_test.category_name = X_test.category_name.fillna('Others')
print(sum(X_test.category_name.isnull()))
X_test.category_name[~X_test.category_name.isin(object_cate)] = 'Others'
X_test.brand_name[X_test.brand_name == ''] = 'Others'
X_test.brand_name[~X_test.brand_name.isin(object_brand)] = 'Others'
feature_test = X_test.drop(['name', 'item_description'], axis = 1)
print(feature_test.info())
print(X_test.info())
brand_name_test=le1.transform(list(feature_test.brand_name))
category_name_test=le2.transform(list(feature_test.category_name))
feature_processed_test = zip(brand_name_test, category_name_test, X_test.item_condition_id, X_test.shipping)
feature_processed_test = list(feature_processed_test)
print(feature_processed_test[:10])
#Predict Output (test_data)
#predicted_clf = clf.predict(feature_processed_test)
#predicted_ols = ols.predict(feature_processed_test) 
predicted_rfr = rfr.predict(feature_processed_test)
#predicted_mlp = mlp.predict(feature_processed_test)
#predicted_reg = reg.predict(feature_processed_test)
Y_pred_rfr = np.expm1(predicted_rfr)
#Y_pred_mlp = np.expm1(predicted_mlp)
#Y_pred_reg = np.expm1(predicted_reg)
#Y_pred_rfr_mlp = (Y_pred_rfr + Y_pred_mlp)/2
submission_rfr = pd.DataFrame({
        "test_id": test["test_id"],
        "price": Y_pred_rfr
    })

#submission_mlp = pd.DataFrame({
#        "test_id": test["test_id"],
#        "price": Y_pred_mlp
#    })

#submission_rfr_mlp = pd.DataFrame({
#        "test_id": test["test_id"],
#        "price": Y_pred_rfr_mlp
#    })

#submission_reg = pd.DataFrame({
#        "test_id": test["test_id"],
#        "price": Y_pred_reg
#    })
#print(submission_rfr_mlp.head())
#print(submission_reg.head())
#submission_rfr.to_csv('submission.csv', index=False)
#submission_mlp.to_csv('submission.csv', index=False)
submission_rfr.to_csv('submission.csv', index=False)