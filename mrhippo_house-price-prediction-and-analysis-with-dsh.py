!pip install datasciencehelper # pip install Data Science Helper
# data science and visualization
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew 
import DataScienceHelper as dsh 

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dsh.what_is_DSH()
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.describe()
train.info()
f,ax = plt.subplots(figsize=(16, 12))
sns.heatmap(train.corr())
plt.xticks(rotation = 65)
plt.title("Correlations", fontsize = 15)
plt.show()
fig = plt.figure(figsize = (12,6))
sns.distplot(train["SalePrice"])
plt.title("SalePrice kdeplot")
plt.show()
train, dropped_columns = dsh.nan_value_vis_and_dropping(train, train.keys(), 40)
train = train.drop(["Id"],axis = 1) # drop irrelevant feature
train.info()
train = dsh.fill_nan_categorical(train, list(train.select_dtypes(include='object').keys()))
train = dsh.fill_nan_numeric(train, list(train._get_numeric_data().keys()))
numeric_columns = list(train._get_numeric_data().keys())
dsh.show_kdeplot(train,numeric_columns)
numeric_columns = list(train._get_numeric_data().keys())
dsh.show_boxplot(train,numeric_columns)
train.info()
train.isnull().sum().sum()
numeric_columns = list(train._get_numeric_data().keys())
x_train, scores = dsh.outlier_detector(train, numeric_columns, "WoodDeckSF", "BsmtFinSF1", -2) 
x_train.info()
y_train = x_train["SalePrice"]
x_train = x_train.drop(["SalePrice"],axis = 1)
test.head()
test_ID = test['Id']
test = test.drop(dropped_columns,axis = 1) # dropped columns that are taken from "nan_value_vis_and_dropping" function of "Data Science Helper"
test = test.drop(["Id"],axis = 1)
test.info()
test = dsh.fill_nan_categorical(test,list(test.select_dtypes(include='object').keys()))       
test = dsh.fill_nan_numeric(test, list(test._get_numeric_data().keys()))
test.info()
test.isnull().sum().sum()
ntrain = train.shape[0]
x_ntrain = x_train.shape[0]
all_data = pd.concat((x_train, test)).reset_index(drop=True)
#all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.head()
all_data.isnull().sum().sum()
all_data.info()
x_train.info()
numeric_columns = list(all_data._get_numeric_data().keys())
all_data = dsh.boxcox_skewed_data(all_data,numeric_columns)
all_data = pd.get_dummies(all_data)
x_train = all_data[:x_ntrain]
test = all_data[x_ntrain:]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(x_train,y_train,test_size = 0.2 , random_state = 42) #x,y

parameter_values = [100,120,140,160,180,200,220,240]
test_scores = []
train_scores = []
for i in parameter_values:
    rf = RandomForestRegressor(n_estimators = i, random_state = 42) 
    rf.fit(x_train_val,y_train_val)
    test_scores.append(rf.score(x_test_val,y_test_val))
    train_scores.append(rf.score(x_train_val,y_train_val))

best_parameter_value = dsh.show_sklearn_model_results(test_scores, train_scores, parameter_values, "N Estimators")
rf_best = RandomForestRegressor(n_estimators = best_parameter_value, random_state = 42)
rf_best = rf_best.fit(x_train_val,y_train_val)
predictions = rf_best.predict(test)
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = predictions
sub.to_csv('submission.csv',index=False)