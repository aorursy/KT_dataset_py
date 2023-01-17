# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
my_data = pd.read_csv("../input/kc_house_data.csv")

my_data.isna().sum()

my_data.info()
data = my_data



print(len(data["id"].unique()))

data.shape
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



cols = ['price', 'bedrooms','bathrooms','sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']



sns.heatmap(data[cols].corr(), annot=True, cmap='RdYlGn')

fig = plt.gcf()

fig.set_size_inches(10, 7)

plt.show()
cate = [ "date", "yr_built","yr_renovated", "zipcode"]

for i in cate:

    ax = sns.boxplot(x=i, y="price", data=data)





    # Change the orientation of X axis Labels

    plt.xticks(rotation=30)



    # setting space between each box plot

    plt.subplots_adjust(hspace=0.8)



    fig = plt.gcf()



    # displaying the final output graph

    plt.show()
data["yr_built"].unique()
from sklearn.preprocessing import LabelBinarizer

cols_x = ['price', 'bedrooms','bathrooms','sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'grade', 'sqft_above', 'sqft_basement', 'sqft_living15']

cols_y = ['price']



data = data.dropna()

data = data.reset_index(drop=True)



x = data[cols_x]

Y = data[cols_y]



lb = LabelBinarizer()

lb_results = lb.fit_transform(data["yr_built"])

lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

x = x.join(lb_results_df)



lb_results = lb.fit_transform(data["zipcode"])

lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

x = x.join(lb_results_df)



# lb_results = lb.fit_transform(data["yr_renovated"])

# lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

# x = x.join(lb_results_df)
x.keys()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

from sklearn.tree import DecisionTreeRegressor 

from sklearn import ensemble

from sklearn import metrics

import xgboost as xgb

from sklearn import linear_model

import statsmodels.api as sm



X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.30, random_state=0)

model = RandomForestRegressor()

# model = linear_model.LinearRegression()

scaler = StandardScaler()

# Fit only to the training data

scaler.fit(X_train.values)



# Now apply the transformations to the data:

X_train = scaler.transform(X_train.values)

X_test = scaler.transform(X_test.values)





params = {

    'max_depth': 6, 

    'n_estimators': 800, 

    'subsample': 0.95, 

    'colsample_bytree': 0.3, 

    'learning_rate': 0.05, 

    'reg_alpha': 0.1

}





model = xgb.XGBRegressor(**params)

model.fit(X_train, y_train)

ypred = model.predict(X_test)

ypred_train = model.predict(X_train)





print("Mean Absoulte error w.r.t pred :", metrics.mean_absolute_error(y_test, ypred))

print("MSE Test                       :", metrics.mean_squared_error(y_test, ypred))

print("MSE Train                      :", metrics.mean_squared_error(y_train, ypred_train))

print("Mean Absoulte error w.r.t train:", metrics.mean_absolute_error(y_train, ypred_train))

print("R square score                 :", metrics.r2_score(y_test, ypred) )
plt.xlabel('data points')

plt.ylabel('price')

plt.plot(y_test.values, 'y', label='actual')

plt.plot(ypred, 'r', label='predicted')

plt.legend()

fig = plt.gcf()

plt.show()