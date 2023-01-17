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
#load data in

data = pd.read_csv("../input/housing.csv")

#fill missing values

median_num_bedrooms = data["total_bedrooms"].median()

data["total_bedrooms"].fillna(median_num_bedrooms, inplace=True)

data.head()
data.info()
data["ocean_proximity"].value_counts()
#turn categorical values into numeric ones

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer



df = data.drop('median_house_value', axis=1)

num_attrs = list(df)

num_attrs.remove("ocean_proximity")

cat_attrs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([

    ("num", SimpleImputer(strategy='median'),num_attrs),

    ("cat", OneHotEncoder(), cat_attrs),

])

X = full_pipeline.fit_transform(df)

X = pd.DataFrame(X, columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 

                               'population', 'households', 'median_income', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'])

X.head()



#train test split

from sklearn.model_selection import train_test_split



y = data.iloc[:, 9:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1105233)

print(X_train.shape, y_train.shape)

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

#define regressor function

def run_reg(regressor, X_train, X_test, y_train, y_test):

    #train classifier using train data

    regressor.fit(X_train, y_train)

    pred = regressor.predict(X_test)

    for i in range(pred.size):

        if(pred[i]<15000):

            pred[i] = 15000

        elif(pred[i]>500000):

            pred[i] = 500000

    MAE = mean_absolute_error(y_test, pred)

    plt.scatter(pred, y_test, alpha=0.5)

    plt.title('MAE = ' + str(MAE))

    plt.xlabel("Ground truth")

    plt.ylabel("Prediction")

    plt.show()

    #plot coefficient of the model

    coef = regressor.coef_

    plt.scatter(X.columns, coef, alpha=0.8)

    plt.xticks(rotation=90)

    plt.ylabel("Coefficient")

    plt.xlabel("Feature")

    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.4)

    plt.show()

    return MAE   

from sklearn.linear_model import Ridge

ridge_reg1 = Ridge(alpha=0.1)

ridge_mae1 = run_reg(ridge_reg1, X_train, X_test, y_train, y_test)

ridge_reg2 = Ridge(alpha=0.001)

ridge_mae2 = run_reg(ridge_reg2, X_train, X_test, y_train, y_test)
from sklearn.linear_model import Lasso

lasso_reg1 = Lasso(alpha=0.1)

lasso_mae1 = run_reg(lasso_reg1, X_train, X_test, y_train, y_test)

lasso_reg2 = Lasso(alpha=0.001)

lasso_mae2 = run_reg(lasso_reg2, X_train, X_test, y_train, y_test)
from sklearn.linear_model import ElasticNet

elastic_net1 = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic_net_mae1 = run_reg(elastic_net1, X_train, X_test, y_train, y_test)

elastic_net2 = ElasticNet(alpha=0.001, l1_ratio=0.5)

elastic_net_mae2 = run_reg(elastic_net2, X_train, X_test, y_train, y_test)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(3)

X_train_poly3 = poly.fit_transform(X_train)

X_test_poly3 = poly.fit_transform(X_test)

X = pd.DataFrame(X_train_poly3)

ridge_reg1 = Ridge(alpha=0.1)

ridge_mae1 = run_reg(ridge_reg1, X_train_poly3, X_test_poly3, y_train, y_test)

ridge_reg2 = Ridge(alpha=0.001)

ridge_mae2 = run_reg(ridge_reg2, X_train_poly3, X_test_poly3, y_train, y_test)
lasso_reg1 = Lasso(alpha=0.1)

lasso_mae1 = run_reg(lasso_reg1, X_train_poly3, X_test_poly3, y_train, y_test)

lasso_reg2 = Lasso(alpha=0.001)

lasso_mae2 = run_reg(lasso_reg2, X_train_poly3, X_test_poly3, y_train, y_test)
elastic_net1 = ElasticNet(alpha=0.1, l1_ratio=0.5)

elastic_net_mae1 = run_reg(elastic_net1, X_train_poly3, X_test_poly3, y_train, y_test)

elastic_net2 = ElasticNet(alpha=0.001, l1_ratio=0.5)

elastic_net_mae2 = run_reg(elastic_net2, X_train_poly3, X_test_poly3, y_train, y_test)