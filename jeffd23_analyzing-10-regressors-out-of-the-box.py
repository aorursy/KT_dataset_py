# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv")



# Any results you write to the current directory are saved as output.
target = np.log(train.SalePrice)

train = train.drop(['Id','SalePrice'], axis=1)



catagorical_feats = [i for i in train.columns.values if train[i].dtype == 'object']

numeric_feats = [i for i in train.columns.values if train[i].dtype != 'object']


## One-Hot-Encoding

for cat in catagorical_feats:

    #train[cat] = LabelEncoder().fit_transform(train[cat].values)

    one_hot = pd.get_dummies(train[cat], prefix=cat)

    train = train.drop(cat, axis=1)

    train = pd.concat([train, one_hot], axis=1)

    

for num in numeric_feats:

    train[num] = np.log1p(train[num])



train = train.fillna(-1)



train.tail()



train = train[numeric_feats]
train = train[['1stFlrSF', '2ndFlrSF']]
target = np.log(train.SalePrice)

train = train.drop(['Id','SalePrice'], axis=1)



catagorical_feats = [i for i in train.columns.values if train[i].dtype == 'object']

numeric_feats = [i for i in train.columns.values if train[i].dtype != 'object']
## One-Hot-Encoding

for cat in catagorical_feats:

    one_hot = pd.get_dummies(train[cat], prefix=cat)

    train = train.drop(cat, axis=1)

    train = pd.concat([train, one_hot], axis=1)

    

for num in numeric_feats:

    train[num] = np.log1p(train[num])



## Standard Scaler

train = train.fillna(0.)



train.tail()
X_train, X_test, y_train, y_test = train_test_split(

                                       train, target, test_size=0.33, random_state=23)
from sklearn import linear_model, tree, svm, ensemble 



# List of 10 Regressor Objects

regressors = [

    linear_model.LinearRegression(),

    linear_model.Ridge(),

    linear_model.Lasso(),

    linear_model.ElasticNet(),

    linear_model.BayesianRidge(),

    linear_model.RANSACRegressor(),

    svm.SVR(),

    ensemble.GradientBoostingRegressor(),

    tree.DecisionTreeRegressor(),

    ensemble.RandomForestRegressor()

]



# Logging for Visual Comparison

log_cols=["Regressor", "RMSE Loss"]

log = pd.DataFrame(columns=log_cols)



for reg in regressors:

    reg.fit(X_train, y_train)

    name = reg.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    predictions = reg.predict(X_test)

    rmse = mean_squared_error(y_test, np.exp(predictions))

    print("Root Mean Squared Error: {}".format(rmse))

    

    

    log_entry = pd.DataFrame([[name, rmse]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
X_train, X_test, y_train, y_test = train_test_split(

                                       train, target, test_size=0.33, random_state=23)
from sklearn import linear_model, tree, svm, ensemble 



# List of 10 Regressor Objects

regressors = [

    linear_model.LinearRegression(),

    linear_model.Ridge(),

    linear_model.Lasso(),

    linear_model.ElasticNet(),

    linear_model.BayesianRidge(),

    linear_model.RANSACRegressor(),

    svm.SVR(),

    ensemble.GradientBoostingRegressor(),

    tree.DecisionTreeRegressor(),

    ensemble.RandomForestRegressor()

]



# Logging for Visual Comparison

log_cols=["Regressor", "RMSE Loss"]

log = pd.DataFrame(columns=log_cols)



for reg in regressors:

    reg.fit(X_train, y_train)

    name = reg.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    predictions = reg.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("Root Mean Squared Error: {}".format(rmse))

    

    

    log_entry = pd.DataFrame([[name, rmse]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)