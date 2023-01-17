%matplotlib inline



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from numpy import random

from time import time

import calendar

from datetime import datetime
from sklearn.metrics import r2_score

from sklearn import metrics

from sklearn.metrics import mean_squared_error



# For splitting dataset

from sklearn.cross_validation import ShuffleSplit, train_test_split



import sklearn.learning_curve as curves

from sklearn.learning_curve import validation_curve



# k-fold cross validation

from sklearn.cross_validation import KFold, cross_val_score



# Import sklearn models

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression
enet = ElasticNet(alpha=0.1, l1_ratio=0.7)

lasso = Lasso(alpha=0.1)

reg = Ridge(alpha = .5)

svr = SVR()

tree = DecisionTreeRegressor()

Forest = RandomForestRegressor(random_state = 0, max_depth = 20, n_estimators = 150)

gbr = GradientBoostingRegressor()

lr = LinearRegression()
def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

train_data.head()
test_data.head()
train_data["year"] = train_data.datetime.apply(lambda x : x.split()[0].split("-")[0])

train_data["month"] = train_data.datetime.apply(lambda x : x.split()[0].split("-")[1])

train_data["day"] = train_data.datetime.apply(lambda x : x.split()[0].split("-")[2])

train_data["hour"] = train_data.datetime.apply(lambda x : x.split()[1].split(":")[0])

train_data = train_data.drop('datetime', axis=1)
test_data.head()
test_data["year"] = test_data.datetime.apply(lambda x : x.split()[0].split("-")[0])

test_data["month"] = test_data.datetime.apply(lambda x : x.split()[0].split("-")[1])

test_data["day"] = test_data.datetime.apply(lambda x : x.split()[0].split("-")[2])

test_data["hour"] = test_data.datetime.apply(lambda x : x.split()[1].split(":")[0])

test_features = test_data.drop('datetime', axis=1)
test_data.head()
df = pd.DataFrame()

df['c'] = train_data['count']

df['hour'] = train_data['hour']

df.shape
sns.lmplot('c', 'hour', data=df, fit_reg=False)
sns.kdeplot(df.hour)
sns.distplot(df.c)
target = train_data['count']

features = train_data.drop(['casual','registered','count'], axis=1)
features.head()
from sklearn.preprocessing import MinMaxScaler

scaled_features = MinMaxScaler().fit_transform(features)

scaled_test_features = MinMaxScaler().fit_transform(test_features)
test_features.head()
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size= 0.3,random_state=42)
def train_and_predict_model(model, model_name, X_train, X_test, y_train, y_test, selected_cols):

    t0 = time()

    model.fit(X_train, y_train)

    train_time = time() - t0

    

    t1 = time()

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    predict_time = time() - t1

    

    train_score = r2_score(y_train, train_pred)

    

    test_score = r2_score(y_test, test_pred)

    

    root_mean_squared_log_error = rmsle(y_test, test_pred)

    

    print ("r2_score of training set of {} is {}".format(model_name, train_score))

    print ("r2_score of testing set of {} is {}".format(model_name, test_score))

    print ("Root mean squared log error of {} is {}".format(model_name, root_mean_squared_log_error))

    print ("cross_val_score of {} is {}".format(model_name, cross_val_score(model, features, target , cv = 10).mean()))

    

    print ("Time taken to train {} is {}".format(model_name, train_time))

    print ("Time taken to predict {} is {}".format(model_name, predict_time))

    return model
total_features = list(features.columns)

print (total_features)

#selected_features = ['hour', 'temp', 'workingday']

model = train_and_predict_model(Forest, 'Forest', X_train, X_test, y_train, y_test, total_features)
from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca_features = pca.fit_transform(features)
len(total_features)
pred = model.predict(scaled_test_features)
test_data['count'] = pred

test_data.head()
df = test_data[['datetime', 'count']]

df.to_csv('submission.csv', index = False)

df.head()
def draw_feature_imp(model, features):

    from matplotlib import pyplot

    importance = model.feature_importances_

    names = list(features)

    pyplot.bar(range(len(importance)), importance)

    print (sorted(zip(map(lambda x: round(x, 4), importance), names), 

                 reverse=True))

    pyplot.show()
draw_feature_imp(Forest, features)
selected_features = ['hour', 'year', 'workingday', 'temp', 'atemp', 'atemp']