import numpy as np

import pandas as pd

import os

import sys

import tqdm

from multiprocessing import  Pool

import warnings

warnings.filterwarnings("ignore")

from math import sqrt

train_on_gpu = False



# Visualisation libs

import matplotlib.pyplot as plt

from matplotlib.legend_handler import HandlerLine2D

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.decomposition import PCA

from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor
print('In input directory:')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/train.csv')

test = pd.read_csv('/kaggle/input/test.csv')

sample_submission = pd.read_csv('/kaggle/input/sample_submission.csv')



train.shape, test.shape, sample_submission.shape
def score(y_actual, y_predicted):

    return sqrt(mean_squared_log_error(y_actual, y_predicted))

    

def fillNaNInfinity(df):

    df.replace([np.inf, -np.inf], np.nan)

    df.fillna(0, inplace=True)

    return df



def fillInfinity(df):

    df.replace([np.inf, -np.inf], np.nan)

    return df
train.corr().style.background_gradient(cmap='coolwarm')
# From https://www.kaggle.com/miguelangelnieto/pca-and-regression#Simple-Neural-Network, loved it

nans = pd.isnull(train).sum()

nans[ nans > 0 ]
columns_to_remove = nans[ nans > 500 ].reset_index()['index'].tolist()

columns_to_remove
train.dtypes.value_counts()
data = pd.concat([

    train.loc[:, train.columns != 'SalePrice'], test

])



target = np.log(train['SalePrice'] + 1)



data = fillInfinity(data)



data.shape, target.shape
data.drop(labels=columns_to_remove, axis=1, inplace=True)

data.shape
data = pd.get_dummies(data)

data.shape
imp = Imputer(missing_values='NaN', strategy='most_frequent', copy=False)



imp.fit_transform(data)

data.shape
scaler = StandardScaler()



data = scaler.fit_transform(data)



data.shape
data
data[np.isnan(data)] = 0
# pca = PCA(n_components=250)

# pca.fit(data)

# data = pca.transform(data)

# data.shape
# print('Total variance', pca.explained_variance_ratio_.sum())

# pca.explained_variance_ratio_

X = data[:1460]

y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



n_estimators = [5, 10, 20, 25, 50, 100, 200]

train_pred_mse = []

test_pred_mse = []



for n_estimator in n_estimators:

    model = RandomForestRegressor(

        n_estimators = n_estimator,

        min_samples_leaf = 1,

        max_depth = 8

    )

    model.fit(X_train, y_train)



    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)



    train_pred_mse.append(score(y_train, y_train_pred))

    test_pred_mse.append(score(y_test, y_test_pred))

        

fig = plt.figure(figsize=[12,6])



line1, = plt.plot(n_estimators, train_pred_mse, 'b', label="Train MSE")

line2, = plt.plot(n_estimators, test_pred_mse, 'r', label="Test MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('MSE score')

plt.xlabel('n_estimators')

plt.show()
max_depths = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]



train_pred_mse = []

test_pred_mse = []



for max_depth in max_depths:

    model = RandomForestRegressor(

        n_estimators = 25, # because showing best performace in above plot

        min_samples_leaf = 1,

        max_depth = max_depth

    )

    model.fit(X_train, y_train)



    y_train_pred = model.predict(X_train)

    y_test_pred = model.predict(X_test)



    train_pred_mse.append(score(y_train, y_train_pred))

    test_pred_mse.append(score(y_test, y_test_pred))



fig = plt.figure(figsize=[12,6])



line1, = plt.plot(max_depths, train_pred_mse, 'b', label="Train MSE")

line2, = plt.plot(max_depths, test_pred_mse, 'r', label="Test MSE")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('MSE score')

plt.xlabel('max_depth')

plt.show()
# based on above plot, picking the best params (note, don't just check the best dip, also look for overfitting)

clf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
X = data[:1460]

y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf.fit(X_train, y_train)
score(y_train, clf.predict(X_train)), score(y_test, clf.predict(X_test))
xgbr = XGBRegressor(max_depth=5, n_estimators=400)

X = data[:1460]

y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



xgbr.fit(X_train, y_train)
score(np.exp(y_train) - 1, np.exp(xgbr.predict(X_train)) - 1), score(np.exp(y_test) - 1, np.exp(xgbr.predict(X_test)) - 1)
test = data[1460:]

sample_submission['SalePrice'] = xgbr.predict(test)

sample_submission['SalePrice'] = np.exp(sample_submission['SalePrice']) - 1

sample_submission.head()
sample_submission.to_csv('submission.csv', index=False)