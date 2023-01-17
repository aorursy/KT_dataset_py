# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from numpy import arange

from matplotlib import pyplot

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

raw_data = pd.read_csv("../input/boston-house-prices/housing.csv", delim_whitespace=True, names=names)
#Checking Missing Values

raw_data.isnull().sum()
print(raw_data.shape)
#To know the data type of each columns

raw_data.dtypes
# correlation

set_option('precision', 2)

raw_data.corr(method='pearson')
raw_data.hist(bins=10,figsize=(9,7),grid=False);
prices = raw_data['MEDV']

raw_data = raw_data.drop(['CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'], axis = 1)

features = raw_data.drop('MEDV', axis = 1)

raw_data.head()
#histogram

sns.distplot(raw_data['MEDV']);
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))

# i: index

for i, col in enumerate(features.columns):

    # 3 plots here hence 1, 3

    plt.subplot(1, 6, i+1)

    x = raw_data[col]

    y = prices

    plt.plot(x, y, 'o')

    # Create regression line

    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('prices')
# box and whisker plots

raw_data.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)

pyplot.show()
corr=raw_data.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
from scipy import stats

#histogram and normal probability plot

sns.distplot(raw_data['MEDV'], hist=True);

fig = plt.figure()

res = stats.probplot(raw_data['MEDV'], plot=plt)
# Split-out validation dataset

array = raw_data.values

X = array[:,0:6]

Y = array[:,6]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric using Root Mean Square error method

num_folds = 10

seed = 7

RMS = 'neg_mean_squared_error'
# Spot Check Algorithms

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=RMS)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))

pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))

results = []

names = []

for name, model in pipelines:

	kfold = KFold(n_splits=num_folds, random_state=seed)

	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=RMS)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
fig = pyplot.figure()

fig.suptitle('Scaled Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])

param_grid = dict(n_neighbors=k_values)

model = KNeighborsRegressor()

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold)

grid_result = grid.fit(rescaledX, Y_train)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))

raw_data.shape
ensembles = []

ensembles.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))

ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))

ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))

ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))

results = []

names = []

for name, model in ensembles:

	kfold = KFold(n_splits=num_folds, random_state=seed)

	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=RMS)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

raw_data.shape
fig = pyplot.figure()

fig.suptitle('Scaled Ensemble Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))

model = GradientBoostingRegressor(random_state=seed)

kfold = KFold(n_splits=num_folds, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=RMS, cv=kfold)

grid_result = grid.fit(rescaledX, Y_train)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# prepare the model

scaler = StandardScaler().fit(X_train)

rescaledX = scaler.transform(X_train)

model = GradientBoostingRegressor(random_state=seed, n_estimators=400)

model.fit(rescaledX, Y_train)

# transform the validation dataset

rescaledValidationX = scaler.transform(X_validation)

predictions = model.predict(rescaledValidationX)

print(mean_squared_error(Y_validation, predictions))
predictions=predictions.astype(int)

submission = pd.DataFrame({

        "Org House Price": Y_validation,

        "Pred House Price": predictions

    })



submission.to_csv("PredictedPrice.csv", index=False)