# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import sklearn as skl
%matplotlib inline

import seaborn as sns #не особенно нужна, я просто хотела красивый график
sns.set(style="whitegrid")


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/hsemath2020flights/flights_train.csv")
df_train.head()
df_train.isnull().values.any()
type(df_train['DATE'][1])
df_train["MONTH"]=df_train["DATE"].apply(lambda x: int(x.split('-')[1]))
df_train["dep_delayed_15min_int"] = df_train["dep_delayed_15min"].astype(int)
df_train_changed=df_train[["MONTH", "DEPARTURE_TIME", "DISTANCE", "dep_delayed_15min_int"]]
df_train_changed.head()
df_train_changed.describe()
df_train_changed.plot(y='MONTH', kind='hist', bins=12, legend=False, title = 'MONTH')
df_train_changed.plot(y='DEPARTURE_TIME', kind='hist', bins=100, legend=False, title = 'DEPARTURE_TIME')
df_train_changed.plot(y='DISTANCE', kind='hist', bins=100, legend=False, title = 'DISTANCE')
df_train_changed.plot(y='dep_delayed_15min_int', kind='hist', legend=False, title = 'dep_delayed_15min')
plt.show()
ax = sns.barplot(x="MONTH", y="dep_delayed_15min_int", data=df_train_changed)
ax = sns.lineplot(x="DEPARTURE_TIME", y="dep_delayed_15min_int", data=df_train_changed)
ax = sns.lineplot(x="DISTANCE", y="dep_delayed_15min_int", data=df_train_changed)
df_n=df_train_changed.to_numpy()
X,y=df_n[:,:-1], df_n[:,-1]
df_test = pd.read_csv("/kaggle/input/hsemath2020flights/flights_test.csv")
df_test.head()
df_test.isnull().values.any()
type(df_test['DATE'][1])
df_test["MONTH"]=df_test["DATE"].apply(lambda x: int(x.split('-')[1]))
df_test_changed=df_test[["MONTH", "DEPARTURE_TIME", "DISTANCE"]]
df_test_changed.head()
X_test=df_test_changed.to_numpy()
print('Test dataframe description:')
df_test_changed.describe()
print('Train dataframe description:')
df_train_changed.describe()
print('Ниже приведены гистограммы частоты распределения значений по параметрам MONTH, DEPRTURE TIME и DISTANCE. Голубым цветом нарисованы распределения частот для тренировочной выборки, содержащий 200 000 значений. Зеленым цветом нарисованы распределения частот для тестовой выборки, содержащей 100 000 значений')

ax = df_train_changed.plot(y='MONTH', kind='hist', bins=12, legend=False, title = 'MONTH')
df_test_changed.plot(y='MONTH', kind='hist', ax=ax, color='green', bins=12, legend=False)
plt.show()
ax = df_train_changed.plot(y='DEPARTURE_TIME', kind='hist', bins=100, legend=False, title = 'DEPARTURE_TIME')
df_test_changed.plot(y='DEPARTURE_TIME', kind='hist', ax=ax, legend=False, color='green', bins=100)
plt.show()
ax = df_train_changed.plot(y='DISTANCE', kind='hist', bins=100, legend=False, title = 'DISTANCE')
df_test_changed.plot(y='DISTANCE', kind='hist', ax=ax, legend=False, color='green', bins=100)
plt.show()
import sklearn.model_selection as model_selection
import sklearn.linear_model as linear_model
logreg_model = linear_model.LogisticRegression()
parameters = {'C': np.linspace(0.001, 1, 10), 'penalty': ['l2'], 'solver' : ['lbfgs']}
grid_search = model_selection.GridSearchCV(logreg_model, parameters)
grid_search.fit(X, y)
print(grid_search.best_params_)
parameters = {'C': np.linspace(0.001, 1.01, 10), 'penalty': ['l2'], 'solver': ['lbfgs']}
random_search = model_selection.RandomizedSearchCV(logreg_model, parameters)
random_search.fit(X, y)
print(random_search.best_params_)
logreg_model = linear_model.LogisticRegression(C=grid_search.best_params_['C'], penalty=grid_search.best_params_['penalty'])
score_logreg_kfold = model_selection.cross_val_score(logreg_model, X, y, cv=10, scoring='roc_auc')
print(score_logreg_kfold) 
logreg_model.fit(X, y)
logreg_prediction = logreg_model.predict_proba(X_test)
submission = pd.read_csv('/kaggle/input/hsemath2020flights/flights_sample_submission.csv', index_col='id')
submission['dep_delayed_15min'] = logreg_prediction[:, 1]
submission.to_csv('flights_logreg_prediction.csv')
from sklearn import ensemble as ensemble 
from sklearn import model_selection as model_selection
random_forest_model=ensemble.RandomForestRegressor()
parameter = {'n_estimators': range(90, 100)}
grid_search = model_selection.GridSearchCV(random_forest_model, parameter)
grid_search.fit(X, y)
print(grid_search.best_params_)
parameter = {'n_estimators': range(90, 100)}
random_search = model_selection.RandomizedSearchCV(random_forest_model, parameter)
random_search.fit(X, y)
print(random_search.best_params_)
random_forest_model = ensemble.RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'])
score_random_forest = model_selection.cross_val_score(random_forest_model, X, y, cv=4, scoring='roc_auc')
print(score_random_forest)
random_forest_model.fit(X, y)
random_forest_prediction = random_forest_model.predict(X_test)
submission = pd.read_csv('/kaggle/input/hsemath2020flights/flights_sample_submission.csv', index_col='id')
submission['dep_delayed_15min'] = random_forest_prediction
submission.to_csv('flights_random_forest_prediction.csv')