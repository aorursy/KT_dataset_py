# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head()
df.info()
df.describe()
sns.pairplot(df.drop(['id'], axis=1))
df.isna().sum()
def preprocess(df):

    df['type'] = np.where(df['type']=='new', 1, 0)

    df['type'].fillna(df['type'].mode()[0], inplace=True)

    df.fillna(df.mean(), inplace=True)

    return(df)
df = preprocess(df)
def plot_norm(y):

    x = y

    x1 = np.log(x)



    fig, ax =plt.subplots(1,2)

    sns.distplot(x, fit=stats.norm, ax=ax[0])

    sns.distplot(x1, fit=stats.norm, ax=ax[1])



    fig, ax = plt.subplots(1, 1)

    stats.probplot(x, plot=plt)

    stats.probplot(x1, plot=plt)

    fig.show()



    res = stats.probplot(x, plot=plt)

    res = stats.probplot(x1, plot=plt)
plot_norm(df['feature3'])
plot_norm(df['feature5'])
plot_norm(df['feature9'])
plot_norm(df['feature11'])
#df['feature2'] = np.log(df['feature2'])

df['feature3'] = np.log(df['feature3'])

#df['feature6'] = np.log(df['feature6'])

df['feature5'] = np.log(df['feature5'])

#df['feature9'] = np.log(df['feature9'])

#df['feature11'] = np.log(df['feature11'])
#correlation matrix

corrmat = df.drop('id', axis=1).corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmin=-0.8, vmax=0.8, square=False, annot=True);
f, ax = plt.subplots(figsize=(12, 9))

sns.boxplot(data=df.drop(['id', 'rating'], axis=1))
from sklearn.ensemble import IsolationForest



X = df.drop(['id', 'rating'], axis=1)

clf = IsolationForest( behaviour = 'new', max_samples=100, random_state = 1, contamination= 'auto')

preds = clf.fit_predict(X)

print("The number of predicted outliers in train dataset: {}".format(list(preds).count(-1)))
df_lessout = pd.concat([df, pd.Series(preds)], axis=1)

df_lessout = df_lessout[df_lessout[0]!=-1]

print(len(df_lessout), len(df))
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt



X = df.drop(['id', 'rating'], axis=1)

y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

clf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)

clf.fit(X_train, y_train)

y_pred = np.round(clf.predict(X_test))
print("R2 of RF without Data explore: {}\n".format(clf.score(X_test, y_test)))

rms = sqrt(mean_squared_error(y_test, y_pred))

print("RMSE of RF without Data explore: {}\n".format(rms))
from sklearn.model_selection import RandomizedSearchCV



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2600, num = 13)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [3, 5, 10]

min_samples_leaf = [2, 4, 8]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
X = df.drop(['id', 'rating'], axis=1)

y = df['rating']



clf = RandomForestRegressor()

clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

clf_random.fit(X, y)
clf_random.best_params_
from sklearn.model_selection import GridSearchCV



param_grid = {

    'max_depth': [10, 20, 30],

    'min_samples_leaf': [1, 2, 3],

    'min_samples_split': [3, 5],

    'n_estimators': [2600, 2700],

    'bootstrap' : [False],

    'max_features': ['sqrt']

}



clf_grid = GridSearchCV(estimator = clf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



clf_grid.fit(X, y)
clf_grid.best_params_
test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

test = preprocess(test)

y_pred_test = np.round(clf_grid.predict(test.drop('id', axis=1)))

final = pd.concat([test.id, pd.Series(y_pred_test)], axis=1)

final.columns = ['id', 'rating']

final.to_csv('subFinal.csv', index=False)