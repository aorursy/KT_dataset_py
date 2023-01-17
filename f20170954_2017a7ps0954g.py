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
data = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

data.head()
#like one hot encoding but not the same

data['type'] = data['type'].replace("new",1)

data['type'] = data['type'].replace("old",0)

data.fillna(data.mean(), inplace=True)
data.info()
import pandas as pd

import numpy as np

import seaborn as sns



X = data.iloc[:,0:14]  #independent columns

y = data.iloc[:,-1]    #target column i.e price range

#get correlations of each features in dataset

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler



drop_cols = ['id', 'rating']

X = RobustScaler().fit_transform(data.drop(drop_cols, axis=1))

y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

clasf = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=100)

clasf.fit(X_train, y_train)

y_pred = np.round(clasf.predict(X_test))
from sklearn.metrics import mean_squared_error

from math import sqrt



print("Calculating R2 {}\n".format(clasf.score(X_test, y_test)))

rms = sqrt(mean_squared_error(y_test, y_pred))

print("Calculating RMSE {}\n".format(rms))
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 150, stop = 3000, num = 20)]

max_features = ['auto', 'sqrt', 'log2']

max_depth = [int(x) for x in np.linspace(10, 150, num = 15)]

max_depth.append(None)

min_samples_split = [3, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

paramd = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
X = data.drop(['id', 'rating'], axis=1)

y = data['rating']



clasf = RandomForestRegressor()

clasf_random = RandomizedSearchCV(estimator = clasf, param_distributions = paramd, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

clasf_random.fit(X, y)

clasf_random.best_params_
from sklearn.model_selection import GridSearchCV



param_grid = {

    'max_depth': [30, 40, 50],

    'min_samples_leaf': [1, 2],

    'min_samples_split': [3, 5],

    'n_estimators': [1200, 1300],

    'bootstrap' : [False],

    'max_features': ['log2']

}



clasf_grid = GridSearchCV(estimator = clasf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



clasf_grid.fit(X, y)

clasf_grid.best_params_
test_data= pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

test_data['type'] = test_data['type'].replace("new",1)

test_data['type'] = test_data['type'].replace("old",0)

test_data['type'].fillna(test_data['type'].mode()[0], inplace=True)

test_data.fillna(test_data.mean(), inplace=True)







y_pred = np.round(clasf_grid.predict(test_data.drop('id', axis=1)))

answer = pd.concat([test_data.id, pd.Series(y_pred)], axis=1)

answer.columns = ['id', 'rating']

answer.to_csv('answer.csv', index=False)