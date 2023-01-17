%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
t_df = pd.read_csv('../input/train.csv')
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

t_df['Sex'] = t_df['Sex'].apply(lambda x: {'male': 1, 'female': 0}.get(x, x))

s_df = t_df.loc[t_df['Survived'] == 1]
d_df = t_df.loc[t_df['Survived'] == 0]
#Cabin Embarked
s_df.groupby('Embarked').plot()
s_df.describe()
mean_age = t_df[t_df['Age'].notnull()]['Age'].mean()
t_df.loc[t_df['Age'].isnull(), 'Age'] = mean_age
inputs = t_df.reindex(columns=t_df.columns[2:]).values
results = t_df['Survived'].values
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()
param_grid ={
        'loss':['hinge'],
                 'fit_intercept': [True, False],
        'penalty': ['none'],
        'n_iter': [50],
        'warm_start': [True, False],
        'average': [True, False]
  }



sgd_gs = GridSearchCV(sgd_clf,param_grid=param_grid,cv=20)
sgd_gs.fit(inputs, results)

print('sgd best params: {}\nbest score: {}'.format(sgd_gs.best_params_,sgd_gs.best_score_))
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
param_grid = {
    'n_estimators': [10, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': list(range(2, 8)),
    'warm_start': [True, False]
}

rf_gs = GridSearchCV(rf_clf, param_grid=param_grid, cv=20)
rf_gs.fit(inputs, results)

print('rf best params: {}\nbest score: {}'.format(rf_gs.best_params_,rf_gs.best_score_))
t_df