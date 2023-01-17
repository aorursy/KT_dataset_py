# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
FILEPATH = '/kaggle/input/heart-disease-uci/heart.csv'
df = pd.read_csv(FILEPATH)
df.sample(2)
X = df.drop(['target'], axis = 1)

y = df['target']
from sklearn.svm import SVC
results = pd.DataFrame(columns = ['train_score', 'test_score'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 23)



model = SVC()

model.fit(X_train, y_train)



score_train = model.score(X_train, y_train)

score_test = model.score(X_test, y_test)



#print('training score : ', score_train)

#print('testing score : ', score_test)



col_name = 'SVC - '



results.loc[col_name] = (score_train, score_test)
results
# Param 1

param_grid = {

    'C' : [1, 10],

    'gamma' : [0.1, 0.01],

    'kernel' : ['rbf', 'linear']

}



# Param 2

from sklearn.utils.fixes import loguniform



# param_grid = {

#     'C': loguniform(1e0, 1e3),

#     'gamma': loguniform(1e-4, 1e-3),

#     'kernel': ['rbf']

# }
param_grid
grid_model = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv = 5, return_train_score = False)



grid_model.fit(X_train, y_train)



best_params = grid_model.best_params_

best_estimator = grid_model.best_estimator_



print('\nbest_params : ', best_params)

print('\nbest_estimator : ', best_estimator)



score_train = grid_model.score(X_train, y_train)

score_test = grid_model.score(X_test, y_test)



print('\ntraining score : ', score_train)

print('\ntesting score : ', score_test)
grid_model.cv_results_
cv_results = pd.DataFrame(grid_model.cv_results_)
cv_results
df_min_results = cv_results[['param_C', 'param_kernel', 'mean_test_score']]
df_min_results
grid_model.best_score_
grid_model.best_params_
from sklearn.metrics import classification_report
y_pred = grid_model.predict(X_test)
# classification_report(y_test, y_pred)
print(classification_report(y_test, y_pred))