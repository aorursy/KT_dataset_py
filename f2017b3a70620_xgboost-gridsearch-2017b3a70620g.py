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

        

from sklearn.model_selection import train_test_split

        

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
df.head()
df['target'].value_counts()
df.info()
df.describe()
y_train = df['target']

X_train = df.drop('target',axis=1)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import RandomizedSearchCV
xgb = XGBClassifier(seed = 42)
# define evaluation procedure

cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=1)
parameters = {

    'max_depth': [2,4,6],

    'n_estimators': [60,120,200],

    'learning_rate': [0.1, 0.01, 0.05]

}
grid = RandomizedSearchCV(xgb,parameters, n_jobs=-1, cv=cv, scoring='roc_auc', verbose = 25)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# report all configurations

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
df_test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
y_pred_test = grid_result.predict_proba(df_test)[:,1]
y_pred_test
import pandas as pd
y_pred_test_pd = pd.DataFrame(data=y_pred_test, columns = ["Y_Pred"])
y_pred_test_pd.head()
my_submission = pd.DataFrame({'id': df_test.id, 'target': y_pred_test})
my_submission.head()
my_submission.to_csv('submission.csv', index=False)