# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/detection-of-parkinson-disease/parkinsons.csv')

df.head()
X = df.loc[:, df.columns!='status'].values[:, 1:]

y = df.loc[:, 'status'].values



print(X.shape, y.shape)

print('positive: ', len(y[y==1]), 'negative: ', len(y[y!=1]))
scaler = MinMaxScaler((-1, 1))

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

from sklearn.model_selection import GridSearchCV



param = {'max_depth': range(2, 10, 1),

         'n_estimators': range(50, 300, 50),

         'learning_rate': [0.1, 0.01, 0.05]}



estimator = XGBClassifier()

grid_model = GridSearchCV(

            estimator=estimator,

            param_grid=param,

            scoring='roc_auc',

            n_jobs=10,

            cv=10,

            verbose=True

            )

grid_model.fit(X_train, y_train)
print(grid_model.best_params_)

print('accuracy:', grid_model.best_score_)