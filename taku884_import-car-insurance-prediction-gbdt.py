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
train = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)

test = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
train
test
train.dtypes
del train['normalized-losses']

del test['normalized-losses']
train = train.replace(['?'], np.nan)

test = test.replace(['?'], np.nan)
columns = train.columns

for c in columns:

    if train[c].isna().any():

        if train[c].dtypes != np.object:

            median = train[c].median()

            train[c] = train[c].replace(np.NaN, median)

        else:

            mfv = train[c].mode()[0]

            train[c] = train[c].replace(np.NaN, mfv)
columns = test.columns

for c in columns:

    if test[c].isna().any():

        if test[c].dtypes != np.object:

            median = test[c].median()

            test[c] = test[c].replace(np.NaN, median)

        else:

            mfv = test[c].mode()[0]

            test[c] = test[c].replace(np.NaN, mfv)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']:

  train[col] = le.fit_transform(train[col])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']:

  test[col] = le.fit_transform(test[col])
X = train.drop('symboling', axis=1).values

y = train['symboling'].values
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()

model.fit(X, y)
from sklearn.model_selection import GridSearchCV

model = GradientBoostingRegressor(n_estimators=200)

params = {'max_depth':[3, 4, 5, 6, 10, 100], 'learning_rate':[0.01, 0.02, 0.05, 0.1]}

gscv = GridSearchCV(model, params, cv = 10, scoring = 'neg_mean_squared_error', n_jobs = 1)

gscv.fit(X, y)
gscv.best_score_
p_test = gscv.predict(test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')