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
train_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv', index_col=0)
test_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv', index_col=0)
inttrain_df = train_df.select_dtypes(include='int')
inttrain_df
inttest_df = test_df.select_dtypes(include='int')
inttest_df
all_int = pd.concat([inttrain_df.drop('value_eur', axis=1), inttest_df])
columns = all_int.columns

for c in columns:

    all_int[c] = pd.to_numeric(all_int[c], errors='ignore')

all_int
all_int.dtypes
X_train = all_int[:len(inttrain_df)].to_numpy()

y_train = train_df['value_eur'].to_numpy()



X_test = all_int[len(inttrain_df):].to_numpy()
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

reg = DecisionTreeRegressor()

params = {'criterion':('mae', 'mse'), 'max_depth':[5, 10]}

gscv = GridSearchCV(reg, params, cv=3, scoring='neg_mean_squared_log_error')

gscv.fit(X_train, y_train)
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))
reg = DecisionTreeRegressor(criterion = 'mae', max_depth = 5)
reg.fit(X_train, y_train)
reg.score(X_train, y_train)
p_test = reg.predict(X_test)
p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = p_test
submit_df.to_csv('submission.csv')