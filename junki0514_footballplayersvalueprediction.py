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
df_train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv')
df_train.isnull().sum()
df_train
X = df_train.drop('nation_position', axis=1)

X = X.drop('nation_jersey_number', axis=1)

X = X.drop('player_traits', axis=1)

X = X.drop('player_positions', axis=1)

X = X.drop('work_rate', axis=1)

X = X.drop('team_position', axis=1)

X = X.drop('joined', axis=1)

X = X.drop('ldm', axis=1)

X = X.drop('cdm', axis=1)

X = X.drop('rdm', axis=1)

X = X.drop('rwb', axis=1)

X = X.drop('lb', axis=1)

X = X.drop('lcb', axis=1)

X = X.drop('rcb', axis=1)

X = X.drop('rb', axis=1)
X = X.drop('ls', axis=1)

X = X.drop('st', axis=1)

X = X.drop('rs', axis=1)

X = X.drop('lf', axis=1)

X = X.drop('cf', axis=1)

X = X.drop('rf', axis=1)

X = X.drop('rw', axis=1)

X = X.drop('cam', axis=1)

X = X.drop('lam', axis=1)

X = X.drop('ram', axis=1)

X = X.drop('lm', axis=1)

X = X.drop('lcm', axis=1)

X = X.drop('cm', axis=1)

X = X.drop('rm', axis=1)

X = X.drop('lwb', axis=1)

X = X.drop('lw', axis=1)

X = X.drop('rcm', axis=1)

X = X.drop('cb', axis=1)
y = df_test.drop('nation_position', axis=1)

y = y.drop('nation_jersey_number', axis=1)

y = y.drop('player_traits', axis=1)

y = y.drop('player_positions', axis=1)

y = y.drop('work_rate', axis=1)

y = y.drop('team_position', axis=1)

y = y.drop('joined', axis=1)

y = y.drop('ldm', axis=1)

y = y.drop('cdm', axis=1)

y = y.drop('rdm', axis=1)

y = y.drop('rwb', axis=1)

y = y.drop('lb', axis=1)

y = y.drop('lcb', axis=1)

y = y.drop('rcb', axis=1)

y = y.drop('rb', axis=1)
y = y.drop('ls', axis=1)

y = y.drop('st', axis=1)

y = y.drop('rs', axis=1)

y = y.drop('lf', axis=1)

y = y.drop('cf', axis=1)

y = y.drop('rf', axis=1)

y = y.drop('rw', axis=1)

y = y.drop('cam', axis=1)

y = y.drop('lam', axis=1)

y = y.drop('ram', axis=1)

y = y.drop('lm', axis=1)

y = y.drop('lcm', axis=1)

y = y.drop('cm', axis=1)

y = y.drop('rm', axis=1)

y = y.drop('lwb', axis=1)

y = y.drop('lw', axis=1)

y = y.drop('rcm', axis=1)

y = y.drop('cb', axis=1)
X
X.isnull().sum()
X['loaned'] = X['loaned'].map({'no':1,'yes':2})

X['preferred_foot'] = X['preferred_foot'].map({'Right':1,'Left':2})
y['loaned'] = y['loaned'].map({'no':1,'yes':2})

y['preferred_foot'] = y['preferred_foot'].map({'Right':1,'Left':2})
y.isnull().sum()
X = X.fillna(0)
y = y.fillna(0)
X_ = X.drop('value_eur', axis=1).values

y_ = X['value_eur'].values

X_test = y.values
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.1)

sel.fit(X_)

X_ = sel.transform(X_)

X_test = sel.transform(X_test)
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

est = RandomForestClassifier()

fs  = SelectFromModel(est)

fs.fit(X_, y_)

X_ = fs.transform(X_)

X_test = fs.transform(X_test)
print(X_.shape)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_, y_, test_size=0.2, random_state=0)
import xgboost as xgb
clf = xgb.XGBRegressor()

clf.fit(X_train,y_train)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train, y_train)
p_test = model.predict(X_test)
df_submit = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

df_submit['value_eur'] = p_test

df_submit.to_csv('submission.csv')