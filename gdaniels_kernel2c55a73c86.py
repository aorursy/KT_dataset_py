# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split



from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



import sklearn.metrics

series_data = pd.read_csv("../input/careercon2019/X_series_train.csv")
# Create test and train data sets

X = series_data.drop(columns= ['series_id', 'group_id', 'surface'] )



y_raw = series_data['surface']

label = LabelEncoder()

y = label.fit_transform(y_raw)



test_size = 0.20

seed = 33



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# Create transformed dataset

scale = StandardScaler()

X_train_scale = scale.fit_transform(X_train)
# Create transformed dataset

scale = StandardScaler()

X_scale = scale.fit_transform(X)
# Create transformed dataset

scale = StandardScaler()

X_test_scale = scale.fit_transform(X_test)
scoring = 'accuracy'

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
from numpy import loadtxt

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
model = XGBClassifier(learning_rate=0.01, n_estimators=1000)

model.fit(X_scale, y)
X_kaggle_test_raw.info()
X_kaggle_test_raw = pd.read_csv("../input/careercon2019/X_series_test.csv")

X_kaggle_test = X_kaggle_test_raw.drop(columns= ['0'] )
# Create transformed dataset

scale = StandardScaler()

X_kaggle_test_scale = scale.fit_transform(X_kaggle_test)
y_test_pred = model.predict(X_kaggle_test_scale)
predictions = label.inverse_transform(y_test_pred)
submit_raw = pd.read_csv('../input/careercon2019/X_series_test.csv')
submit = pd.DataFrame()

submit['series_id'] = submit_raw['0']

submit['surface'] = predictions
submit.to_csv('submission.csv', index=False)
submit