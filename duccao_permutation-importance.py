import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
import os
os.listdir('../input')
os.listdir('../input/hospital-readmissions')
os.listdir('../input/new-york-city-taxi-fare-prediction')
data = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv')
data.head()
from sklearn.ensemble import RandomForestRegressor

y = data['passenger_count']
feature_names = [i for i in data.columns if data[i].dtype in [np.float64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestRegressor(random_state=0).fit(train_X, train_y)
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
data = pd.read_csv('../input/hospital-readmissions/train.csv')
data.head()
data.columns.tolist()
y = (data['readmitted'] == 1)
feature_names = [i for i in data.columns if i != 'readmitted']
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())