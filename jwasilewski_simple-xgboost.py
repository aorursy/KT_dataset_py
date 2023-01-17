import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import xgboost as xgb
data = pd.read_csv("/kaggle/input/iris/Iris.csv", index_col=0)

data.head()
from pandas.api.types import CategoricalDtype

species = CategoricalDtype(data["Species"].unique(), ordered=False)

species
X = data[data.columns[0:4]].values

y = data["Species"].astype(species).cat.codes.values

y = np.reshape(y, (-1,1))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)
param = {

    'max_depth': 2,

    'eta': 0.1,

    'objective': 'multi:softmax',

    'num_class': 3

}



num_round = 10



bst = xgb.train(param, dtrain, num_round)
from sklearn.metrics import accuracy_score



pred = bst.predict(dtest)

accuracy_score(y_test, pred)