!curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
PATH = os.path.join("/kaggle", "working", "flightdata.csv")
data = pd.read_csv(PATH).drop("Unnamed: 25", axis = 1)
data.head()
data.info()
data.isna().sum()
data = data[["MONTH", "DAY_OF_MONTH","DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]
data.head()
data.isnull().sum()
data["ARR_DEL15"] = data["ARR_DEL15"].fillna(1)
data.isnull().sum()
data.shape
data = pd.get_dummies(data, columns = ["ORIGIN", "DEST"])
data.head()
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = shuffle(data)
y = data["ARR_DEL15"]
data = data.drop("ARR_DEL15", axis = 1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.25, random_state = 11)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
import xgboost as xgb
from xgboost import XGBClassifier
bst = XGBClassifier(
 learning_rate = 0.3,
 eta =  0.3,
 n_estimators = 50,
 max_depth = 6,
 min_child_weight = 1,
 gamma = 0,
 subsample = 0.8,
 colsample_bytree = 0.8,
 objective = 'binary:logistic',
 nthread = 2,
 scale_pos_weight = 1,
 seed=27)
from sklearn.model_selection import cross_val_score
cross_val_score(bst, X_train, y_train)
bst = bst.fit(X_train, y_train)
accuracy_score(bst.predict(X_train), y_train)
from sklearn.metrics import accuracy_score
accuracy_score(bst.predict(X_test), y_test)
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

params = {"n_estimators": 1000, 'booster': 'gbtree', 'max_depth': 4, 'eta': 0.3, 
    'silent': 1, 'objective': 'binary:logistic', 'nthread': 1}
model = xgb.train(params, dtrain, num_boost_round = 10)
y_pred = model.predict(xgb.DMatrix(X_test))
y_pred = np.round(y_pred)
y_pred
np.unique(y_pred)
accuracy_score(np.round(model.predict(xgb.DMatrix(X_train))), y_train)
xgb.cv(params, dtrain, nfold = 5)
accuracy_score(y_pred, y_test)
