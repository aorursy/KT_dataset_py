# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt#Visual representation and EDA
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading the data
data = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
data
data.shape
data.columns
for i in data.columns:
    if type(data[i].iloc[0]) == str:
        factor = pd.factorize(data[i])
        data[i] = factor[0]
        definitions = factor[1]
from sklearn.model_selection import train_test_split
#Independent Vector
X = data[list(data.columns)[:-1]].values
#Dependent Vector
y = data[data.columns[-1]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(np.nan_to_num(X_train))
X_test = scaler.transform(np.nan_to_num(X_test))
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
model = RandomForestClassifier(n_jobs=64,n_estimators=200,criterion='entropy',oob_score=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc =  metrics.accuracy_score(y_test, y_pred)
from sklearn.metrics import roc_curve, auc
print('accuracy ' +str(acc))
#print('average auc ' +str(roc_auc["average"]))
prfs = precision_recall_fscore_support(y_test, y_pred, labels = [0,1])
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('precision:',prfs[0] )
print('recall', prfs[1])
print('fscore', prfs[2])
from sklearn.model_selection import StratifiedKFold
from xgboost  import XGBClassifier
from sklearn.metrics import roc_auc_score
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
from sklearn.model_selection import RandomizedSearchCV
folds = 3
param_comb = 5
xgb = XGBClassifier(n_estimators=100)
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb,
                                   scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=3, random_state=1001)
random_search.fit(X, y)
xgb2 = XGBClassifier(base_score=0.5, booster='gbtree',
                                           colsample_bylevel=1,
                                           colsample_bynode=1,
                                           colsample_bytree=1, gamma=0,
                                           gpu_id=-1, importance_type='gain',
                                           interaction_constraints='',
                                           learning_rate=0.300000012,
                                           max_delta_step=0, max_depth=6,
                                           min_child_weight=1, missing=np.nan,
                                           num_parallel_tree=1, random_state=0,
                                           reg_alpha=0, reg_lambda=1,
                                           scale_pos_weight=1, subsample=1,
                                           tree_method='exact',
                                           validate_parameters=1,
                                           verbosity=None)
training_start = time.perf_counter()
xgb2.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb2.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))
