# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn import decomposition
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV  
import imblearn
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read training data
data = pd.read_csv('../input/train.csv')
from sklearn.model_selection import train_test_split
train_X = data.drop(["target","id"], axis=1)
le = LabelEncoder()
le.fit(data["target"])
train_y = le.transform(data["target"])
# split train set into 2 parts with same distribution: 80% train, 20% validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(train_X.values, train_y):
    X_train = train_X.values[train_index]
    X_val = train_X.values[test_index]

    y_train = train_y[train_index]
    y_val = train_y[test_index]
missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column.sum())
data.describe()
data["target"].value_counts().plot.bar()
data["target"].value_counts()
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X_train, y_train)

unique, counts = np.unique(y_ros, return_counts=True)

print(np.asarray((unique, counts)).T)
pd.Series(y_ros).value_counts().plot.bar()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_data = pd.read_csv('../input/test.csv')
test_X = test_data.drop(["id"], axis=1)
scaler_all = StandardScaler()
train_X_scaled = scaler_all.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)
pca = decomposition.PCA(n_components=20)
pca.fit(X_train_scaled)

X_train_pca = pca.transform(X_train_scaled)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
pca = decomposition.PCA()
pca.fit(X_train_scaled)

X_train_pca = pca.transform(X_train_scaled)
#print(np.cumsum(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
xgb = XGBClassifier()
xgb.fit(X_train_scaled, y_train)
preds = xgb.predict_proba(X_val_scaled)
score = log_loss(y_val, preds)
print("test data log loss eval : {}".format(log_loss(y_val,preds)))
xgb.get_params
from sklearn.model_selection import GridSearchCV

"""
param_test = {
    'n_estimators': [300],
    'n_jobs': [4], #Number of jobs to run in parallel. -1 means using all processors
}
gsearch = GridSearchCV(estimator = XGBClassifier(), param_grid = param_test, scoring='neg_log_loss', n_jobs=-1,iid=False, cv=3,verbose=1, return_train_score=True)
gsearch.fit(X_train_scaled,y_train)
pd.DataFrame(gsearch.cv_results_)
"""
scores = []
n_estimators = [100,200,400,450,500,525,550,600,700]

for nes in n_estimators:
    xgb = XGBClassifier(learning_rate =0.1, n_estimators=nes, max_depth=7, min_child_weight=3, subsample=0.8, 
                             colsample_bytree=0.8, nthread=4, seed=42, objective='multi:softprob')
    xgb.fit(X_train_scaled, y_train)
    preds = xgb.predict_proba(X_val_scaled)
    score = log_loss(y_val, preds)
    scores.append(score)
    print("test data log loss eval : {}".format(log_loss(y_val,preds)))
plt.plot(n_estimators,scores,'o-')
plt.ylabel(log_loss)
plt.xlabel("n_estimator")
print("best n_estimator {}".format(n_estimators[np.argmin(scores)]))
scores_md = []
max_depths = [1,3,5,6,7,8,10]

for md in max_depths:
    xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmin(scores)], 
                        max_depth=md, min_child_weight=3, subsample=0.8, 
                        colsample_bytree=0.8, nthread=4, seed=42, objective='multi:softprob')
    xgb.fit(X_train_scaled, y_train)
    preds = xgb.predict_proba(X_val_scaled)
    score = log_loss(y_val, preds)
    scores_md.append(score)
    print("test data log loss eval : {}".format(log_loss(y_val,preds)))
plt.plot(max_depths,scores_md,'o-')
plt.ylabel(log_loss)
plt.xlabel("max_depth")
print("best max_depth {}".format(max_depths[np.argmin(scores_md)]))
scores_mcw = []
min_child_weights = [1,2,3,4,5]

for mcw in min_child_weights:
    xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmin(scores)],
                        max_depth=max_depths[np.argmin(scores_md)], 
                        min_child_weight=mcw, subsample=0.8, 
                        colsample_bytree=0.8, nthread=4, seed=42, objective='multi:softprob')
    xgb.fit(X_train_scaled, y_train)
    preds = xgb.predict_proba(X_val_scaled)
    score = log_loss(y_val, preds)
    scores_mcw.append(score)
    print("test data log loss eval : {}".format(log_loss(y_val,preds)))
plt.plot(min_child_weights,scores_mcw,"o-")
plt.ylabel(log_loss)
plt.xlabel("min_child_weight")
print("best min_child_weight {}".format(min_child_weights[np.argmin(scores_mcw)]))
scores_ss = []
subsamples = [0.5,0.6,0.7,0.8,0.9,1]

for ss in subsamples:
    xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmin(scores)], 
                        max_depth=max_depths[np.argmin(scores_md)],
                        min_child_weight=min_child_weights[np.argmin(scores_mcw)], subsample=ss, 
                        colsample_bytree=0.8, nthread=4, seed=42, objective='multi:softprob')
    xgb.fit(X_train_scaled, y_train)
    preds = xgb.predict_proba(X_val_scaled)
    score = log_loss(y_val, preds)
    scores_ss.append(score)
    print("test data log loss eval : {}".format(log_loss(y_val,preds)))
plt.plot(subsamples,scores_ss,"o-")
plt.ylabel(log_loss)
plt.xlabel("subsample")
print("best subsample {}".format(subsamples[np.argmin(scores_ss)]))
scores_cb = []
colsample_bytrees = [0.5,0.6,0.7,0.8,0.9,1]

for cb in colsample_bytrees:
    xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmin(scores)], 
                        max_depth=max_depths[np.argmin(scores_md)], 
                        min_child_weight=min_child_weights[np.argmin(scores_mcw)], 
                        subsample=subsamples[np.argmin(scores_ss)], 
                        colsample_bytree=cb, nthread=4, seed=42, objective='multi:softprob')
    xgb.fit(X_train_scaled, y_train)
    preds = xgb.predict_proba(X_val_scaled)
    score = log_loss(y_val, preds)
    scores_cb.append(score)
    print("test data log loss eval : {}".format(log_loss(y_val,preds)))
plt.plot(colsample_bytrees,scores_cb,"o-")
plt.ylabel(log_loss)
plt.xlabel("colsample_bytree")
print("best colsample_bytree {}".format(colsample_bytrees[np.argmin(scores_cb)]))
scores_eta = []
etas = [0.001,0.01,0.1,0.2,0.3,0.5,1]

for eta in etas:
    xgb = XGBClassifier(learning_rate =eta, n_estimators=n_estimators[np.argmin(scores)], 
                        max_depth=max_depths[np.argmin(scores_md)], 
                        min_child_weight=min_child_weights[np.argmin(scores_mcw)], 
                        subsample=subsamples[np.argmin(scores_ss)], 
                        colsample_bytree=colsample_bytrees[np.argmin(scores_cb)], 
                        nthread=4, seed=42, objective='multi:softprob')
    xgb.fit(X_train_scaled, y_train)
    preds = xgb.predict_proba(X_val_scaled)
    score = log_loss(y_val, preds)
    scores_eta.append(score)
    print("test data log loss eval : {}".format(log_loss(y_val,preds)))
plt.plot(etas,scores_eta,"o-")
plt.ylabel(log_loss)
plt.xlabel("eta")
print("best eta {}".format(etas[np.argmin(scores_eta)]))
xgb = XGBClassifier(learning_rate =eta, n_estimators=n_estimators[np.argmin(scores)], 
                        max_depth=max_depths[np.argmin(scores_md)], 
                        min_child_weight=min_child_weights[np.argmin(scores_mcw)], 
                        subsample=subsamples[np.argmin(scores_ss)], 
                        colsample_bytree=colsample_bytrees[np.argmin(scores_cb)], 
                        nthread=4, seed=42, objective='multi:softprob')
calibrated_xgb = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
calibrated_xgb.fit(X_train_scaled, y_train)
preds = calibrated_xgb.predict_proba(X_val_scaled)
score = log_loss(y_val, preds)
scores_eta.append(score)
print("test data log loss eval : {}".format(log_loss(y_val,preds)))
xgb = XGBClassifier(learning_rate =0.1, n_estimators=525, max_depth=8, min_child_weight=3, subsample=0.7, 
                       colsample_bytree=0.7, nthread=4, seed=42, objective='multi:softprob')
my_model = CalibratedClassifierCV(xgb, cv=5, method='isotonic')
my_model.fit(train_X_scaled,train_y)
test_preds = my_model.predict_proba(test_X_scaled)
output = pd.DataFrame(test_preds,columns=["Class_"+str(i) for i in range(1,10)])
output.insert(loc=0, column='id', value=test_data.id)
output.to_csv('submission.csv', index=False)
test_data.head()
output.head()