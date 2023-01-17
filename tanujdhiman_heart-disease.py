# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()
data.info()
data.describe()
import matplotlib.pyplot as plt
import seaborn as sns
data.hist(figsize = (10, 10))
plt.show()
plt.figure(figsize = (14, 14))
sns.heatmap(data.corr(), annot = True, cmap = "coolwarm")
plt.show()
data.target.unique()
have_disease = len(data[data.target == 1])
have_not_disease = len(data[data.target == 0])
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((have_disease / (len(data.target))*100)))
print("Percentage of Patients Have not Heart Disease: {:.2f}%".format((have_not_disease / (len(data.target))*100)))
sns.countplot(data.target)
plt.show()
data.target.value_counts()
data.sex.value_counts()
sns.countplot(data.sex)
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()
countFemale = len(data[data.sex == 0])
countMale = len(data[data.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(data.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(data.sex))*100)))
data.shape
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
X
y
params={
 "learning_rate"    : [ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15, 20],
 "min_child_weight" : [ 1, 3, 5, 7, 9 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 , 0.6],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 0.10 ],
 "subsample"        : [ 0.5, 0.6, 0.7, 0.8, 0.9],
 "nthread"          : [ 3, 4, 5, 6, 7],
 "scale_pos_weight" : [ 0.8, 0.9, 1, 1.1]
}
from sklearn.model_selection import RandomizedSearchCV
import xgboost
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
clf=xgboost.XGBClassifier()
random_search=RandomizedSearchCV(clf,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=7,verbose=3)
from datetime import datetime
start_time = timer(None) 
random_search.fit(X,y)
timer(start_time) 
random_search.best_estimator_

random_search.best_params_
clf=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.15, max_delta_step=0, max_depth=3,
              min_child_weight=3, missing=np.nan, monotone_constraints='()',
              n_estimators=100, n_jobs=5, nthread=6, num_parallel_tree=1,
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=0.8,
              subsample=0.5, tree_method='exact', validate_parameters=1,
              verbosity=None)
from sklearn.model_selection import cross_val_score
score=cross_val_score(clf,X,y,cv=10)
score
score.mean()