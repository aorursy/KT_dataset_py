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
df=pd.read_csv("../input/bank-customers/Churn Modeling.csv")
df.head()
import seaborn as sns
import matplotlib.pyplot as plt

corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
fig=sns.heatmap(df[top_corr_features].corr(),annot=True,
                cmap="Accent")

X=df.iloc[:,3:13]
y=df.iloc[:,13]
drop_cols=["Geography","Gender"]
X.drop(drop_cols,axis=1,inplace=True)
X.head()
params={
    "learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth":[3,4,5,6,8,10,12,15],
    "min_child_weight":[1,3,5,7],
    "gamma":[0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree":[0.3,0.4,0.5,0.7]
}
from sklearn.model_selection import RandomizedSearchCV
import xgboost
classifier=xgboost.XGBClassifier(tree_method='gpu_hist',gpu_id=0)
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',cv=5,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time) # timing ends here for "start_time" variable
random_search.best_params_
random_search.best_estimator_
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.4, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=4,
              min_child_weight=3, 
              monotone_constraints='(0,0,0,0,0,0,0,0)', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)
from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y,cv=10)
print(score.mean())
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
accuracy=[]
skf=StratifiedKFold(n_splits=10,random_state=None)
skf.get_n_splits(X,y)
for train_index,test_index in skf.split(X,y):
    X1_train,X1_test=X.iloc[train_index],X.iloc[test_index]
    Y1_train,Y1_test=y.iloc[train_index],y.iloc[test_index]
    
    classifier.fit(X1_train,Y1_train)
    prediction=classifier.predict(X1_test)
    socre=accuracy_score(prediction,Y1_test)
    accuracy.append(score)

print(accuracy)
print(np.array(accuracy).mean())
