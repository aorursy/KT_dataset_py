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
train = pd.read_csv('../input/train.csv')
train = train.fillna(-99)

correlation_with_target = {}
columns_use = []
tresh = 0.1
for c in train.columns:
    if c not in ['Target',"Id"]:
        corr_coef = train[c].corr(train['Target'])
        if np.abs(corr_coef) > tresh:
            print("correlation between %s and Target : %.2f" % (c,corr_coef))
            correlation_with_target[c] = np.abs(corr_coef)
            columns_use.append(c)
correlation_each_other = {}
_use =[]
delete_cols = []
for c in train.columns:
    for c2 in train.columns:
        name = "%s_%s" % (c,c2)
        name2 = "%s_%s" % (c2,c)
        if c not in ['Target',"Id"] and c != c2 and name not in _use :
            _use.extend([name,name2])
            try:
                corr_coef = train[c].corr(train[c2])
                
                if corr_coef >= 0.8:
                    print("correlation between %s and %s : %.2f" % (c,c2,corr_coef))
                    correlation_each_other[c] = np.abs(corr_coef)
                    delete_cols.append(c2)
            except:
                pass
delete_cols
ids = train['Id']
target = train['Target']
Train = train.drop(['Id','idhogar','dependency',
'edjefe',
'edjefa','Target'] + delete_cols,axis=1)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500,  max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0,max_leaf_nodes=None, min_impurity_decrease=0.0, 
                       min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, 
                       random_state=None, verbose=0, warm_start=False, class_weight=None)
test = pd.read_csv('../input/test.csv')
ids_test = test['Id']
Test = test.drop(['Id','idhogar','dependency',
'edjefe',
'edjefa'] + delete_cols,axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Train, target, 
                                                    test_size=0.4, random_state=0)
rf.fit(X_train,y_train)
rf.score(X_test, y_test)
from sklearn.model_selection import cross_val_score
cross_val_score(rf, Train, target, cv=5)
Test = Test.fillna(-99)
pred = rf.predict(Test)
sample = pd.DataFrame([],columns=["Id","Target"])
sample["Id"] = ids_test
sample['Target'] = pred
sample = sample.set_index('Id')
sample.to_csv("pred.csv")
