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
!ls -l 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/BlackFriday.csv")
df.shape
df.head()
df.isnull().sum()
df2 = df.drop(columns=['Product_Category_2', 'Product_Category_3'])
df2.shape
%%time
import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split

train, test, y_train, y_test = train_test_split(df2.drop(["Purchase"], axis=1), df2["Purchase"],
                                                random_state=10, test_size=0.25)

train.shape

train.head(3)
y_train.head(3)
%%time
import lightgbm as lgb
from sklearn import metrics, model_selection
def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)

'''
class lightgbm.LGBMModel(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100,
    subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20,
    subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True,
    importance_type='split', **kwargs)[source]
'''
param_dist = {"max_depth": [-1], # default -1
              "learning_rate" : [0.1, 0.2], # default 0.1
              "num_leaves": [31], # default 31
              "n_estimators": [100] # default 100
             }
grid_search = model_selection.GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=10)
grid_search.fit(train,y_train)

print("{}".format(grid_search.best_estimator_))
print("{}".format(grid_search.best_score_))
print("{}".format(grid_search.best_params_))