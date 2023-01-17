# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_X = train.drop(['ID','TARGET'],axis=1)
test_X = test.drop(['ID'],axis=1)
train_Y = train.TARGET
ids = test.ID
from sklearn.ensemble import RandomForestClassifier
from sklearn.mode_selection import GridSearchCV

forest = RandomForestClassifier()

param_grid = {"max_depth": [3, None],
              "max_features": ['auto', 'sqrt', 'log2'],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
             "n_estimators" : [50,150,300]}

gs = GridSearchCV(estimator=forest,param_grid=param_grid, cv=3,n_jobs=-1)

gs.fit(train_X,train_Y)
Y_pred_forest = gs.predict_proba(test_X)
Y_pred = []
for n in Y_pred_forest:
    Y_pred.append(n[1])
submit = pd.DataFrame({'ID':ids,'TARGET':Y_pred}).set_index('ID')
submit.to_csv("submission.csv",encoding = 'utf-8')