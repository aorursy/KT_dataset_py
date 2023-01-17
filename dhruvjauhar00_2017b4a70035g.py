# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_table = pd.read_csv('../input/minor-project-2020/train.csv')

# I had locally found that col_39 has high correlations, and decided to not continue with it.

data_table.drop(['id', 'col_39'], axis=1, inplace = True)
Y = data_table.target.values
X = data_table.drop(['target'], axis=1).values

#most of what I have followed are through Kaggle tutorials uploaded by other users and on geeksforgeeks.
#it is a very straightforward approach.
from sklearn.preprocessing import StandardScaler, PowerTransformer
sc = StandardScaler()
X = sc.fit_transform(X)
import imblearn
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_train, Y_train = sm.fit_sample(X, Y.ravel())
test = pd.read_csv('../input/minor-project-2020/test.csv')
test.drop(['id', 'col_39'], axis=1, inplace = True)
test = sc.fit_transform(test)
from sklearn.linear_model import LogisticRegression
#I had locally done a gridsearch on train data, and using that C value I got my highest submission score.
#I think the C value was 7 but I'm not 100% sure.

regression_result = LogisticRegression(C=7, max_iter=1000)
regression_result.fit(X_train, Y_train)
prediction=regression_result.predict(test)
#used in one of my submissions but not the other

#parameters = {
#    'C': np.linspace(1,1000, 10)
#             }
#regression = LogisticRegression(max_iter=1000)
#grid = GridSearchCV(regression, parameters, cv=3, n_jobs=-1)
#grid.fit(X_train, Y_train)
#prediction=grid.predict(test)
test1 = pd.read_csv('../input/minor-project-2020/test.csv')
idlist = test1["id"]
output = pd.DataFrame(data={"id" : idlist, "target" : prediction})
output.to_csv('dhruv_predictions15.csv',index=False)