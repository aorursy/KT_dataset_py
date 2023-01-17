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
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import math
import dateutil.parser
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
path = '/kaggle/input/flavours-of-physics-kernels-only/'
train = pd.read_csv(path+'training.csv.zip', index_col='id')
test  = pd.read_csv(path+'test.csv.zip')
train.head(3)
features = list(train.columns[1:-5])

print("Random Forest")
rf = RandomForestClassifier(n_estimators=250, n_jobs=-1, criterion="entropy", random_state=11)
rf.fit(train[features], train["signal"])
print("XGBoost")
params = {"objective": "binary:logistic",
          "eta": 0.3,
          "max_depth": 5,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=250
gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)
print("Predictions on the test set")
test_probs = (rf.predict_proba(test[features])[:,1] +
              gbm.predict(xgb.DMatrix(test[features])))/2

