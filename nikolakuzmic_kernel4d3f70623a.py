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
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import gc
import os
import logging
import datetime
import warnings

from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

max_depth_val = 5
learning_rate_val = 1.0
seed = 42
df_train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
df_test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
X_train, y_train = df_train.iloc[:,2:], df_train[['target']]
X_test = df_test.iloc[:,1:]
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    tt = tt.loc[tt['Percent'] > 0]
    return(np.transpose(tt))
missing_data(df_train)
#Correlation between features and targets
features = X_train 
target = y_train
correlations = {}
pval = {}
for f in features.columns:
    x1 = features[f]
    x2 = target
    correlations[f],pval[f] = spearmanr(x1,x2)
    
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations = data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
data_correlations = data_correlations.iloc[0:30]
most_correlated_ftrs = data_correlations.index.to_list()
X_train, X_test = X_train[most_correlated_ftrs], X_test[most_correlated_ftrs]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
from sklearn import tree
from sklearn.metrics import accuracy_score
tree_clf = tree.DecisionTreeClassifier(max_depth=max_depth_val, random_state=seed)
tree_clf.fit(X_train, y_train)
y_pred_dtr = tree_clf.predict(X_val)

print('Acc: {}'.format(accuracy_score(y_val, y_pred_dtr)))
from xgboost import XGBClassifier
xgb_simple = XGBClassifier(max_depth=max_depth_val, n_estimators=100, random_state=seed)
xgb_simple.fit(X_train, y_train)
y_pred_xgb_simple = xgb_simple.predict(X_val)

print('Acc: {}'.format(accuracy_score(y_val, y_pred_xgb_simple)))
sub_df = pd.DataFrame({"ID_code":df_test["ID_code"].values})
sub_df["target"] = xgb_simple.predict(X_test)
sub_df.to_csv("submission.csv", index=False)
