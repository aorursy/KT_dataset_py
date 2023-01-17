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
path = '/kaggle/input/1056lab-brain-cancer-classification/'
df_train = pd.read_csv(path + '/train.csv', index_col=0)
df_test = pd.read_csv(path + '/test.csv', index_col=0)
df_train
df_train['type'] = df_train['type'].map({'normal':0, 'ependymoma':1, 'glioblastoma':2, 'medulloblastoma':3, 'pilocytic_astrocytoma':4})
df_train
from collections import Counter
print('Original dataset shape %s' % Counter(df_train['type']))
train_X = df_train.drop(columns='type').values
train_y = df_train['type'].values
from sklearn.decomposition import PCA
fs = PCA(n_components=16)
fs.fit(train_X)
train_X_ = fs.transform(train_X)
train_X_
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
X_train, X_valid, y_train, y_valid = train_test_split(train_X_, train_y, test_size=0.3) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

estimators = [
        ('svc', SVC(max_iter=200, random_state=0)),
        ('lgb', lgb.LGBMClassifier(learning_rate=0.01, max_depth=2,
                                   n_estimators=200, num_leaves=5, 
                                   random_state=0))
        ]

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=300),
)
clf.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_valid)
confusion_matrix(y_valid, y_pred)
from sklearn.metrics import f1_score
f1_score(y_valid, y_pred, average=None)
clf.fit(train_X_, train_y)
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(train_X_)
confusion_matrix(train_y, y_pred)
from sklearn.metrics import f1_score
f1_score(train_y, y_pred, average=None)
test = fs.transform(df_test.values)
test
y_pred = clf.predict(test)
submit = pd.read_csv(path + '/sampleSubmission.csv')
submit['type'] = y_pred
submit.to_csv('submission7.csv', index=False)
