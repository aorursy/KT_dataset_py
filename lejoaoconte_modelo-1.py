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
data_train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col = 0)

y_train = data_train['Survived']
X_train = data_train.drop(['Survived'], axis = 1)
X_train.head()
y_train.head()
data_test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col = 0)

data_test.head()
X_train = X_train.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis = 1)
X_test = data_test.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis = 1)

X_train.head()
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train.head()
X_train.shape
X_test.shape
X_val = X_train.iloc[:100]
X_train = X_train.iloc[100:]

X_train.shape
y_val = y_train.iloc[:100]
y_train = y_train.iloc[100:]

y_train.shape
from sklearn.tree import DecisionTreeClassifier
y_train.isnull().sum()
X_train.isnull().sum()
idade_mediana = X_train['Age'].median()

X_train['Age'].fillna(idade_mediana, inplace = True)
X_val['Age'].fillna(idade_mediana, inplace = True)
X_test['Age'].fillna(idade_mediana, inplace = True)
X_train.isnull().sum()
X_val.isnull().sum()
X_train.shape[1]
model_dt = DecisionTreeClassifier(random_state = 0, max_depth = 4, class_weight="balanced")
model_dt.fit(X_train, y_train)
from sklearn.tree import plot_tree
import pylab
fig, ax = pylab.subplots(1,1, figsize=(20,20))
plot_tree(model_dt, ax = ax, feature_names = X_train.columns)
from sklearn.metrics import roc_auc_score, average_precision_score
p = model_dt.predict_proba(X_val)[:, 1]
average_precision_score(y_val, p)
roc_auc_score(y_val, p)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators = 100, max_depth = 3, n_jobs = 1, random_state = 0)
model_rf.fit(X_train, y_train)

p_rf = model_rf.predict_proba(X_val)[:, 1]

print(average_precision_score(y_val, p_rf), roc_auc_score(y_val, p_rf))
from lightgbm import LGBMClassifier

model_lg = LGBMClassifier(random_state = 0, class_weight = "balanced", n_jobs = 5, learning_rate = 0.03)
model_lg.fit(X_train, y_train)

p_lg = model_lg.predict_proba(X_val)[:, 1]

print(average_precision_score(y_val, p_lg), roc_auc_score(y_val, p_lg))
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(random_state = 0, C = 0.5, n_jobs = 6)
model_lr.fit(X_train, y_train)

p_lr = model_lr.predict_proba(X_val)[:, 1]
print(average_precision_score(y_val, p_lr), roc_auc_score(y_val, p_lr))
p = (p_lr + p_rf + p_lg)/3
print(average_precision_score(y_val, p), roc_auc_score(y_val, p))
p
enviar = np.asarray(p)
enviar = np.rint(enviar)
enviar
X_test.head()
p_rf = model_rf.predict_proba(X_test)[:, 1]
p_lg = model_lg.predict_proba(X_test)[:, 1]
p_lr = model_lr.predict_proba(X_test)[:, 1]

p = (p_lr + p_rf + p_lg)/3

enviar = np.asarray(p)
enviar = np.rint(enviar)
enviar = enviar.astype(int)
enviar = pd.Series(enviar, index = X_test.index)
enviar.shape
enviar.to_csv('modelo_final')
