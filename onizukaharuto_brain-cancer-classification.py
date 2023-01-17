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
df_train = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/test.csv', index_col=0)
df_train['type'] = df_train['type'].map({'normal':0, 'ependymoma':1, 'glioblastoma':2, 'medulloblastoma':3, 'pilocytic_astrocytoma':4})
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



X = df_train.drop(columns = 'type',axis=1).values

y = df_train['type'].values

X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=0)



rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train,y_train)
from sklearn.metrics import f1_score

y_pred = rfc.predict(X_valid)

f1_score(y_valid,y_pred,average = 'weighted')
#単変量解析のスコアが低い特徴を削除する

#from sklearn.feature_selection import SelectKBest

#fs = SelectKBest(k=20)

#fs.fit(X, y)

#X_ = fs.transform(X)
#主成分分析を用いて次元を削減する

from sklearn.decomposition import PCA

fs = PCA(n_components=20)

fs.fit(X)

X_ = fs.transform(X)
X_train,X_valid,y_train,y_valid = train_test_split(X_,y,test_size=0.2,random_state=0)

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_valid)

f1_score(y_valid,y_pred,average = 'weighted')
#from imblearn.over_sampling import RandomOverSampler

#sampler = RandomOverSampler(random_state=0)

#X_res, y_res = sampler.fit_resample(X_train, y_train)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42,k_neighbors=2)

X_res, y_res = sm.fit_resample(X_train, y_train)
from collections import Counter

Counter(y_train)
Counter(y_res)
import optuna

def objective(trial):

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

#    splitter = trial.suggest_categorical('splitter', ['best', 'random'])

    max_depth = trial.suggest_int('max_depth', 1, 30)

    n_estimators = trial.suggest_int('n_estimators',10,300)

#    min_samples_split = trial.suggest_int('min_samples_split', 1, 5)

#    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators, random_state=0, n_jobs=-1)

    model.fit(X_res, y_res)

    y_pred = model.predict(X_valid)

    return -f1_score(y_valid,y_pred,average = 'weighted')



study = optuna.create_study()

study.optimize(objective, n_trials=100)

study.best_params
criterion=study.best_params['criterion']

max_depth=study.best_params['max_depth']

n_estimators=study.best_params['n_estimators']

#splitter=study.best_params['splitter']

#min_samples_leaf=study.best_params['min_samples_leaf']

#model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf,random_state=0)

model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators, random_state=0, n_jobs=-1)

model.fit(X_res, y_res)

y_pred = model.predict(X_valid)

f1_score(y_valid,y_pred,average = 'weighted')
model.fit(X_,y)
y_pred = model.predict(X_)

f1_score(y,y_pred,average = 'weighted')
X_test = df_test.values

X_test_ = fs.transform(X_test)

p = model.predict(X_test_)
submit = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv')

submit['type'] = p

submit.to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



sns.countplot(df_train['type'])

plt.show()