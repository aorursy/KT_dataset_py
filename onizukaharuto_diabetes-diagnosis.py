# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/test.csv', index_col=0)
df_train_dummies = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)

df_test_dummies = pd.get_dummies(df_test, columns=['Gender'], drop_first=True)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



X_train_dummies = df_train_dummies.drop(columns='Diabetes').values

y_train_dummies = df_train_dummies['Diabetes'].values

X_train, X_valid, y_train, y_valid = train_test_split(X_train_dummies, y_train_dummies, test_size=0.2, random_state=0)



rfc = RandomForestClassifier(random_state=0)

rfc.fit(X_train, y_train)
from sklearn.metrics import roc_curve, auc



y_pred = rfc.predict_proba(X_valid)[:,1]  # 予測

fpr, tpr, thresholds = roc_curve(y_valid, y_pred)  # ROC曲線を求める

auc(fpr, tpr)  # 評価
import optuna

def objective(trial):

    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    max_depth = trial.suggest_int('max_depth', 1, 30)

    n_estimators = trial.suggest_int('n_estimators',10,300)

    model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_valid)[:,1]  # 予測

    fpr, tpr, thresholds = roc_curve(y_valid, y_pred)  # ROC曲線を求める

    return (auc(fpr, tpr))  # 評価



study = optuna.create_study()

study.optimize(objective, n_trials=100)

study.best_params
criterion=study.best_params['criterion']

max_depth=study.best_params['max_depth']

n_estimators=study.best_params['n_estimators']

model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

model.fit(X_train, y_train)

y_pred = model.predict_proba(X_valid)[:,1]  # 予測

fpr, tpr, thresholds = roc_curve(y_valid, y_pred)  # ROC曲線を求める

auc(fpr, tpr)  # 評価
model.fit(X_train_dummies, y_train_dummies)



X_test = df_test_dummies.values

y_pred = rfc.predict_proba(X_test)[:, 1]
submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = y_pred

submit.to_csv('submission.csv', index=False)