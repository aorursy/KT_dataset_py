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
df_train['type'] = df_train['type'].map({'normal':0, 'ependymoma':1, 'pilocytic_astrocytoma':2, 'medulloblastoma':3, 'glioblastoma':4})
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.countplot(x='type', data=df_train)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

# Class count

count_class_0, count_class_1, count_class_2, count_class_3, count_class_4 = df_train.type.value_counts()

 

# Divide by class

df_class_0 = df_train[df_train['type'] == 0]

df_class_1 = df_train[df_train['type'] == 1]

df_class_2 = df_train[df_train['type'] == 2]

df_class_3 = df_train[df_train['type'] == 3]

df_class_4 = df_train[df_train['type'] == 4]



df_class_0_over = df_class_0.sample(count_class_0, replace=True)

df_class_2_over = df_class_2.sample(count_class_0, replace=True)

df_class_3_over = df_class_3.sample(count_class_0, replace=True)

df_class_4_over = df_class_4.sample(count_class_0, replace=True)



df_train_over = pd.concat([df_class_0_over, df_class_1, df_class_2_over, df_class_3_over, df_class_4_over], axis=0)



print(df_train_over.type.value_counts())



%matplotlib inline

sns.countplot(x='type', data=df_train_over)

plt.show()
X = df_train_over.drop('type', axis=1).values

y = df_train_over['type'].values
from sklearn.feature_selection import SelectKBest

fs = SelectKBest(k=20)

fs.fit(X, y)

X_ = fs.transform(X)
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



X_train, X_valid, y_train, y_valid = train_test_split(X_, y, test_size=0.2, random_state=0)  # 訓練用と検証用に分ける

model = SVC()

model.fit(X_train, y_train)  # 訓練用で学習

predict = model.predict(X_valid)

f1_score(y_valid, predict, average='weighted')
import optuna

from sklearn.svm import SVC



def objective(trial):

    kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])  # カーネル

    C = trial.suggest_loguniform('C', 1e2, 1e3)  # C

    model = SVC(kernel=kernel, C=C, gamma='auto')  # SVC

    model.fit(X_train, y_train)  # 訓練用で学習

    y_pred = model.predict(X_valid)  # 検証用を予測

    return f1_score(y_valid, y_pred, average='weighted')



study = optuna.create_study()  # Oputuna

study.optimize(objective, n_trials=10)  # 最適か

study.best_params  # 最適パラメーター
C = study.best_params['C']  # Cの最適値

kernel = study.best_params['kernel']  # カーネルの最適値

model = SVC(kernel=kernel, C=C, gamma='auto')  # 最適パラメーターのSVC

model.fit(X_train, y_train)  # 訓練用で学習

predict = model.predict(X_valid)  # 検証用を予測

f1_score(y_valid, predict, average='weighted')  # 検証用に対する予測を評価
X_test  = df_test.values

X_test_ = fs.transform(X_test)
from sklearn.svm import SVC



model = SVC(kernel=kernel, C=C, gamma='auto')  # 最適パラメーターのSVC

model.fit(X_, y)

predict = model.predict(X_test_)
submit = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv')

submit['type'] = predict

submit.to_csv('submission.csv', index=False)
predict