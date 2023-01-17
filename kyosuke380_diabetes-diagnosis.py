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
df_train = pd.read_csv('../input/1056lab-diabetes-diagnosis/train.csv',index_col=0)

df_test = pd.read_csv('../input/1056lab-diabetes-diagnosis/test.csv',index_col=0)

df_train
df_train['Diabetes'].value_counts()
df_train_dummies = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)

df_train_dummies
df_test_dummies = pd.get_dummies(df_test, columns=['Gender'], drop_first=True)

df_test_dummies
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.pairplot(df_train_dummies, hue='Diabetes')

plt.show()
df_train_dummies2 = df_train_dummies[['HDL Chol', 'Chol/HDL ratio', 'Weight', 'Systolic BP', 'Diastolic BP','Waist','Hip','Diabetes']]  # 列を選択

df_test_dummies2 = df_test_dummies[['HDL Chol', 'Chol/HDL ratio', 'Weight', 'Systolic BP', 'Diastolic BP','Waist','Hip']]  # 列を選択
df_train_dummies2
df_test_dummies2
from sklearn.model_selection import train_test_split

from sklearn.metrics import auc

X_train_dummies = df_train_dummies2.drop('Diabetes', axis=1).values

y_train_dummies = df_train_dummies2['Diabetes'].values

X_train, X_valid, y_train, y_valid = train_test_split(X_train_dummies, y_train_dummies, test_size=0.2, random_state=0)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

params = {'criterion':('gini', 'entropy'),

          'n_estimators':[240,260,280,300],

          'max_depth':[1, 2, 3, 4, 5],

          'random_state':[5]

         }

gscv = GridSearchCV(clf, params, cv=5,scoring='roc_auc')

gscv.fit(X_train, y_train)
scores = gscv.cv_results_['mean_test_score']

params = gscv.cv_results_['params']

for score, param in zip(scores, params):

  print('%.3f  %r' % (score, param))
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))
clf = RandomForestClassifier(criterion='entropy', max_depth= 4, n_estimators= 280,random_state= 5)

clf.fit(X_train_dummies,y_train_dummies)
X_test = df_test_dummies2.values  # Numpy行列を取り出す

y_pred = clf.predict_proba(X_test)[:, 1]  # 予測
submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = y_pred

submit.to_csv('submission2.csv', index=False)