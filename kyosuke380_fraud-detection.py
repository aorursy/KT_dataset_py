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
df_train = pd.read_csv('../input/1056lab-fraud-detection-in-credit-card/train.csv',index_col=0)

df_test = pd.read_csv('../input/1056lab-fraud-detection-in-credit-card/test.csv',index_col=0)

df_test
df_train.isnull().sum()
df_train['Class'].value_counts()
from sklearn.metrics import accuracy_score, confusion_matrix



X = df_train.drop(['Time','Class'],axis=1)

y = df_train['Class'].values

X_test=df_test.drop(['Time'],axis=1)
# ライブラリ

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

# 正例の数を保存

positive_count_train = y.sum()

print('positive count:{}'.format(positive_count_train))



# アンダーサンプリングで正常利用の割合を減らし不正利用の割合を1%まで増やす

rus = RandomUnderSampler(sampling_strategy={0:positive_count_train*220, 1:positive_count_train},random_state=0)

#学習用データに反映

X_train_undersampled, y_train_undersampled = rus.fit_sample(X, y)



# SMOTEで不正利用の割合を約20%まで増やす

smote = SMOTE(sampling_strategy={0:positive_count_train*220, 1:positive_count_train*20},random_state=0)



# 学習用データに反映

X_train_resampled, y_train_resampled = smote.fit_sample(X_train_undersampled, y_train_undersampled)

print('X_train_resampled.shape: {}, y_train_resampled: {}'.format(X_train_resampled.shape, y_train_resampled.shape))

print('y_train_resample:\n{}'.format(pd.Series(y_train_resampled).value_counts()))

X_train_resampled
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

params = {'criterion':('gini', 'entropy'),'max_depth':[1, 2, 3],'n_estimators':[50,100,200]}

gscv = GridSearchCV(clf, params, cv=5,scoring='roc_auc')

gscv.fit(X_train_resampled, y_train_resampled)
scores = gscv.cv_results_['mean_test_score']

params = gscv.cv_results_['params']

for score, param in zip(scores, params):

  print('%.3f  %r' % (score, param))
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))
p = gscv.predict(X_test)
df_submit=pd.read_csv('../input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv',index_col=0)

df_submit['Class']=p

df_submit.to_csv('Submission5.csv')