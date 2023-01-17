# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import random

from sklearn.model_selection import KFold

import lightgbm as lgb

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
SEED = 2019

random.seed(SEED)

np.random.seed(SEED)
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df.head()
df['SARS-Cov-2 exam result'] = [0 if a == 'negative' else 1 for a in df['SARS-Cov-2 exam result'].values]



Y = df['SARS-Cov-2 exam result']



df = df.drop([

    "SARS-Cov-2 exam result",

    "Patient ID",

    'Patient addmited to regular ward (1=yes, 0=no)',

    'Patient addmited to semi-intensive unit (1=yes, 0=no)',

    'Patient addmited to intensive care unit (1=yes, 0=no)'

], axis=1)
df = df.fillna(df.mean())
# Fill NaNs with -10

df = df.fillna(-10, axis=1)
categorical_features = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object'] ]
X = pd.get_dummies(df, prefix=categorical_features, columns=categorical_features)
clf = RandomForestClassifier(max_depth=50, random_state=0,n_estimators=1000)

K = 10

folds = KFold(K, shuffle = True, random_state = SEED)



for fold , (train_index,test_index) in enumerate(folds.split(X, Y)):

    print('Fold:',fold+1)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    

    

    clf.fit(X_train,y_train)



    

    pred_y = clf.predict(X_test)

   

    

    print(classification_report(y_test,pred_y))
#Acurracy 0.90 but recall for the class "positive" close to zero or zero