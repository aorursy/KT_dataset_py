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
#性別をダミー変数へ

df_train_dummies = pd.get_dummies(df_train, columns=['Gender'], drop_first=True)

df_test_dummies = pd.get_dummies(df_test, columns=['Gender'], drop_first=True)
from sklearn.tree import DecisionTreeClassifier



X_train_dummies = df_train_dummies.drop(columns='Diabetes').values

y_train_dummies = df_train_dummies['Diabetes'].values



clf = DecisionTreeClassifier(criterion='gini', max_depth=4)

clf.fit(X_train_dummies, y_train_dummies)



clf.score(X_train_dummies, y_train_dummies)
X_test = df_test_dummies.values

p = clf.predict_proba(X_test)[:, 1]
submit = pd.read_csv('/kaggle/input/1056lab-diabetes-diagnosis/sampleSubmission.csv')

submit['Diabetes'] = p

submit.to_csv('submission.csv', index=False)