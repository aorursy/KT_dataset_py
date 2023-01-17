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
df_train = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv', index_col=0)
X_train = df_train.drop('Class', axis=1).values

y_train = df_train['Class'].values
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')

clf.fit(X_train, y_train)
clf.score(X_train, y_train)
X_test = df_test.values

p = clf.predict_proba(X_test)
df_submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv', index_col=0)

df_submit['Class'] = p[:,1]

df_submit.to_csv('submission4.csv')