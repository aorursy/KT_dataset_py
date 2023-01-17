# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', None)  # or 1000

pd.set_option('display.max_rows', None)  # or 1000

pd.set_option('display.max_colwidth', -1)  # or 199
df_train = pd.read_csv("../input/train.csv")

df_train.head()
df_train["target"].value_counts()
df_train.shape
df_test = pd.read_csv("../input/test.csv")

df_test.head()
df_train.describe()
X_train = df_train.iloc[:,1:-1]

y_train = df_train['target']
randomforest = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, bootstrap = True, oob_score = True, n_jobs =-1)
randomforest.fit(X_train, y_train)
X_test = df_test.drop(columns=['id'])



X_test.shape

y_test = randomforest.predict(X_test)
y_test
my_submission = pd.DataFrame({'Id': df_test['id'], 'Target': y_test})

my_submission.to_csv('submission.csv', index=False)