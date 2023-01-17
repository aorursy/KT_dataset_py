# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', header=0)



df.head()
from sklearn.model_selection import train_test_split

# df.drop(columns=['ID', 'LIMIT_BAL'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop('default payment next month', axis=1), df['default payment next month'].values, test_size=0.3, random_state=42)

df.head()
from sklearn.tree import DecisionTreeClassifier



classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,

 max_depth=None, min_samples_leaf=5, splitter='best')

classifier.fit(X_train, y_train)

classifier.score(X_train, y_train)
classifier.score(X_test, y_test)
from sklearn.metrics import roc_auc_score



train_pred = classifier.predict(X_train)

roc_auc_score(y_train, train_pred)
valid = pd.read_csv('../input/valid.csv', header=0)

test = pd.read_csv('../input/test.csv', header=0)

# test = test[-4500:]



result = pd.DataFrame(columns=['ID', 'Default'])

result.ID = np.append(valid['ID'].values, test['ID'].values)



# valid.drop(columns=['ID'], axis=1, inplace=True)



predictedValid = classifier.predict(valid)

predictedTest = classifier.predict(test)

predictions = np.append(predictedValid, predictedTest)

result.Default = predictions

result.to_csv("output.csv", encoding="utf8", index=False)

result.head()