# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir('../input')

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/forest-cover-type-prediction/train.csv')

df.head()
df.columns
df.select_dtypes('object').columns
X = df.drop('Cover_Type', axis=1)

y = df['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=0)

model.fit(X_train, y_train)

accuracy_score(model.predict(X_test), y_test)
model.fit(X, y)
test_df = pd.read_csv('../input/forest-cover-type-prediction/test.csv')

test_df.head()
submission = test_df[['Id']]

submission['Cover_Type'] = model.predict(test_df)
submission.to_csv('submission.csv',index=False)