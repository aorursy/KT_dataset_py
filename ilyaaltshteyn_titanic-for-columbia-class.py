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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
from sklearn.ensemble import RandomForestClassifier

df = df.fillna(method='ffill')

X = df[['Pclass', 'Age', 'Fare']]
y = df['Survived']

clf = RandomForestClassifier()
clf_trained = clf.fit(X, y)

clf_trained.score(X, y)
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test = df_test.fillna(method='ffill')

X_test = df_test[['Pclass', 'Age', 'Fare']]

predictions = clf_trained.predict(X_test)

predictions_df = pd.DataFrame(
    pd.concat([df_test['PassengerId'], pd.DataFrame(predictions)], axis=1)
)
predictions_df.columns = ['PassengerId', 'Survived']
predictions_df.to_csv('predictions.csv', index=False)
