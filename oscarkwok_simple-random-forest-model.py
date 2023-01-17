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
train = pd.read_csv('/kaggle/input/learn-together/train.csv')
test = pd.read_csv('/kaggle/input/learn-together/test.csv')
train.head()
test.head()
from sklearn.model_selection import train_test_split
X = train.drop(['Id', 'Cover_Type'], axis=1)

y = train['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
from sklearn.metrics import classification_report, accuracy_score
rfc.score(X_train, y_train)
predictions = rfc.predict(X_test)
accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
test_Id = test['Id'] #store tests' Id column for the output file
test = test.drop('Id', axis=1) #delete the Id column for the prediction
test.head()
test_pred = rfc.predict(test)
output = pd.DataFrame({'Id': test_Id,

                       'Cover_Type': test_pred})

output.to_csv('submission.csv', index=False)