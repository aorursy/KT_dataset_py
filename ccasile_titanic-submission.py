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
import pandas as pd



data = pd.read_csv('/kaggle/input/titanic/train.csv')

data_test =  pd.read_csv('/kaggle/input/titanic/test.csv')

data_test.head()
data_test.info()
data.head()
data.isna().sum()
data.info()
data.Age.fillna(30, inplace=True)

data.dropna(inplace=True, subset=['Embarked'])



data_test.Age.fillna(30, inplace=True)



data_test.info()
data.info()
y = data['Survived']

X = data

X_test_final = data_test

X.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived', 'Parch', 'Fare'], inplace=True)

X_test_final.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'Fare'], inplace=True, axis='columns')
X_test_final.isna().sum()
X_test_final.count()
X.head()
X = pd.get_dummies(X)

X_test_final = pd.get_dummies(X_test_final)

from sklearn import preprocessing



cols = X.columns

x = X.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, index=data.index, columns=cols)

X.head()



cols = X_test_final.columns

x = X_test_final.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X_test_final = pd.DataFrame(x_scaled, index=X_test_final.index, columns=cols)

X.head()
X.corr()
X.info()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)



logreg = LogisticRegression(random_state=0)



logreg.fit(X_train,y_train)



predictions = logreg.predict(X_test)

final_predictions = logreg.predict(X_test_final)

accuracy_score(y_test, predictions)
print(len(final_predictions))
id_row = data_test['PassengerId']

id_row.head()
my_submission = pd.DataFrame({'Id': id_row, 'Survived': final_predictions})

my_submission.to_csv('submission.csv', index=False)
from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(n_estimators=1000)



clf.fit(X_train,y_train)



predictions = clf.predict(X_test)



accuracy_score(y_test, predictions)
from sklearn.neighbors import KNeighborsClassifier



classifier = KNeighborsClassifier(n_neighbors=20)



classifier.fit(X_train, y_train)



predictions = classifier.predict(X_test)



accuracy_score(y_test, predictions)