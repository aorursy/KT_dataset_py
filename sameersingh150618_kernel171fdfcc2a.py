# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

test_dataset = pd.read_csv('/kaggle/input/titanic/test.csv')

X_train = train_dataset.iloc[:, [False,False,True,False,True,True,True,True,False,True,False,False]].values

y_train = train_dataset.iloc[:, 1].values

X_test = test_dataset.iloc[:, [False,True,False,True,True,True,True,False,True,False,False]].values
print(X_train)
print(X_test)
print(y_train)
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')

X_train = np.array(ct.fit_transform(X_train))

X_test = np.array(ct.fit_transform(X_test))
print(X_train)
print(X_test)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(X_train)

X_train = imputer.transform(X_train)

imputer.fit(X_test)

X_test = imputer.transform(X_test)
print(X_train)
print(X_test)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
for i in range(len(X_train)):

  if X_train[i][0]>X_train[i][1]:

    X_train[i][0] = 1

    X_train[i][1] = 0

  else:

    X_train[i][0] = 0

    X_train[i][1] = 1

print(X_train)
for i in range(len(X_test)):

  if X_test[i][0]>X_test[i][1]:

    X_test[i][0] = 1

    X_test[i][1] = 0

  else:

    X_test[i][0] = 0

    X_test[i][1] = 1

print(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)

classifier.fit(X_train, y_train)
y_test = classifier.predict(X_test)

print(y_test)
output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': y_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")