# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

# from sklearn.preprocessing import LabelEncoder

from pandas.api.types import CategoricalDtype

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.head()
y = train_data['Survived']

x_pd = train_data.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'])



x_pd['Age'] = x_pd['Age'].fillna( x_pd['Age'].median()).round(0).astype(int)



cat_sex = CategoricalDtype(categories=['male','female'],ordered=False)

x_pd['Sex'] = x_pd['Sex'].astype(cat_sex)

Sex_int = pd.get_dummies(x_pd['Sex'])

x_pd =pd.concat([x_pd, Sex_int], axis=1)



cat_embarked = CategoricalDtype(categories=['S', 'C', 'Q'],ordered=False)

x_pd['Embarked'] = x_pd['Embarked'].fillna('S')

x_pd['Embarked'] = x_pd['Embarked'].astype(cat_embarked)

Embarked_int = pd.get_dummies(x_pd['Embarked'])

x_pd =pd.concat([x_pd, Embarked_int], axis=1)



# x_pd.head()

# x_pd.dtypes

from sklearn import preprocessing



X = x_pd.drop(columns=['Sex', 'Embarked'])

# x = X.values #returns a numpy array

# min_max_scaler = preprocessing.MinMaxScaler()

# x_scaled = min_max_scaler.fit_transform(x)

# X = pd.DataFrame(x_scaled)

X.describe()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

svclassifier = svm.SVC(kernel='linear')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

x_pd = test_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])



x_pd['Age'] = x_pd['Age'].fillna( x_pd['Age'].median()).round(0).astype(int)



cat_sex = CategoricalDtype(categories=['male','female'],ordered=False)

x_pd['Sex'] = x_pd['Sex'].astype(cat_sex)

Sex_int = pd.get_dummies(x_pd['Sex'])

x_pd =pd.concat([x_pd, Sex_int], axis=1)



cat_embarked = CategoricalDtype(categories=['S', 'C', 'Q'],ordered=False)

x_pd['Embarked'] = x_pd['Embarked'].fillna('S')

x_pd['Embarked'] = x_pd['Embarked'].astype(cat_embarked)

Embarked_int = pd.get_dummies(x_pd['Embarked'])

x_pd =pd.concat([x_pd, Embarked_int], axis=1).drop(columns=['Sex', 'Embarked'])



x_pd['Fare'] = x_pd['Fare'].fillna(x_pd['Fare'].median())





y_test = svclassifier.predict(x_pd)

submission = pd.DataFrame({ 'PassengerId': test_data['PassengerId'],

                            'Survived': y_test })

submission.to_csv("submission.csv", index=False)
