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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
X_train = train_data.drop(labels=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

y_train = train_data.Survived



X_train.head()
X_values = X_train.values
"""

Missing values

"""

# fill age missing values with mean

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_values[:, 2:3] = imputer.fit_transform(X_values[:, 2:3])



# fill Embarked missing values with most frequent value

imputer = SimpleImputer(strategy='most_frequent')

X_values[:, 6:7] = imputer.fit_transform(X_values[:, 6:7])
# new DataFrames with all values filled

df = pd.DataFrame(data=X_values, columns=list(X_train.columns))

df.info()
"""

Categorical features

"""

df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

df['Embarked'] = df['Embarked'].map({'S': 2, 'Q': 1, 'C': 0})
# convert types

for col in df.columns:

    df[col] = df[col].astype('float64')



df_train = pd.DataFrame(data=df, columns=list(X_train.columns))

df_train.info()
df_train.head()
"""

Features standard scaling

"""

X_values = df_train.values

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_values = sc.fit_transform(X_values)
"""

Train model

"""

from sklearn.ensemble import RandomForestClassifier

cls = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

cls.fit(X_values, y_train)
"""

cross validation score

"""

from sklearn.model_selection import cross_val_score

import numpy as np

scores = cross_val_score(cls, X_values, y_train, cv=10)

print("mean of scores : ", np.mean(scores))
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_test.head()
"""

Missing values

"""

X_test_values = X_test.values

# fill age missing values with mean

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_test_values[:, 2:3] = imputer.fit_transform(X_test_values[:, 2:3])

imputer = SimpleImputer(strategy='mean')

X_test_values[:, 5:6] = imputer.fit_transform(X_test_values[:, 5:6])

df_test = pd.DataFrame(data=X_test_values, columns=X_test.columns)
df_test['Sex'] = df_test['Sex'].map({'female': 1, 'male': 0})

df_test['Embarked'] = df_test['Embarked'].map({'S': 2, 'Q': 1, 'C': 0})



for col in df_test.columns:

    df_test[col] = df_test[col].astype('float64')



df_test = pd.DataFrame(data=df_test, columns=list(X_test.columns))
df_test.info()
from sklearn.preprocessing import StandardScaler

X_test_values = df_test.values

sc = StandardScaler()

X_test_values = sc.fit_transform(X_test_values)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': cls.predict(X_test_values)})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")