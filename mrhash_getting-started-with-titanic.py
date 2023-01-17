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

train_data.describe()
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.info()
women = train_data.loc[train_data.Sex == "female"]["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == "male"]["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
train_data.columns[train_data.isnull().any()]
test_data.columns[test_data.isnull().any()]
train_data['Age'].fillna((train_data['Age'].mean()), inplace=True)

test_data['Age'].fillna((test_data['Age'].mean()), inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



y = train_data["Survived"]



features = ["Sex", "Age", "Pclass", "SibSp", "Parch"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features])

X = train_data[features].values

X_test = test_data[features].values

# print(X)

labelEncoder_gender =  LabelEncoder()

X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

X = np.vstack(X[:, :]).astype(np.float)

# print(X)



labelEncoder_gender_test =  LabelEncoder()

X_test[:,0] = labelEncoder_gender_test.fit_transform(X_test[:,0])

X_test = np.vstack(X_test[:, :]).astype(np.float)



model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")