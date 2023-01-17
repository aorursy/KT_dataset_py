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
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head(10)
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head(10)
women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
train['Age'] = train['Age'].fillna(round(train['Age'].mean(), 1))
test['Age'] = test['Age'].fillna(round(test['Age'].mean(), 1))
train['Fare'] = train['Fare'].fillna(round(train['Fare'].mean(), 4))
test['Fare'] = test['Fare'].fillna(round(test['Fare'].mean(), 4))
y = train["Survived"]
features = ["Pclass","Sex", "SibSp", "Parch", "Age"]
X = pd.get_dummies(train[features])
finalX_test = pd.get_dummies(test[features])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)
val = model.predict(X_test)



X_train
X_test
y_train
y_test
len(val)
sum(abs(val-y_test))/len(y_test)
predictions = model.predict(finalX_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print(output.head(10))
print("Your submission was successfully saved!")