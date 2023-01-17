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
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

# 元々のロジックでのテスト用にデータ分割
from sklearn.model_selection import train_test_split
t_data, check_data = train_test_split(train_data, random_state=0)
from sklearn.model_selection import train_test_split
t_data, check_data = train_test_split(train_data, random_state=0)

from sklearn.ensemble import RandomForestClassifier

y = t_data["Survived"]
y_test = check_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(t_data[features])
X_test = pd.get_dummies(check_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X)
predictions_test = model.predict(X_test)
# 評価
from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import confusion_matrix

print('Train score: {}'.format(model.score(X, y)))
print('Test score: {}'.format(model.score(X_test, y_test)))

print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, predictions_test)))

print('f1 score: {:.3f}'.format(f1_score(y_test, predictions_test)))

print('train log_loss: {:.3f}'.format(log_loss(y, model.predict_proba(X))))
print('test log_loss: {:.3f}'.format(log_loss(y_test, model.predict_proba(X_test))))
# 特徴量追加版用のデータ分割
from sklearn.model_selection import train_test_split
t_data, check_data = train_test_split(train_data, random_state=0)


t_data = modifyFeature(t_data)
check_data = modifyFeature(check_data)

y = t_data["Survived"]
y_test = check_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", 'Embarked']
X = pd.get_dummies(t_data[features])
X_test = pd.get_dummies(check_data[features])

X['Fare'] = t_data["Fare"]
X['Age'] = t_data["Age"]

X_test['Fare'] = check_data["Fare"]
X_test['Age'] = check_data["Age"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

predictions = model.predict(X)
predictions_test = model.predict(X_test)
# 評価
from sklearn.metrics import f1_score, log_loss
from sklearn.metrics import confusion_matrix

print('Train score: {}'.format(model.score(X, y)))
print('Test score: {}'.format(model.score(X_test, y_test)))

print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, predictions_test)))

print('f1 score: {:.3f}'.format(f1_score(y_test, predictions_test)))

print('train log_loss: {:.3f}'.format(log_loss(y, model.predict_proba(X))))
print('test log_loss: {:.3f}'.format(log_loss(y_test, model.predict_proba(X_test))))

# だいぶ良くなっている。loglossが　0.423　=> 0.397


train_data = modifyFeature(train_data)
test_data = modifyFeature(test_data)

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", 'Embarked']
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

X['Fare'] = train_data["Fare"]
X['Age'] = train_data["Age"]

X_test['Fare'] = test_data["Fare"]
X_test['Age'] = test_data["Age"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

predictions = model.predict(X)
predictions_test = model.predict(X_test)
# 新しいロジックでcsvを出力
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_test})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
