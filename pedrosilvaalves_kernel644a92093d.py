import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)))
train_data_raw = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data_raw = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data = train_data_raw.fillna(0)

test_data = test_data_raw.fillna(0)

features = ["Pclass", "Age", "SibSp", "Parch","Fare"]

X = pd.get_dummies(train_data[features])

y = train_data["Survived"]



X['Sex'] = train_data['Sex'].map( {'male': 0, 'female' : 1}).astype(int)

X['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 0:-1} ).astype(int)

X
X_test = pd.get_dummies(test_data[features])

X_test['Sex'] = test_data['Sex'].map( {'male': 0, 'female' : 1}).astype(int)

X_test['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

X_test
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)

mode = model.fit(X,y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")