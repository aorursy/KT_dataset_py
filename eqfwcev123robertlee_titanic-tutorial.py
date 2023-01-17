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

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

women = train_data[train_data.Sex == "female"]["Survived"]
# type(women) => Series
rate_women = sum(women) / len(women)
print(f"% of women who survied: {rate_women}")
men = train_data[train_data.Sex == "male"]["Survived"]
rate_men = sum(men)/len(men)
print(f"% of men who survied: {rate_men}")
from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]

features = ["Pclass","Sex","SibSp","Parch"]

x = pd.get_dummies(train_data[features])
x_test =  pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x, y)
predictions = model.predict(x_test)

output = pd.DataFrame({"PassengerId":test_data.PassengerId, "Survived":predictions})
output.to_csv("my_submission.csv", index=False)
print("Your submission was successfully saved!")
