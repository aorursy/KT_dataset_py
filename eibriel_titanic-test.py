# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/train.csv"))
train_file = '../input/train.csv'

nan_values = {'Survived': 0, 'Pclass': 1, 'Sex': 0, 'Age': 30, 'Fare': 10}

# Any results you write to the current directory are saved as output.
full_data = pd.read_csv(train_file, usecols=["Survived", "Pclass", "Sex", "Age", "Fare"])
full_data = full_data.fillna(value=nan_values)
display(full_data.head())
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train = []
y_train = []
for _, passenger in full_data.iterrows():
    sex = 0 if passenger.Sex == "male" else 1
    X_train.append([passenger.Pclass, passenger.Age, sex, passenger.Fare])
    y_train.append(passenger.Survived)
    
# clf = DecisionTreeClassifier(max_depth=5)
clf = KNeighborsClassifier(3)
clf.fit(X_train, y_train)


test_file = '../input/test.csv'

# Any results you write to the current directory are saved as output.
test_data = pd.read_csv(test_file)
test_data = test_data.fillna(value=nan_values)
display(test_data.head())
X_test = []
y_test = []
pass_id = []
for _, passenger in test_data.iterrows():
    sex = 0 if passenger.Sex == "male" else 1
    X_test.append([passenger.Pclass, passenger.Age, sex, passenger.Fare])
    pass_id.append(passenger.PassengerId)

y_test = clf.predict(X_test)
y_expanded = np.expand_dims(y_test, axis=1)
pass_expanded = np.expand_dims(np.array(pass_id), axis=1)
out_data = np.hstack((pass_expanded, y_expanded))
print(out_data)

result = pd.DataFrame(out_data, columns=['PassengerId', 'Survived'])
display(result.head())
result.to_csv('submission.csv', index=False)
