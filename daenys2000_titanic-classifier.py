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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
y = train_data['Survived']

features = ['Sex','Pclass','SibSp', 'Parch']

x_train = pd.get_dummies(train_data[features])

x_test = pd.get_dummies(test_data[features])

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators= 200)

classifier.fit(x_train, y)
y_test = classifier.predict(x_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")