# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#### Load Data

import pandas as pd

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

#### Fill N/As

train.Age = train.Age.fillna(train.Age.mean())

train.Fare = train.Fare.fillna(train.Fare.mean())

test.Age = train.Age.fillna(test.Age.mean())

test.Fare = train.Fare.fillna(test.Fare.mean())



#### Split

from sklearn.model_selection import train_test_split



x_all = train[['Age', 'Fare']]

y_all = train['Survived']



x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.20, random_state=23)



#### model

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(x_train, y_train)



#### prediction

from sklearn.metrics import accuracy_score

predictions = clf.predict(x_test)

print(accuracy_score(y_test, predictions))



#### prediction for test data

x_real_test = test[['Age', 'Fare']]

real_pred = clf.predict(x_real_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": real_pred

    })

# submission.to_csv('../output/submission.csv', index=False)