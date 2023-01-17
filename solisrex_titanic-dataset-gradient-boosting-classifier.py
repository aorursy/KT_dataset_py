# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train_csv = pd.read_csv("/kaggle/input/titanic/train.csv")

test_csv = pd.read_csv("/kaggle/input/titanic/test.csv")



X_train = train_csv

y_train = train_csv.Survived

del X_train["Survived"]

X_test = test_csv

train_len, _ = X_train.shape

X_total = X_train.append(X_test)

X_total.Name = [name.split(",")[0] for name in X_total.Name]

X_total.fillna(-1, inplace=True)

X_total_dummy = pd.get_dummies(X_total)



scaler = StandardScaler()

scaler.fit(X_total_dummy)

X_total_dummy = scaler.transform(X_total_dummy)

#Create a model, fit it to the training data, print its accuracy,

#and predict the categories of the test data.

gbrt = GradientBoostingClassifier(random_state=0)

gbrt.fit(X_total_dummy[:train_len],y_train)

print("Accuracy on training set: {}".format(gbrt.score(X_total_dummy[:train_len],y_train)))

predicted = gbrt.predict(X_total_dummy[train_len:])

my_submission = pd.DataFrame({'PassengerId': test_csv.PassengerId, 'Survived': predicted})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.