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
train_set = pd.read_csv('/kaggle/input/titanic/train.csv')

test_set = pd.read_csv('/kaggle/input/titanic/test.csv')
train_set.head()

#test_set.head()

y = train_set["Survived"]

test_set.head()
features = ['Fare','Age']

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression as lg

trains_set= pd.get_dummies(train_set[features])

tests_set = pd.get_dummies(test_set[features]).fillna(test_set[features].mean())

trains_set = trains_set.fillna(train_set[features].mean())



X_Poly = MinMaxScaler().fit_transform(trains_set)

test_Poly = MinMaxScaler().fit_transform(tests_set)



reg= lg().fit(X_Poly,y)

predict = reg.predict(test_Poly)



output = pd.DataFrame({'Passengerid': test_set.PassengerId, 'Survived': predict})

output.to_csv('my_newsubmission.csv',index=False)

print("Your submission was successfully saved!")
