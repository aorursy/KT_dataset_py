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

train_data = train_data.dropna()

# convert sex, male = 1, female = 0

def sex_convert(x):

    if x == 'female':

        return 0

    else:

        return 1

    

# convert Embarked, Q = 1, S = 2, C = 3

def sex_convert(x):

    if x == 'Q':

        return 1

    elif x == "S":

        return 2

    else:

        return 3



    

train_data['Sex'] = train_data['Sex'].map(sex_convert)

train_data['Embarked'] = train_data['Embarked'].map(sex_convert)

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data = test_data.fillna(0)

test_data['Sex'] = test_data['Sex'].map(sex_convert)

test_data['Embarked'] = test_data['Embarked'].map(sex_convert)

test_data
from pandas import Series, DataFrame

import pandas as pd

from patsy import dmatrices

%pylab inline
train_data['target'] = 0.0

train_data['target'][train_data['Survived'] > 0] = 1.0

train_data['target'].value_counts()
formula = 'target ~ 0 + Pclass + Age + SibSp + Parch + Fare + C(Embarked) + C(Sex)'
Y_train, X_train = dmatrices(formula, train_data, return_type='dataframe')

y_train = Y_train['target'].values

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

result = model.fit(X_train, y_train)
from sklearn import metrics



prediction_train = model.predict(X_train)

print (metrics.accuracy_score(y_train, prediction_train))
test_data = test_data.drop(['Name', 'Ticket','Cabin'], axis=1)

test_data.head()
sol_prediction = result.predict(test_data)
sol_ids = test_data["PassengerId"]

submission_df = {"PassengerId": sol_ids,

                 "Survived": sol_prediction}

submission = pd.DataFrame(submission_df)
submission.to_csv("sol_output.csv", index = False)
