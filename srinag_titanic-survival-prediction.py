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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train.head()
clean_up = { 'Sex': {'male': 0, 'female': 1} }



train.replace(clean_up, inplace=True)

train.head()
train.describe()
age_avg = train.Age.mean()

train.Age = train.Age.fillna(int(age_avg))

train.Age = train.Age.astype('int')
train = train.dropna(axis=0, subset=['Embarked'])

train
y = train.Survived

sel_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

x1 = pd.get_dummies(train[sel_cols])

x1
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test.replace(clean_up, inplace=True)

test_age_avg = test.Age.mean()

test.Age = test.Age.fillna(int(test_age_avg))

test.Age = test.Age.astype('int')

test_Fare_avg = test.Fare.mean()

test.Fare = test.Fare.fillna(test_Fare_avg) 

t1 = pd.get_dummies(test[sel_cols]) 
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



class_model = {

    'logisticRegression' : LogisticRegression(penalty='l2', C=50, random_state=1, max_iter=5000)

}



for model_name, model_class in class_model.items():

    model_class.fit(x1, y)

    y_pred = model_class.predict(t1)

    #print(model_name, ' accuracy_score:',metrics.accuracy_score(y_pred, y))3



output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y_pred})

output.to_csv("final.csv", index=False)
