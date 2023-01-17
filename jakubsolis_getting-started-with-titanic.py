# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading the data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
#% of female passengers who survived

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)

#% of male passengers who survived

men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
def chart(feature):

    survived = train_data[train_data['Survived']==1][feature].value_counts()

    dead = train_data[train_data['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked='True')
chart('Sex')
chart('Pclass')
chart('SibSp')
chart('Parch')
chart('Embarked')
train_data.isnull().sum()
test_data.isnull().sum()
train_data.head(90)
title_age ={"Mr": 0, "Miss": 1, "Mr": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mille":3 , "Countess": 3, "Ms": 3, "Lady":3, "Jonkheer": 3,

            "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3 }

for dataset in train_data:

    dataset['Title']=dataset['Title'].map(title_age)
#Random forest model

from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")