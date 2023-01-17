# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
print("Dimensions of train: {}".format(train_data.shape))

print("Dimensions of test: {}".format(test_data.shape))
train_data.info()
missing_values_perc = train_data.isnull().sum()/train_data.isnull().count()*100

print(missing_values_perc)

missing_values_per = test_data.isnull().sum()/test_data.isnull().count()*100

print(missing_values_per)
print(train_data['Sex'].describe())

sexplot = train_data.pivot_table(index="Sex",values="Survived")

sexplot.plot.bar()

plt.show()
print(train_data['Pclass'].describe())

classplot = train_data.pivot_table(index="Pclass",values="Survived")

classplot.plot.bar()

plt.show()
data = [train_data, test_data]

for dataset in data:

    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['Relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['Relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

train_data['not_alone'].value_counts()

relativesplot = sns.factorplot('Relatives','Survived', data=train_data, aspect = 2.5, )
train_data = train_data.drop(['Cabin'], axis=1)

test_data = test_data.drop(['Cabin'], axis=1)

train_data = train_data.drop(['Name'], axis=1)

test_data = test_data.drop(['Name'], axis=1)

train_data = train_data.drop(['Ticket'], axis=1)

test_data = test_data.drop(['Ticket'], axis=1)

train_data = train_data.drop(['PassengerId'], axis=1)

train_data.head()
train_data['Age'].describe()
survived = train_data[train_data["Survived"] == 1]

died = train_data[train_data["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=30)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=30)

plt.legend(['Survived','Died'])

plt.show()
mean_value = 29

data = [train_data, test_data]



for dataset in data:

    dataset['Age'] = dataset['Age'].fillna(mean_value)



print(train_data['Age'].isnull().sum())

print(test_data['Age'].isnull().sum())
data = [train_data, test_data]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6



train_data['Age'].value_counts()
train_data['Embarked'].describe()
top_value = 'S'

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(top_value)

    

print(train_data['Embarked'].isnull().sum())

print(test_data['Embarked'].isnull().sum())
ports = {"S": 0, "C": 1, "Q": 2}

data = [train_data, test_data]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
print(train_data.head(10))

print(train_data['Fare'].describe())
mean_value = 32

data = [train_data, test_data]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(mean_value)



print(train_data['Fare'].isnull().sum())

print(test_data['Fare'].isnull().sum())
data = [train_data, test_data]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

train_data['Fare'].value_counts()
genders = {"male": 0, "female": 1}

data = [train_data, test_data]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train_data.head()
test_data.head()
X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()



decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

 

decision_tree.score(X_train, Y_train)
prediction = decision_tree.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was saved.")