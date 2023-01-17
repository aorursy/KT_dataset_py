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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(type(train))
train.head()
print(train.info)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
def pie_chart(feature):

    feature_ratio = train[feature].value_counts(sort=False)

    print(feature_ratio)

    feature_size = feature_ratio.size

    feature_index = feature_ratio.index

    survived = train[train['Survived'] == 1][feature].value_counts()

    dead = train[train['Survived'] == 0][feature].value_counts()

    

    plt.plot(aspect='auto')

    plt.pie(feature_ratio, labels = feature_index, autopct = '%1.1f%%')

    plt.title(feature + '\'s ratio in total')

    plt.show()

    

    for i, index in enumerate(feature_index):

        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')

        plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')

        plt.title(str(index) + '\'s ratio')

        

    plt.show
def bar_chart(feature):

    survived = train[train['Survived'] == 1][feature].value_counts()

    dead = train[train['Survived'] == 0][feature].value_counts()

    df = pd.DataFrame([survived, dead])

    df.index = ['Survived', 'Dead']

    df.plot(kind='bar', stacked=True, figsize=(10, 5))

    
pie_chart('Sex')
bar_chart('SibSp')
train_and_test = [train, test]

train.head(5)
for dataset in train_and_test:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

    

train.head(5)
pd.crosstab(train['Title'], train['Sex'])
for dataset in train_and_test:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
for dataset in train_and_test:

    dataset['Title'] = dataset['Title'].astype(str)

    dataset['Sex'] = dataset['Sex'].astype(str)
train.Embarked.value_counts(dropna=False)
for dataset in train_and_test:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].astype(str)
for dataset in train_and_test:

    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

    dataset['Age'] = dataset['Age'].astype(int)

    train['AgeBand'] = pd.cut(train['Age'], 5)



print(train[['AgeBand', 'Survived']].groupby(['AgeBand'],as_index=False).mean())



for dataset in train_and_test:

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    

    dataset['Age'] = dataset['Age'].map({0: 'Child', 1:'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)



train.head(5)
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print(test[test["Fare"].isnull()]["Pclass"])
for dataset in train_and_test:

    dataset['Fare'] = dataset['Fare'].fillna(13.675) # The only one empty fare data's pclass is 3.
for dataset in train_and_test:

    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3

    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4

    dataset['Fare'] = dataset['Fare'].astype(int)
for dataset in train_and_test:

    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]

    dataset['Family'] = dataset['Family'].astype(int)
# features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

# train = train.drop(features_drop, axis=1)

# test = test.drop(features_drop, axis=1)



train.head(5)
train = train.drop(['PassengerId', 'AgeBand'], axis=1)
train.head(5)
train = pd.get_dummies(train)

test = pd.get_dummies(test)



train_label = train['Survived']

train_data = train.drop('Survived', axis=1)

test_data = test.drop("PassengerId", axis=1).copy()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.utils import shuffle
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
def train_and_test(model):

    model.fit(train_data, train_label)

    prediction = model.predict(test_data)

    accuracy = round(model.score(train_data, train_label)*100, 2)

    print('Accuracy : ', accuracy, '%')

    return prediction
log_pred = train_and_test(LogisticRegression())

svm_pred = train_and_test(SVC())

rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
submission = pd.DataFrame({"PassengerId": test["PassengerId"], 

                          "Survived": rf_pred})



submission.to_csv('submission_rf.csv', index=False)