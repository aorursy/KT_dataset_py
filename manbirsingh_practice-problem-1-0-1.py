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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.info()

train_df.describe()
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df.info()
#Data processing

#1. In the cabin variable, create new column and add there only first letters of the column

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}

data = [train_df, test_df]



for dataset in data:

    dataset['Deck'] = dataset['Cabin'].fillna("U")

    dataset['Deck'] = dataset['Cabin'].astype(str).str[0] 

    dataset['Deck'] = dataset['Deck'].str.capitalize()

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int) 



train_df['Deck'].value_counts()

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
data = [train_df, test_df]



for dataset in data:

    mean = train_df["Age"].mean()

    # fill NaN values in Age column with random values generated

    dataset["Age"] = train_df["Age"].fillna(mean)

train_df["Age"].isnull().sum()
dataset["Age"].value_counts()
train_df['Embarked'].value_counts()



common_value = 'S'



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)



train_df['Embarked'].describe()

train_df.info()
data = [train_df, test_df] 

embarkedMap = {"S": 0, "C": 1, "Q": 2}

genderMap = {"male": 0, "female": 1}

for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int) 

    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)

    dataset['Sex'] = dataset['Sex'].map(genderMap)

    #print (dataset['Embarked'])

    

train_df['Sex'].describe()

train_df['Embarked'].describe()

train_df.info()
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 12, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 24), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 30), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 35), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 65), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 65, 'Age'] = 6



# let's see how it's distributed 

train_df['Age'].value_counts()
data = [train_df, test_df]



#train_df['category_fare'] = pd.qcut(train_df['Fare'], 4)



#train_df['category_fare'].value_counts()



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df['Fare'].value_counts()
train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)
train_df.info()
train_df.drop("Ticket", axis = 1, inplace = True)

test_df.drop("Ticket", axis = 1, inplace = True)
train_df.drop("PassengerId", axis = 1, inplace = True)

#test_df.drop("Ticket", axis = 1, inplace = True)
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']



test_df.head(10)
X_test = test_df.drop("PassengerId", axis=1).copy()

X_test.head(10)
# Decision tree

from sklearn.tree import DecisionTreeClassifier

clf_decision_tree = DecisionTreeClassifier(random_state=0)

                                           

clf_decision_tree.fit(X_train, Y_train)



Y_pred = clf_decision_tree.predict(X_test)

score = clf_decision_tree.score(X_train, Y_train)



print (score)    
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")