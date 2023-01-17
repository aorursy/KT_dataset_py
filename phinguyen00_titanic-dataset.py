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

train_data.head()

train_data.columns
train_data.describe()
print(train_data.isnull().sum())

# Null values in our data
# Let's try and fix the null (NaN) values in Cabin

train_data.Cabin = train_data.Cabin.fillna("unknown")

print(train_data.isnull().sum())
# Let's see the data, and the types

print(train_data.shape)

print()

print(train_data.dtypes)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
import matplotlib.pyplot as plt



train_data.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("Survived")
# Sex vs Survived

women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
# Class vs Survived

class1 = train_data.loc[train_data.Pclass == 1]['Survived']

rate_class1 = sum(class1)/len(class1)



print("% of 1st class who survived:", rate_class1)



class2 = train_data.loc[train_data.Pclass == 2]['Survived']

rate_class2 = sum(class2)/len(class2)



print("% of 2nd class who survived:", rate_class2)



class3 = train_data.loc[train_data.Pclass == 3]['Survived']

rate_class3 = sum(class3)/len(class3)



print("% of 3rd class who survived:", rate_class3)

# Children that survived

train_data[train_data.Age < 18].Survived.value_counts().plot(kind='barh')
# female children

train_data[(train_data.Age < 18) & (train_data.Sex == 'female')].Survived.value_counts().plot(kind = 'barh')
# male children

train_data[(train_data.Age < 18) & (train_data.Sex == 'male')].Survived.value_counts().plot(kind = 'barh')
col_target = ['Survived']

col_train = ['Age', 'Pclass', 'Sex']



X = train_data[col_train]

y = train_data[col_target]
print(X['Sex'].isnull().sum())

print(X['Pclass'].isnull().sum())

print(X['Age'].isnull().sum())
X['Age'] = X['Age'].fillna(X['Age'].mean())
X['Age'].isnull().sum()
dic = {'male': 0, 'female': 1}

X['Sex'] = X['Sex'].apply(lambda x:dic[x])

X['Sex'].head()
X.head()
# First split the training data

from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.3, random_state = 12)
from sklearn import svm



clf1 = svm.LinearSVC()



clf1.fit(X_train, Y_train)

print(clf1.score(X_test, Y_test))

print()

print(clf1)

train_data['Child'] = float('NaN')

train_data['AgeClass'] = float('NaN')



train_data['AgeClass'][train_data['Pclass'] == 1] = 1*train_data['Age']

train_data['AgeClass'][train_data['Pclass'] == 2] = 2*train_data['Age']

train_data['AgeClass'][train_data['Pclass'] == 3] = 3*train_data['Age']



train_data["Child"][train_data["Age"] < 18] = 1

train_data["Child"][train_data["Age"] >= 18] = 0

train_data[train_data.Child == 1]
print(train_data.columns.values)
train_data.isnull().sum()
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())
train_data["Sex"][train_data["Sex"] == "male"] = 0

train_data["Sex"][train_data["Sex"] == "female"] = 1



# Replace the null Embarked variable

train_data["Embarked"] = train_data["Embarked"].fillna(train_data['Embarked'].mode())



# Convert the Embarked classes to integer

train_data["Embarked"][train_data["Embarked"] == "S"] = 0

train_data["Embarked"][train_data["Embarked"] == "C"] = 1

train_data["Embarked"][train_data["Embarked"] == "Q"] = 2



train_data.isnull().sum()

train_data["AgeClass"] = train_data["AgeClass"].fillna(train_data['AgeClass'].median())
train_data.head()
from sklearn import tree



train_data["Embarked"] = train_data["Embarked"].fillna(0)

train_data["Child"] = train_data["Child"].fillna(0)



target = train_data["Survived"].values

features1 = train_data[["Pclass", "Sex", "Age", "Fare","Child", "SibSp","Embarked","AgeClass"]].values



dec_tree = tree.DecisionTreeClassifier()

dec_tree = dec_tree.fit(features1, target)



print(dec_tree.score(features1, target))

## Now we can predict

train_data.isnull().sum()
test_data.isnull().sum()

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())

test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())



test_data["Sex"][test_data["Sex"] == "male"] = 0

test_data["Sex"][test_data["Sex"] == "female"] = 1



test_data['AgeClass'] = float('NaN')



test_data['AgeClass'][test_data['Pclass'] == 1] = 1*test_data['Age']

test_data['AgeClass'][test_data['Pclass'] == 2] = 2*test_data['Age']

test_data['AgeClass'][test_data['Pclass'] == 3] = 3*test_data['Age']



test_data.isnull().sum()

test_data["AgeClass"] = test_data["AgeClass"].fillna(test_data['AgeClass'].median())

test_data['Child'] = float('NaN')

test_data["Child"][test_data["Age"] < 18] = 1

test_data["Child"][test_data["Age"] >= 18] = 0



# Convert the Embarked classes to integer

test_data["Embarked"][test_data["Embarked"] == "S"] = 0

test_data["Embarked"][test_data["Embarked"] == "C"] = 1

test_data["Embarked"][test_data["Embarked"] == "Q"] = 2



test_data.isnull().sum()
testing_features = test_data[["Pclass", "Sex", "Age", "Fare","Child", "SibSp","Embarked","AgeClass"]].values

features2 = train_data[["Pclass", "Sex","Fare","Child", "SibSp","Embarked","AgeClass"]].values

testing_features2 = test_data[["Pclass", "Sex","Fare","Child", "SibSp","Embarked","AgeClass"]].values





pred = dec_tree.predict(testing_features)

print(dec_tree.score(features1, target))

print(pred)



PassengerId = np.array(test_data["PassengerId"]).astype(int)

solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])

print(solution)



print(solution.shape)



solution.to_csv("solution2.csv", index_label = ["PassengerId"])
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(features1, target)



print(logreg.score(features1, target))



pred2 = logreg.predict(testing_features)

print(pred2)



PassengerId2 = np.array(test_data["PassengerId"]).astype(int)

solution2 = pd.DataFrame(pred2, PassengerId2, columns = ["Survived"])

print(solution2)



print(solution2.shape)



solution2.to_csv("solution3.csv", index_label = ["PassengerId"])
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(features1, target)

pred3 = random_forest.predict(testing_features)



print(random_forest.score(features1, target))



PassengerId3 = np.array(test_data["PassengerId"]).astype(int)

solution3 = pd.DataFrame(pred3, PassengerId3, columns = ["Survived"])

print(solution3)



print(solution3.shape)



solution3.to_csv("solution4.csv", index_label = ["PassengerId"])
new_tree = tree.DecisionTreeClassifier(random_state = 1, max_depth = 7, min_samples_split = 2)

new_tree.fit(features1, target)



print(new_tree.score(features1, target))



pred4 = new_tree.predict(testing_features)



PassengerId4 = np.array(test_data["PassengerId"]).astype(int)

solution4 = pd.DataFrame(pred4, PassengerId4, columns = ["Survived"])

print(solution4)



print(solution4.shape)



solution4.to_csv("solution5.csv", index_label = ["PassengerId"])