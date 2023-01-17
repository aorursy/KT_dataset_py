# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
titanic = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

test_df = pd.read_csv("../input/titanic/test.csv")



titanic.head()
titanic.info()

print('___________________________')

test.info()
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test = test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)



titanic.drop(['Cabin'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True)
embarked = titanic['Embarked']

print(embarked.isnull().sum())

embarked.value_counts()
titanic['Embarked'] = titanic['Embarked'].fillna("S")



sns.factorplot('Embarked','Survived', data=titanic,size=4,aspect=3)
embark_perc = titanic[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'])
titanic.drop(['Embarked'], axis=1,inplace=True)

test.drop(['Embarked'], axis=1,inplace=True)     # doesn't seem to be a good predictor of survival 
average_age_titanic   = titanic["Age"].mean()     # average age

std_age_titanic       = titanic["Age"].std()     # std age

count_nan_age_titanic = titanic["Age"].isnull().sum()   # number of missing values 



print('Average age is: ' + str(average_age_titanic), 'Std is: ' + str(std_age_titanic), 'Missing values: ' + str(count_nan_age_titanic))
average_age_test   = test["Age"].mean()     # average age

std_age_test       = test["Age"].std()     # std age

count_nan_age_test = test["Age"].isnull().sum()   # number of missing values 



print('Average age is: ' + str(average_age_test), 'Std is: ' + str(std_age_test), 'Missing values: ' + str(count_nan_age_test))
# generate random numbers between (mean - std) & (mean + std) and size is equal to count missing values 



rand1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic,

                         size = count_nan_age_titanic)

rand2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, 

                         size = count_nan_age_test)



# replacing missing values with random ints



titanic['Age'][np.isnan(titanic["Age"])] = rand1

test['Age'][np.isnan(test["Age"])] = rand2



# converting to all int types 



titanic['Age'] = titanic['Age'].astype(int)

test['Age'] = test['Age'].astype(int)



titanic['Age'].hist(bins=70)
sns.factorplot('Pclass', 'Survived', data=titanic)



pclass_dummies_titanic  = pd.get_dummies(titanic['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



titanic.drop(['Pclass'],axis=1,inplace=True)

test.drop(['Pclass'],axis=1,inplace=True)



titanic = titanic.join(pclass_dummies_titanic)

test = test.join(pclass_dummies_test)

# sex



def getperson(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex 



titanic['Person'] = titanic[['Age', 'Sex']].apply(getperson, axis=1)

test['Person'] = titanic[['Age', 'Sex']].apply(getperson, axis=1)



titanic.drop(['Sex'], axis=1, inplace=True)

test.drop(['Sex'], axis=1, inplace=True)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Person', data=titanic, ax=axis1)



# average of survived for each Person(male, female, or child)

person_perc = titanic[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])





person_dummies_titanic = pd.get_dummies(titanic['Person'])

person_dummies_titanic.columns = ['Child', 'Female', 'Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test = pd.get_dummies(titanic['Person'])

person_dummies_test.columns = ['Child', 'Female', 'Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



titanic.drop(['Person'],axis=1,inplace=True)

test.drop(['Person'],axis=1,inplace=True)



titanic = titanic.join(person_dummies_titanic)

test = test.join(person_dummies_test)
titanic['Family'] =  titanic["Parch"] + titanic["SibSp"]

titanic['Family'].loc[titanic['Family'] > 0] = 1

titanic['Family'].loc[titanic['Family'] == 0] = 0



test['Family'] = test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1

test['Family'].loc[test['Family'] == 0] = 0



titanic = titanic.drop(['SibSp','Parch'], axis=1)

test = test.drop(['SibSp','Parch'], axis=1)



fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



sns.countplot(x='Family', data=titanic, order=[1,0], ax=axis1)



family_perc = titanic[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
test["Fare"].fillna(test["Fare"].median(), inplace=True)



titanic['Fare'] = titanic['Fare'].astype(int)

test['Fare']    = test['Fare'].astype(int)



fare_not_survived = titanic["Fare"][titanic["Survived"] == 0]

fare_survived     = titanic["Fare"][titanic["Survived"] == 1]



avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])


X_train = titanic.drop("Survived",axis=1)

Y_train = titanic["Survived"]

X_test  = test.copy()





sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)