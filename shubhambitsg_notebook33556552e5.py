import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
titanic_df.info()
test_df = pd.read_csv('../input/test.csv')

test_df.head()
titanic_df= titanic_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test_df = test_df.drop(['Name', 'Ticket'], axis=1)
test_df.info()
titanic_df['Embarked']= titanic_df['Embarked'].fillna('S')
embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])

embark_dummies_test = pd.get_dummies(test_df['Embarked'])



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df = test_df.join(embark_dummies_test)



titanic_df.drop('Embarked', axis=1, inplace=True)

test_df.drop('Embarked', axis=1, inplace=True)
titanic_df.head()
test_df.head()
titanic_df['Fare'] = titanic_df['Fare'].astype(int)



test_df['Fare'].fillna(test_df['Fare'].median(), inplace= True)

test_df['Fare'] = test_df['Fare'].astype(int)



average_age_titanic   = titanic_df["Age"].mean()

std_age_titanic       = titanic_df["Age"].std()

count_nan_age_titanic = titanic_df["Age"].isnull().sum()



average_age_test   = test_df["Age"].mean()

std_age_test       = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
titanic_df.head()
titanic_df.drop(['Cabin'], axis=1, inplace=True)

test_df.drop(['Cabin'], axis=1, inplace=True)
titanic_df.head()
test_df.head()
titanic_df['Family'] = titanic_df['SibSp'] + titanic_df['Parch']

titanic_df['Family'].loc[titanic_df['Family'] >0] =1

titanic_df['Family'].loc[titanic_df['Family'] ==0] =0



test_df['Family'] = test_df['SibSp'] + test_df['Parch']

test_df['Family'].loc[test_df['Family'] >0] =1

test_df['Family'].loc[test_df['Family'] ==0] =0



titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

    

titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)



titanic_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']



person_dummies_test  = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child','Female','Male']
titanic_df = titanic_df.join(person_dummies_titanic)

test_df =  test_df.join(person_dummies_test)
titanic_df.head()
titanic_df.drop('Person', axis=1, inplace= True)

test_df.drop('Person', axis=1, inplace= True)
titanic_df.head()
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df    = test_df.join(pclass_dummies_test)
X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]
X_train.head()
X_train.info()
titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1

test_df["Age"][np.isnan(test_df["Age"])] = rand_2
X_train.head()
X_train.info()
X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]
X_train.info()
# Logistic Regression



logreg = LogisticRegression()



logreg.fit(X_train, Y_train)
logreg.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
X_test.info()
X_test  = test_df.drop("PassengerId",axis=1).copy()
Y_pred = logreg.predict(X_test)
Y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)