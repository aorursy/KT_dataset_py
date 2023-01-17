import numpy as np # linear algebra

import pandas as pd 

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df.head()
train_df.info()

print('<------------>')

test_df.info()
train_df = train_df.drop(['PassengerId','Name','Ticket'], axis = 1)

test_df = test_df.drop(['Name','Ticket'], axis = 1)
%matplotlib inline

train_df["Embarked"] = train_df["Embarked"].fillna("S")

sns.factorplot('Embarked', 'Survived', data = train_df, size = 4, aspect = 3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



sns.countplot(x='Embarked', data=train_df, ax=axis1)

sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', order=['S','C','Q'], data=embark_perc,ax=axis3)
embark_dummies_train  = pd.get_dummies(train_df['Embarked'])

embark_dummies_train.drop(['S'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train_df = train_df.join(embark_dummies_train)

test_df    = test_df.join(embark_dummies_test)



train_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True)
fig, (axis1, axis2) = plt.subplots(1,2,figsize= (10,2))

axis1.set_title("Original Ages")

axis2.set_title("New Ages")



avg_age_train = train_df["Age"].mean()

std_age_train = train_df["Age"].std()

count_nan_age_train = train_df["Age"].isnull().sum()



avg_age_test = test_df["Age"].mean()

std_age_test = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



rand1 = np.random.randint(avg_age_train - std_age_train, avg_age_train + std_age_train, size = count_nan_age_train)

rand2 = np.random.randint(avg_age_test - std_age_test, avg_age_test + std_age_test, size = count_nan_age_test)



train_df["Age"].dropna().astype(int).hist(bins = 70, ax=axis1)



train_df["Age"][np.isnan(train_df["Age"])] = rand1

test_df["Age"][np.isnan(test_df["Age"])] = rand2



train_df["Age"] = train_df["Age"].astype(int)

test_df["Age"] = test_df["Age"].astype(int)



train_df["Age"].hist(bins = 70, ax = axis2)
facet = sns.FacetGrid(train_df, hue = "Survived", aspect = 3)

facet.map(sns.kdeplot, 'Age', shade=True)

facet.set(xlim=(0, train_df["Age"].max()))

facet.add_legend()



fig, axis1 = plt.subplots(1,1,figsize=(18,3))

average_age = train_df[["Age", "Survived"]].groupby(["Age"], as_index=False).mean()

sns.barplot(x="Age",y="Survived", data=average_age)
train_df.drop("Cabin", axis=1, inplace=True)

test_df.drop("Cabin", axis=1, inplace=True)
train_df['Family'] = train_df['Parch'] + train_df["SibSp"]

train_df["Family"].loc[train_df["Family"] > 0] = 1

train_df["Family"].loc[train_df["Family"] == 0] = 0



test_df['Family'] = test_df['Parch'] + test_df["SibSp"]

test_df["Family"].loc[test_df["Family"] > 0] = 1

test_df["Family"].loc[test_df["Family"] == 0] = 0



train_df = train_df.drop(['Parch','SibSp'], axis = 1)

test_df = test_df.drop(['Parch','SibSp'], axis = 1)
fig, (axis1, axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

sns.countplot(x='Family', data=train_df, order=[1,0], ax=axis1)



family_perc = train_df[['Family','Survived']].groupby(['Family'], as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax = axis2)
def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



train_df["Person"] = train_df[['Age','Sex']].apply(get_person, axis=1)

test_df['Person'] = test_df[['Age','Sex']].apply(get_person, axis=1)



train_df.drop(['Sex','Age'], axis = 1, inplace=True)

test_df.drop(['Sex','Age'], axis = 1, inplace=True)



person_dummies_train = pd.get_dummies(train_df['Person'])

person_dummies_train.columns = ['Child', 'Female', 'Male']

person_dummies_train.drop(['Male'], axis = 1, inplace = True)



person_dummies_test = pd.get_dummies(test_df['Person'])

person_dummies_test.columns = ['Child', 'Female', 'Male']

person_dummies_test.drop(['Male'], axis = 1, inplace = True)



train_df = train_df.join(person_dummies_train)

test_df = test_df.join(person_dummies_test)



fig, (axis1, axis2) = plt.subplots(1,2,figsize=(10,5))



sns.countplot(x='Person', data=train_df, ax = axis1)



person_perc = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()

sns.barplot(x='Person', y='Survived', data=person_perc, ax= axis2, order =['male','female', 'child'])



train_df.drop(['Person'], axis = 1, inplace = True)

test_df.drop(['Person'], axis = 1, inplace = True)
sns.factorplot('Pclass', 'Survived', order=[1,2,3], data=train_df, size = 4)



pclass_dummies_train = pd.get_dummies(train_df['Pclass'])

pclass_dummies_train.columns = ['C1','C2', 'C3']

pclass_dummies_train.drop(['C3'], axis=1, inplace=True)



pclass_dummies_test = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['C1','C2', 'C3']

pclass_dummies_test.drop(['C3'], axis=1, inplace=True)



train_df.drop(['Pclass'], axis = 1, inplace=True)

test_df.drop(['Pclass'], axis = 1, inplace=True)



train_df = train_df.join(pclass_dummies_train)

test_df = test_df.join(pclass_dummies_test)
X_train = train_df.drop("Survived",axis = 1)

Y_train = train_df['Survived']

X_test = test_df.drop("PassengerId", axis = 1).copy()
X_train.head()

X_test["Fare"].fillna(X_test["Fare"].median(),inplace = True)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train,Y_train)

print('asa')
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
coeff_df = DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["coeff. est."] = pd.Series(logreg.coef_[0])



coeff_df
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)



submission.save()