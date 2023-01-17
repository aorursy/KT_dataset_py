import pandas as pd

import numpy as np

import random as rnd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



from subprocess import check_output

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]

print(train_df.columns.values)
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
def get_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



def title_map(title):

    if title in ['Mr']:

        return 1

    elif title in ['Master']:

        return 3

    elif title in ['Ms','Mlle','Miss']:

        return 4

    elif title in ['Mme','Mrs']:

        return 5

    else:

        return 2

    

train_df['title'] = train_df['Name'].apply(get_title).apply(title_map)   

test_df['title'] = test_df['Name'].apply(get_title).apply(title_map)

title_xt = pd.crosstab(train_df['title'], train_df['Survived'])

title_xt_pct = title_xt.div(title_xt.sum(1).astype(float), axis=0)



title_xt_pct.plot(kind='bar', 

                  stacked=True, 

                  title='Survival Rate by title')

plt.xlabel('title')

plt.ylabel('Survival Rate')
train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)

test_df = test_df.drop(['Name','Ticket'], axis=1)



# Embarked



# fill the two missing values with the most occurred value, "S".

train_df["Embarked"] = train_df["Embarked"].fillna("S")



# Remove "S" dummy variable, and leave "C" & "Q", since they seem to have a good rate for Survival.

s_train  = pd.get_dummies(train_df['Embarked'])

s_train.drop(['S'], axis=1, inplace=True)



s_test  = pd.get_dummies(test_df['Embarked'])

s_test.drop(['S'], axis=1, inplace=True)



train_df = train_df.join(s_train)

test_df = test_df.join(s_test)



train_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True)
# Fare



# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



train_df.loc[train_df['Fare'] <= 7.91, 'Fare'] = 0

train_df.loc[(train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454), 'Fare'] = 1

train_df.loc[(train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31), 'Fare'] = 2

train_df.loc[train_df['Fare'] > 31, 'Fare'] = 3

test_df.loc[test_df['Fare'] <= 7.91, 'Fare'] = 0

test_df.loc[(test_df['Fare'] > 7.91) & (test_df['Fare'] <= 14.454), 'Fare'] = 1

test_df.loc[(test_df['Fare'] > 14.454) & (test_df['Fare'] <= 31), 'Fare'] = 2

test_df.loc[test_df['Fare'] > 31, 'Fare'] = 3



# convert from float to int

train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)
# Age 



train_df['Age'] = train_df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

test_df['Age'] = test_df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))



# convert from float to int

train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)



train_df.loc[train_df['Age'] <= 16, 'Age'] = 0

train_df.loc[(train_df['Age'] > 16) & (train_df['Age'] <= 32), 'Age'] = 1

train_df.loc[(train_df['Age'] > 32) & (train_df['Age'] <= 48), 'Age'] = 2

train_df.loc[(train_df['Age'] > 48) & (train_df['Age'] <= 64), 'Age'] = 3

train_df.loc[(train_df['Age'] > 64), 'Age'] = 4



test_df.loc[ test_df['Age'] <= 16, 'Age'] = 0

test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1

test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 2

test_df.loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64), 'Age'] = 3

test_df.loc[(test_df['Age'] > 64), 'Age'] = 4
# Cabin

train_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)



# Family

train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]

train_df['Family'].loc[train_df['Family'] > 0] = 1

train_df['Family'].loc[train_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



train_df = train_df.drop(['SibSp','Parch'], axis=1)

test_df = test_df.drop(['SibSp','Parch'], axis=1)



# Sex



# optional: As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

#def get_person(passenger):

    #age,sex = passenger

    #return 'child' if age < 16 else sex

    

#titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)

#test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)



# No need to use Sex column since we created Person column

#titanic_df.drop(['Sex'],axis=1,inplace=True)

#test_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived

#person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])

#person_dummies_titanic.columns = ['Child','Female','Male']

#person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



#person_dummies_test  = pd.get_dummies(test_df['Person'])

#print(person_dummies_test)

#person_dummies_test.columns = ['Child','Female','Male']

#person_dummies_test.drop(['Male'], axis=1, inplace=True)



#titanic_df = titanic_df.join(person_dummies_titanic)

#test_df    = test_df.join(person_dummies_test)



#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

#sns.countplot(x='Person', data=titanic_df, ax=axis1)



# average of survived for each Person(male, female, or child)

#person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()

#sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])



#titanic_df.drop(['Person'],axis=1,inplace=True)

#test_df.drop(['Person'],axis=1,inplace=True)

sexes = sorted(train_df['Sex'].unique())

genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))

train_df['Sex'] = train_df['Sex'].map(genders_mapping).astype(int)

test_df['Sex'] = test_df['Sex'].map(genders_mapping).astype(int)



# Pclass



# optional:create dummy variables for Pclass column, 

# & drop 3rd class as it has the lowest average of survived passengers

#pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

#pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

#pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



#pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

#pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

#pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



#titanic_df.drop(['Pclass'],axis=1,inplace=True)

#test_df.drop(['Pclass'],axis=1,inplace=True)



#titanic_df = titanic_df.join(pclass_dummies_titanic)

#test_df    = test_df.join(pclass_dummies_test)

train_df['age_class'] = train_df['Age'] * train_df['Pclass']

test_df['age_class'] = test_df['Age'] * test_df['Pclass']



train_df.head()

test_df.head()
Y_train = train_df["Survived"].copy()

X_train = train_df.drop("Survived",axis=1).copy()

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression

Logreg = LogisticRegression()

Logreg.fit(X_train, Y_train)

Y_logistic = Logreg.predict(X_test)

Logreg.score(X_train, Y_train)
#Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_svc = svc.predict(X_test)

svc.score(X_train, Y_train)
#KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_knn = knn.predict(X_test)

knn.score(X_train, Y_train)
#Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_bayes = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_rf = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
Y_pred = np.rint((Y_logistic + Y_svc + Y_knn + Y_bayes + Y_rf)/5).astype(int)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('TitanicSubmission.csv', index=False)

#submission