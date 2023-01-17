import pandas as pd

import numpy as np

import random as rnd
titanic = pd.read_csv("../input/train.csv")
titanic.head()
titanic_test = pd.read_csv("../input/test.csv")

titanic_test.head()
titanic.info()
titanic.describe()

#统计描述
pd.isnull(titanic).sum()
pd.isnull(titanic_test).sum()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

titanic.hist(bins=10,figsize=(9,7),grid=False)
titanic.drop("Cabin",axis=1,inplace=True)

titanic_test.drop("Cabin",axis=1,inplace=True)

#Cabin 重要性低，且缺失率高，所以去除
titanic.drop("Name",axis=1,inplace=True)

titanic_test.drop("Name",axis=1,inplace=True)

titanic.drop("Ticket",axis=1,inplace=True)

titanic_test.drop("Ticket",axis=1,inplace=True)

#Name列在分析中无作用，Ticket列对存活率影响也很小
titanic[titanic['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic)
titanic["Embarked"] = titanic["Embarked"].fillna('C')

#由上图填充缺失的Embarked值
guess_ages = np.zeros((2,3))

guess_ages
for dataset in [titanic,titanic_test]:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)





titanic.head()
titanic['AgeBand'] = pd.cut(titanic['Age'], 5)

titanic[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#年龄分组
for dataset in [titanic,titanic_test]:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']

titanic.head()
titanic = titanic.drop(['AgeBand'], axis=1)

combine = [titanic, titanic_test]

titanic.head()
for dataset in [titanic,titanic_test]:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



titanic[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#创建新列isalone

for dataset in [titanic,titanic_test]:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



titanic[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
titanic[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in [titanic,titanic_test]:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



titanic.head()
for dataset in [titanic,titanic_test]:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



titanic.head()
titanic_test['Fare'].fillna(titanic_test['Fare'].dropna().median(), inplace=True)

titanic_test.head()
titanic['FareBand'] = pd.qcut(titanic['Fare'], 4)

titanic[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



titanic = titanic.drop(['FareBand'], axis=1)

combine = [titanic, titanic_test]

    

titanic.head(10)
titanic.drop("Parch",axis=1,inplace=True)

titanic_test.drop("Parch",axis=1,inplace=True)

titanic.drop("SibSp",axis=1,inplace=True)

titanic_test.drop("SibSp",axis=1,inplace=True)

#Name列在分析中无作用，Ticket列对存活率影响也很小
titanic.drop("PassengerId",axis=1,inplace=True)
X_train = titanic.drop("Survived", axis=1)

Y_train = titanic["Survived"]

X_test  = titanic_test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
models = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 

              'Random Forest'],

    'Score': [acc_knn, acc_log, 

              acc_random_forest]})

models.sort_values(by='Score', ascending=False)