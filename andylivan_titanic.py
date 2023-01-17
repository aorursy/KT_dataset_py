import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt





# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
train.describe()
train.info()
class_surv = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

class_surv
plt.pie(class_surv['Survived'], labels=class_surv['Pclass'], autopct='%0.0f%%', shadow=True, explode=[0.05,0.05,0.05], startangle=90)

plt.title('Survival percentage depending on passengers class (P. Class)')

plt.show()
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
sex_survived = train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean()

sex_survived 
plt.pie(sex_survived['Survived'], labels=sex_survived['Sex'], autopct='%0.0f%%', shadow=True, explode=[0.05,0.05], startangle=90)

plt.title('Survival percentage depending on passengers sex (Sex)')

plt.show()
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
sibsp_train = train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean()

sibsp_train
bar1 = plt.bar(sibsp_train['SibSp'], sibsp_train['Survived']*100)

plt.xticks(sibsp_train['SibSp'])

plt.ylabel('Survival rate (%)')

plt.title('Survival percentage depending on number of siblings/spouses aboard (# Sib/Sp)')





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        plt.annotate('%0.1f%%' %height,

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  

                    textcoords="offset points",

                    ha='center', va='bottom')

autolabel(bar1)





plt.show()
parch_train = train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean()

parch_train
bar1 = plt.bar(parch_train['Parch'], parch_train['Survived']*100)

plt.xticks(parch_train['Parch'])

plt.ylabel('Survival rate (%)')

plt.title('Survival percentage depending on number of parents/children aboard (# Par/Ch)')





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        plt.annotate('%0.1f%%' %height,

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  

                    textcoords="offset points",

                    ha='center', va='bottom')

autolabel(bar1)





plt.show()
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
train_df = train.drop(['Ticket', 'Cabin'], axis=1)

test_df = test.drop(['Ticket', 'Cabin'], axis=1)
combined = [train_df, test_df]

for dataset in combined:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combined:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combined:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name','PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combined = [train_df, test_df]
for dataset in combined:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combined:   

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()
for dataset in combined:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combined = [train_df, test_df]
for dataset in combined:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combined:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1



train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine = [train_df, test_df]



train_df.head()
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
test_df.head(10)
X_train = train_df.drop(['Survived'], axis=1)

y_train = train_df['Survived']

X_test = test_df.drop(['PassengerId'], axis=1)

y_test = gender_submission['Survived']
X_train.shape, y_train.shape, X_test.shape, y_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Score: ', logreg.score(X_train, y_train))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

rfor = RandomForestClassifier(n_estimators=100)

rfor.fit(X_train, y_train)

y_pred = rfor.predict(X_test)

print('Score: ', rfor.score(X_train, y_train))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
dtr = DecisionTreeClassifier()

dtr.fit(X_train, y_train)

y_pred = dtr.predict(X_test)

print('Score: ', dtr.score(X_train, y_train))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

print('Score: ', gnb.score(X_train, y_train))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })
submission