import numpy as np 

import pandas as pd 

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

labels = train.Survived
train.info()
train.head()
ax = sns.countplot(train['Survived'])

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x()+0.35, p.get_height()+1))
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
sns.countplot(train['Sex'],hue ='Survived',data=train)
bar_chart('Sex')
bar_chart('Pclass')
ax =  sns.countplot(train['Pclass'],hue ='Survived',data=train)

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))
print("Survival percentage of passenger in 1st class - ", (134/(134+80)*100))

print("Survival percentage of passenger in 2nd class - ", (87/(87+97)*100))

print("Survival Percentage of passenger in 3rd class - ", (119/(119+372)*100)) 
plt.figure(figsize=(20,5))

sns.set(style="darkgrid")

sns.distplot(train['Age'], bins = 40)
def age_pie(x,y,title):

    total_010 = train.loc[(train.Age >= x) & (train.Age <= y) ]['Survived'].count()

    survived_010 = train.loc[(train.Age <= y) & (train.Age >= x)]['Survived'].sum()

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

    ax.pie(x=[total_010 - survived_010,survived_010],labels = ['dead','Survived'], autopct='%1.1f%%')

    ax.set_title(title)

age_pie(0,10,'percentage of people died and survived represented in a pie chart between the age of 0 - 10 ')

age_pie(10,20,'percentage of people died and survived represented in a pie chart between the age of 10 - 20 ')
age_pie(20,30,'percentage of people died and survived represented in a pie chart between the age of 20 - 30 ')

age_pie(30,40,'percentage of people died and survived represented in a pie chart between the age of 30 - 40 ')
age_pie(40,50,'percentage of people died and survived represented in a pie chart between the age of 40 - 50 ')
age_pie(50,60,'percentage of people died and survived represented in a pie chart between the age of 50 - 60 ')
ax = sns.countplot(train['SibSp'],hue ='Survived',data=train)

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x(), p.get_height()+1))

ax = sns.countplot(train['Parch'],hue ='Survived',data=train)

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x(), p.get_height()+1))

plt.figure(figsize=(8,4))

plt.subplot(122)

sns.barplot(train.SibSp, train.Survived)

plt.subplot(122)

sns.barplot(train.Parch, train.Survived)

plt.subplots_adjust(wspace=0.5)

plt.figure(figsize=(20,5))

sns.set(style="darkgrid")

sns.distplot(train['Fare'])
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()
plt.figure(figsize=(12,8))

sns.heatmap(train.isnull(),cbar=False, yticklabels=False, cmap='viridis')
train.Age.mean()
train.Age.isnull().sum()
train.groupby('Pclass').Age.mean()
def inpute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else: return 24

    else: return Age
train['Age']=train[['Age','Pclass']].apply(inpute_age, axis=1)
plt.figure(figsize=(12,8))

sns.heatmap(train.isnull(),cbar=False, yticklabels=False, cmap='viridis')
train.drop('Embarked', axis=1, inplace=True)
plt.figure(figsize=(12,6))

sns.heatmap(train.isnull(),cbar=False, yticklabels=False, cmap='viridis')
train.Cabin.value_counts()
train.drop('Cabin', axis=1, inplace=True)
plt.figure(figsize=(12,6))

sns.heatmap(train.isnull(),cbar=False, yticklabels=False, cmap='viridis')
train.info()


train['Male'] = pd.get_dummies(train['Sex'], drop_first=True)
train.drop(['PassengerId', 'Name', 'Sex', 'Ticket'], axis=1, inplace=True)
train.head()
train.info()
train.info()
#Seperate the feature columns from the target column

X = train.drop('Survived', axis=1)

y = train['Survived']
X.info()
y
x_train, x_test,y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=-1, n_estimators=14)

clf.fit(x_train,y_train)
score = clf.score(x_test,y_test)

print("accuracy", round(score,2)*100)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

clf.fit(x_train, y_train)
score = clf.score(x_test,y_test)

print("accuracy", round(score,2)*100)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
#Clean the test data the same way we did the training data

test_df['Age']=test_df[['Age','Pclass']].apply(inpute_age, axis=1)

test_df.drop('Cabin', axis=1, inplace=True)

test_df['Male'] = pd.get_dummies(test_df['Sex'], drop_first=True)

test_df.drop(['PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
test_df.info()
test_df.drop(['Sex'], axis=1, inplace=True)
test_df.info()
test_df.isnull().sum()
mean = test_df.Fare.mean()
test_df['Fare'] = test_df.Fare.fillna(mean)
test_df.Fare.isnull().sum()
y_prediction = clf.predict(test_df)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

pass_ids = test['PassengerId']
submission = pd.DataFrame({

        "PassengerId": pass_ids,

        "Survived": y_prediction

    })

submission.to_csv('titanic.csv', index=False)