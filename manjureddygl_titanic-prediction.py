import pandas as pd

import numpy as np



# Importing Classifier Modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.svm import SVC



import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.describe(include='all')
test.describe(include='all')
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
import matplotlib.pyplot as plt # Plot the graphes

%matplotlib inline

import seaborn as sns

sns.set() # setting seaborn default for plots
def bar_chart(feature):

    survived = train[train['Survived']==1][feature].value_counts()

    dead = train[train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')

print("Survived :\n",train[train['Survived']==1]['Sex'].value_counts())

print("---------------------------")

print("Dead:\n",train[train['Survived']==0]['Sex'].value_counts())
bar_chart('Pclass')

print("Survived :\n",train[train['Survived']==1]['Pclass'].value_counts())

print("Dead:\n",train[train['Survived']==0]['Pclass'].value_counts())
bar_chart('SibSp')

print("Survived :\n",train[train['Survived']==1]['SibSp'].value_counts())

print("Dead:\n",train[train['Survived']==0]['SibSp'].value_counts())
bar_chart('Parch')

print("Survived :\n",train[train['Survived']==1]['Parch'].value_counts())

print("Dead:\n",train[train['Survived']==0]['Parch'].value_counts())
bar_chart('Embarked')

print("Survived :\n",train[train['Survived']==1]['Embarked'].value_counts())

print("Dead:\n",train[train['Survived']==0]['Embarked'].value_counts())
train.head()
train_test_data = [train,test]



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 

                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }



for dataset in train_test_data:

    dataset['Title'] = dataset["Title"].map(title_mapping)
dataset.head()
test.head()
bar_chart('Title')
# delete unnecessary feature from dataset

train.drop('Name', axis=1, inplace=True)

test.drop('Name', axis=1, inplace=True)
sex_mapping = {"male": 0, "female": 1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace= True)

test["Age"].fillna(test.groupby('Title')['Age'].transform("median"), inplace= True)
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend() 

plt.show()



facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend() 

plt.xlim(10,50);
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,

    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,

    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,

    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
bar_chart('Age')
Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()

Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()

Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index = ['1st Class','2nd Class','3rd Class']

df.plot(kind = 'bar', stacked =  True, figsize=(10,5))

plt.show()

print("Pclass1:\n",Pclass1,"\n")

print("Pclass2:\n",Pclass2,"\n")

print("Pclass3:\n",Pclass3,"\n")
for dataset in train_test_data:

    dataset['Embarked'] =  dataset['Embarked'].fillna('S')

    

train.head()    
embarked_mapping = {'S':0,'C':1,'Q':2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# fill missing Fare with median fare for each Pclass

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)

test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(20)
facet = sns.FacetGrid(train, hue="Survived",aspect=4 )

facet.map(sns.kdeplot, 'Fare', shade = True)

facet.set(xlim = (0, train['Fare'].max()))

facet.add_legend()

plt.show()
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Fare',shade= True)

facet.set(xlim=(0, train['Fare'].max()))

facet.add_legend()

plt.xlim(0, 20)
for dataset in train_test_data:

    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,

    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,

    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,

    dataset.loc[dataset['Fare'] >= 100, 'Fare'] = 3

    



train.head()    
train.Cabin.value_counts()
for dataset in train_test_data:

    dataset['Cabin'] =  dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5));
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
# fill missing Fare with median fare for each Pclass

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'FamilySize',shade= True)

facet.set(xlim=(0, train['FamilySize'].max()))

facet.add_legend()

plt.xlim(0)
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
train.head()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
X_train.drop(['PassengerId','Ticket'],axis=1,inplace=True)

X_train.head()
Y_train.head()
X_test.drop('Ticket',axis=1,inplace=True)
X_test.head()
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest


xgb_classifier = XGBClassifier(n_estimators=100)

xgb_classifier.fit(X_train, Y_train)

Y_pred = xgb_classifier.predict(X_test)

xgb_classifier.score(X_train, Y_train)

acc_xgb_classifier = round(xgb_classifier.score(X_train, Y_train) * 100, 2)

acc_xgb_classifier
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',

              'Decision Tree','XGB classifier'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_decision_tree,acc_xgb_classifier]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})

submission.to_csv('gender_submission.csv', index=False)
submission.head()