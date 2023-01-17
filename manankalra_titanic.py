#imports



import numpy as np

import pandas as pd

import random as rnd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB
train = pd.read_csv('../input/train.csv') #training data

test = pd.read_csv('../input/test.csv') #testing data - has no Survived column entries

full = [train, test]
test.columns.values
plt.figure(figsize=(20,12))

sns.heatmap(train.isnull(), cbar=False, cmap='viridis', yticklabels=False)

#Age, Cabin & Embarked
train.info()

#Embarked column also contains two missing values
test.info()

#Fare column contains one missing value
#not useful in analysis and prediction

#needs to be included in the final submission file

train.drop(['PassengerId'], axis=1, inplace=True)
train.head()
#number of survivors by 'Pclass'

sns.countplot(data=train, x='Survived', hue='Pclass')

#looks like survival rate for first class passengers is more
#number of unique values = 3

#train['Pclass'].nunique()



#number of passengers of each Pclass = 1 -> 216, 2 -> 184, 3 -> 491

#train['Pclass'].value_counts()



#impact on survival rate

train[['Pclass', 'Survived']].groupby('Pclass').mean()

#indicates that passengers with Pclass equal to 1 are more likely to survive
#will be dropped

#a new feature - 'Title' can be extracted from this feature

#might have an impact on survival rate as 'Title' can have a strong correlation with 'Age' and 'Sex'

for data in full:

    data['Title'] = data.Name.str.extract(' ([A-za-z]+)\.', expand=False)

full = [train, test]
#number of titles 

#train['Title'].nunique() - 14

#test['Title].nunique() - 5



#few of them can be combined into a Rare category

for data in full:

    data['Title'] = data['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Lady', 'Sir', 'Don', 'Capt', 'Jonkheer', 'Countess', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

full = [train, test]
train['Title'].value_counts()
#impact on survival

train[['Title', 'Survived']].groupby(['Title']).mean()
#converting categorial titles to ordinal

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data in full:

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)
for data in full:

    data.drop('Name', axis=1, inplace=True)

full = [train, test]
train.head()
test.head()
#number of survivors by 'Sex'

sns.countplot(data=train, x='Survived', hue='Sex')

#looks like survival rate for female sex is more
#number of Passengers of each 'Sex' = male -> 577, female -> 314

#train['Sex'].value_counts()



#impact on survival rate

train[['Sex', 'Survived']].groupby('Sex').mean()

#indicates that around 74% of the female passenegers survived 
#converting this categorical feature into a numerical feature

#gender_train = pd.get_dummies(train['Sex'], drop_first=True)

#train = pd.concat([train,gender_train], axis=1)
#converting this categorical feature into a numerical feature

#gender_test = pd.get_dummies(test['Sex'], drop_first=True)

#test = pd.concat([test,gender_test], axis=1)
for data in full:

    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)

full = [train, test]
train.head()
test.head()
#age of passengers

sns.distplot(train['Age'].dropna(), bins=30, kde=False)

#gives us a rough idea of passengers aboard the ship belong to which age group
#we can fill in the mean age of all the passengers

#a smarter way to do this is by filling in the mean age by Pclass

plt.figure(figsize=(12,8))

sns.boxplot(data=train, x='Pclass', y='Age')

#this shows that the passengers with Pclass = 1 are comparitively older
def fill_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24   

    else:

        return Age
for data in full:

    data['Age'] = data[['Age', 'Pclass']].apply(fill_age, axis=1)

full = [train, test]
plt.figure(figsize=(20,12))

sns.heatmap(train.isnull(), cbar=False, cmap='viridis', yticklabels=False)
#creating age bands

#impact on survival

#age bands need to be created onlyfor training data

train['AgeBand'] = pd.cut(train['Age'], 5)

train[['AgeBand', 'Survived']].groupby(['AgeBand']).mean()
#replacing 'Age' with ordinals based on the generated bands

for data in full:    

    data.loc[data['Age'] <= 16, 'Age'] = 0

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[data['Age'] > 64, 'Age'] = 4

    data['Age'] = data['Age'].astype(int)
train.drop('AgeBand', axis=1, inplace=True)

full = [train, test]
train.head()
test.head()
sns.countplot(data=train, x='SibSp')
sns.countplot(data=train, x='Parch')
#creating a new feature - 'FamilySize' by adding the above two values

for data in full:

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

full = [train, test]



#impact on survival rate

train[['FamilySize', 'Survived']].groupby('FamilySize').mean()
#categorizing people to check whether being alone on the ship has any impact on the survival rate or not

for data in full:

    '''

    if data['FamilySize'] == 1:

        data['isAlone'] = 1

    else:

        data['isAlone'] = 0

    '''

    #optimal way

    data['isAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'isAlone'] = 1

full = [train, test]

    

train[['isAlone', 'Survived']].groupby('isAlone').mean()
#dropping 'SibSp' and 'Parch'

#dropping 'FamilySize' in favor of 'isAlone'

for data in full:

    data.drop(['SibSp', 'Parch', 'FamilySize'], axis=1, inplace=True)

full = [train, test]
train.head()
test.head()
#not useful in analysis and prediction

for data in full:

    data.drop(['Ticket'], axis=1, inplace=True)

full = [train, test]
train.head()
test.head()
#filling missing value

test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
#creating fare bands

#impact on survival

#fare bands need to be created only for training data

train['FareBand'] = pd.qcut(train['Fare'], 4)

train[['FareBand', 'Survived']].groupby(['FareBand']).mean()
#replacing 'Fare' with ordinals based on the generated bands

for data in full:    

    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2

    data.loc[data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)

full = [train, test]
train.drop('FareBand', axis=1, inplace=True)
train.head()
test.head()
#this feature has a lot of null values

for data in full:

    data.drop(['Cabin'], axis=1, inplace=True)

full = [train, test]
train.head()
test.head()
#filling missing values

#as we only have two missing values, we can fill them with the most common occurence, i.e. S

#train['Embarked'].value_counts() - S = 644

for data in full:

    data['Embarked'].fillna('S', inplace=True)

full = [train, test]
#conversion

for data in full:

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

full = [train, test]
train.head()
test.head()
#Supervised Learning
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), train['Survived'], test_size=0.30)
logistic_regression = LogisticRegression()

logistic_regression.fit(X_train, y_train)

predictions = logistic_regression.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
svc = SVC()

svc.fit(X_train, y_train)

predictions = svc.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
gnb = GaussianNB()

gnb.fit(X_train, y_train)

predictions = gnb.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

predictions = linear_svc.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
gnb = GaussianNB()

gnb.fit(X_train, y_train)

predictions = gnb.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
sgdc = SGDClassifier()

sgdc.fit(X_train, y_train)

predictions = sgdc.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

predictions = decision_tree.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

predictions = random_forest.predict(X_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
#using SVC

svc.fit(train.drop('Survived', axis=1), train['Survived'])

predictions_final = svc.predict(test.drop('PassengerId', axis=1))
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions_final})
submission.to_csv('submission.csv', index=False)