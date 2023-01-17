#data analysis

import pandas as pd

import numpy as np



#data visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#to ingore warnings

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
train.shape
test.head()
test.head()
train.describe()
test.describe()
train['Embarked'].value_counts()
train['SibSp'].value_counts()
sns.barplot(x='Pclass', y='Survived',data=train)
sns.barplot(x='Sex', y='Survived', data=train)
sns.distplot((train['Age'].dropna()), kde=False)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)

test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train.head()
train['Title'] = train['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major','Rev','Jonkheer','Dona'],'Rare')

train['Title'] = train['Title'].replace(['Countess','Sir','Lady'],'Royal')

train['Title'] = train['Title'].replace('Mlle','Miss')

train['Title'] = train['Title'].replace('Ms', 'Miss')

train['Title'] = train['Title'].replace('Mme','Mrs')
test['Title'] = test['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')

test['Title'] = test['Title'].replace(['Countess','Sir','Lady'],'Royal')

test['Title'] = test['Title'].replace('Mlle','Miss')

test['Title'] = test['Title'].replace('Ms', 'Miss')

test['Title'] = test['Title'].replace('Mme', 'Mrs')
train[['Title','Age']].groupby('Title').mean()
for i in train['Title']:

    if i=='Master':

        train['Age'] = train['Age'].fillna(4)

    elif i=='Miss':

        train['Age'] = train['Age'].fillna(22)

    elif i=='Mr':

        train['Age'] = train['Age'].fillna(32)

    elif i=='Mrs':

        train['Age'] = train['Age'].fillna(36)

    elif i=='Rare':

        train['Age'] = train['Age'].fillna(46)

    else:

        train['Age'] = train['Age'].fillna(41)
train.isnull().sum()
test[['Title','Age']].groupby('Title').mean()
train.head()
for i in test['Title']:

    if i=='Master':

        test['Age'] = test['Age'].fillna(7)

    elif i=='Miss':

        test['Age'] = test['Age'].fillna(21)

    elif i=='Mr':

        test['Age'] = test['Age'].fillna(32)

    elif i=='Mrs':

        test['Age'] = test['Age'].fillna(38)

    elif i=='Rare':

        test['Age'] = test['Age'].fillna(43)

    else:

        test['Age'] = test['Age'].fillna(41)

        

        

test.isnull().sum()
test[['Pclass','Fare']].groupby('Pclass').mean()
test['Fare'] = test['Fare'].fillna(12)
test.isnull().sum()
train.isnull().sum()
train['Cabin'].shape
train['N_cabin'] = (train['Cabin'].notnull().astype('int'))

test['N_cabin'] = (test['Cabin'].notnull().astype('int'))
test['Cabin'].shape
train['N_cabin'].shape
test['N_cabin'].shape
train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)
train['Embarked'] = train['Embarked'].fillna('S')
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

train['Embarked'] = le.fit_transform(train['Embarked'])

test['Embarked'] = le.fit_transform(test['Embarked'])
train.head()
sex_mapping = {'male':0, 'female':1}



train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)
train.head()
train[['Title','Survived']].groupby(['Title'], as_index=False).mean().sort_values('Survived')
Title_mapping = {'Mr': 1,'Rare': 2,'Master': 3,'Miss': 4,'Mrs': 5,'Royal': 6}



train['Title'] = train['Title'].map(Title_mapping)

test['Title'] = test['Title'].map(Title_mapping)
train.loc[train['Title']]
train_name = train['Name']

for i in train['Name']:

    train['Name'] = train['Name'].replace(i, len(i))
train.head()
test_name = test['Name']

for i in test['Name']:

    test['Name'] = test['Name'].replace(i, len(i))
test['Name'].describe()
bins = [0,25,40, np.inf]

name_labels = ['s_name', 'm_name', 'l_name']

train['Name_len'] = pd.cut(train['Name'], bins, labels=name_labels)

test['Name_len'] = pd.cut(test['Name'], bins, labels=name_labels)
train['Name_len'].value_counts()
train[['Name_len', 'Survived']].groupby('Name_len').mean()
name_mapping = {'s_name':1, 'm_name':2, 'l_name':3}



train['Name_len'] = train['Name_len'].map(name_mapping)

test['Name_len'] = test['Name_len'].map(name_mapping)
train.head()
train = train.drop(['Name', 'PassengerId'], axis=1)

test = test.drop(['Name'], axis=1)
train['Age'].hist(bins=30,color='darkred',alpha=0.8)
bins = [0,5,12,18,24,35,60,np.inf]

Age_label = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']



train['AgeGroup'] = pd.cut(train['Age'], bins, labels = Age_label)

test['AgeGroup'] = pd.cut(test['Age'], bins, labels = Age_label)
train[['AgeGroup', 'Survived']].groupby('AgeGroup').mean()
age_mapping = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 

               'Young Adult':5,'Adult':6, 'Senior':7}





train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train.head()
train['FareBand'] = pd.qcut(train['Fare'], 8, labels = [1,2,3,4,5,6,7,8])

test['FareBand'] = pd.qcut(train['Fare'], 8, labels= [1,2,3,4,5,6,7,8])
#scaling data in fare and age columns



from sklearn.preprocessing import MinMaxScaler



mms = MinMaxScaler()



train['Fare'] = mms.fit_transform(train['Fare'].values.reshape(-1, 1))

test['Fare'] = mms.fit_transform(test['Fare'].values.reshape(-1, 1))



train['Age'] = mms.fit_transform(train['Age'].values.reshape(-1,1))

test['Age'] = mms.fit_transform(test['Age'].values.reshape(-1,1))
train.head()
train['FamilySize'] = train['SibSp']+train['Parch']+1

test['FamilySize'] = train['SibSp']+train['Parch']+1

sns.distplot(train['FamilySize'], kde=False)
train = train.drop(['Ticket'], axis=1)

test = test.drop(['Ticket'], axis=1)
combine = [train, test]



for dataset in combine:

    dataset['Single'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'Single'] = 1

    

train[['Single', 'Survived']].groupby(['Single'], as_index=False).mean()

train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test.head()
train.head()
X_train = train.drop('Survived', axis=1)

Y_train = train['Survived']

X_test = test.drop('PassengerId', axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_logreg = round(logreg.score(X_train, Y_train) * 100, 2)

acc_logreg
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier()

gbc.fit(X_train, Y_train)

y_pred = gbc.predict(X_test)

acc_gbc = round(gbc.score(X_train, Y_train) * 100, 2)

acc_gbc
from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, Y_train)

y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train)* 100, 2)

acc_linear_svc
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

y_pred = dt.predict(X_test)

acc_dt = round(dt.score(X_train, Y_train)* 100, 2)

acc_dt
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

rfc.fit(X_train, Y_train)

y_pred = rfc.predict(X_test)

acc_rfc = round(rfc.score(X_train, Y_train)* 100, 2)

acc_rfc
from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train)* 100, 2)

acc_perceptron
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, Y_train)

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train)* 100, 2)

acc_knn
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train)* 100, 2)

acc_sgd
from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB()

gnb.fit(X_train, Y_train)

y_pred = gnb.predict(X_test)

acc_gnb = round(gnb.score(X_train, Y_train)* 100, 2)

acc_gnb
models = pd.DataFrame({

    'Model': ['Logistic Regression','Gradient Boosting Classifer',

              'Support Vector Machine','Linear SVC','Decision Tree',

              'Random Forest','Perceptron','KNN',

              'Stochastic Gradient Descent',

              'Naive Bayes'],

    'Accuracy' : [acc_logreg,acc_gbc,acc_svc,

                  acc_linear_svc,acc_dt,acc_rfc,

                  acc_perceptron,acc_knn,acc_sgd,

                  acc_gnb]

})



models.sort_values(by='Accuracy', ascending=False)


passid = test['PassengerId']

predictions = rfc.predict(test.drop('PassengerId', axis=1))





submission = pd.DataFrame({ 'PassengerId' : passid, 'Survived': predictions })

submission.to_csv('submission.csv', index=False)