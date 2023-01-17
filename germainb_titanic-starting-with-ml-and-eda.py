import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.info()
test.info()
print('Mean survived : '+str(train['Survived'].mean()))
sns.barplot('Pclass', 'Survived',data=train, order=[1,2,3])
sns.barplot('Sex', 'Survived',data=train)
sns.barplot('Pclass', 'Survived', 'Sex', data=train)
train['Sex'] = train['Sex'].map({'male':0, 'female':1})
test['Sex'] = test['Sex'].map({'male':0, 'female':1})
sns.barplot('SibSp', 'Survived',data=train)
sns.barplot('Parch', 'Survived',data=train)
train['Family'] = train['SibSp'] + train['Parch']
test['Family'] = test['SibSp'] + test['Parch']
sns.barplot('Family', 'Survived', data=train)
s = sns.FacetGrid(train, row='Pclass', col='Sex', aspect=3)
s.map(sns.barplot,'Family', 'Survived', ci=None)
s.add_legend()
train.groupby('Embarked')['Pclass'].count()
test.groupby('Embarked')['Pclass'].count()
sns.barplot('Embarked','Survived', data=train)
def groupembarked(a):
    if a=='C':
        return 1
    return 0
train['EmbarkedAtC'] = train['Embarked'].apply(groupembarked) #I didn't fill the NaN value so we can't use map
test['EmbarkedAtC'] = test['Embarked'].apply(groupembarked)
train[train.Cabin.notnull()]['Cabin'][:10]
def keepfirst(a):
    if isinstance(a, str):
        return a[0]
    return 'U'
train['Cabin'] = train['Cabin'].apply(keepfirst)
test['Cabin'] = test['Cabin'].apply(keepfirst)
sns.barplot('Cabin','Survived', data=train, order=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U'])
train.groupby('Cabin')['Survived'].count()
test.groupby('Cabin')['Pclass'].count()
def cabingroup(a):
    if a in 'CGAF':
        return 1
    elif a!='U':
        return 2
    else :
        return 0

train['CabinGroup'] = train['Cabin'].apply(cabingroup)
test['CabinGroup'] = test['Cabin'].apply(cabingroup)
sns.pointplot('CabinGroup','Survived', data=train)
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train.groupby('Title')['Survived'].mean()
def replaceTitle(df):
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Dona'],'HighF')
    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Major', 'Rev', 'Jonkheer', 'Don','Sir'], 'HighM')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df.loc[(df.Sex == 0)   & (df.Title == 'Dr'),'Title'] = 'Mr'
    df.loc[(df.Sex == 1) & (df.Title == 'Dr'),'Title'] = 'Mrs'
    
replaceTitle(train)
replaceTitle(test)
print('Title replace')
sns.barplot('Title','Survived','Pclass', data=train[train.Sex == 0], ci=None)
sns.barplot('Title','Survived','Pclass', data=train[train.Sex == 1], ci=None)
title_mapping = {"Mr": 0, "Miss": 0, "Mrs": 0, "Master": 1, "HighF": 0, 'HighM':0 }
train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)
sns.barplot('Title','Survived','Pclass', data=train[train.Sex == 0])
train[train.Age.notnull()].Survived.mean()
train[train.Age.notnull() == False].Survived.mean()
combined = pd.concat([train, test])
age_mean = combined['Age'].mean()
age_std = combined['Age'].std()

def fill_missing_age(a):
    if np.isnan(a):
        return np.random.randint(age_mean-age_std, age_mean+age_std, size=1)
    return a
train['AgeFill']=train['Age'].apply(fill_missing_age)
test['AgeFill']=test['Age'].apply(fill_missing_age)
sns.regplot('Age', 'Survived', data=train, order=3)
s = sns.FacetGrid(train, row='Pclass', aspect=3)
s.map(sns.barplot,'Age', 'Survived', ci=None)
s.set(xlim=(1,30))
s.add_legend()
def agegrouping(a):
    if a<2:
        return 1
    if a<12:
        return 2
    if a>60:
        return 3
    return 0
train['AgeGroup'] = train['AgeFill'].apply(agegrouping)
test['AgeGroup'] = test['AgeFill'].apply(agegrouping)
s = sns.FacetGrid(train,hue='Survived',aspect=3, size=4)
s.map(sns.kdeplot,'Fare',shade=True)
s.set(xlim=(0,200))
s.add_legend()
#Ticket=train.groupby('Ticket').count()
combined = pd.concat([train.drop('Survived', axis=1), test])
ticket=combined[combined.duplicated(subset=['Ticket'], keep=False)].sort_values('Ticket')[['Name', 'Ticket','Fare', 'Family', 'Pclass', 'Cabin','Age', 'Embarked']]
ticket.head()
combined = pd.concat([train.drop('Survived', axis=1), test])
ticket_count = combined[combined.duplicated(subset=['Ticket'], keep=False)].sort_values('Ticket')[['Name', 'Ticket']]
ticket_count = ticket_count.groupby('Ticket').count()[['Name']].reset_index()
ticket_count.columns = ['Ticket','Count']
ticket_count.head()
def calculFare(df, ticket_count):
    farecalcul = df['Fare'].copy()
    for row in df.iterrows():
        t = ticket_count[ticket_count.Ticket.str.match(row[1]['Ticket'])]
        if t.empty==False:
            farecalcul[row[0]]= row[1]['Fare']/t['Count'].values[0]
    return farecalcul
train['FareCalcul'] = calculFare(train, ticket_count)
s = sns.FacetGrid(train,hue='Survived',aspect=3, size=4)
s.map(sns.kdeplot,'FareCalcul',shade=True)
s.set(xlim=(0, 60))
s.add_legend()
s = sns.FacetGrid(train,hue='Survived', row='Pclass', aspect=3, size=4)
s.map(sns.kdeplot,'FareCalcul',shade=True)
s.set(xlim=(0, 60))
s.add_legend()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
Keep = ['Pclass', 'Sex', 'Family', 'AgeGroup', 'EmbarkedAtC', 'CabinGroup', 'Title']
X=train[Keep]
y=train['Survived']
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()
param = {'criterion': ['gini', 'entropy'],
         'max_depth': [3, 4, 5, 6, 10, 20, None],
         'max_features': ['sqrt','log2', None],
         'min_samples_leaf': [1, 2, 5, 0.05, 0.1, 0.2],
         'min_samples_split': [2, 0.05, 5, 0.1, 0.2, 0.3]}
GS = GridSearchCV(clf, param, scoring='accuracy', cv=5, n_jobs=5 )
GS.fit(X, y)
pred = GS.predict(X_test)
print(accuracy_score(y_test, pred))
print(GS.best_score_)
GS.best_estimator_
X_test=test[Keep]
DTC = DecisionTreeClassifier()
DTC.fit(X,y)
pred = DTC.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': pred })
output.to_csv('titanic-DecisionTree.csv', index = False)
X_test=test[Keep]
LR = LogisticRegression()
LR.fit(X,y)
pred = LR.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': pred })
output.to_csv('titanic-LogisticRegression.csv', index = False)
X_test=test[Keep]
svc = SVC()
svc.fit(X,y)
pred = svc.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test['PassengerId'], 'Survived': pred })
output.to_csv('titanic-SVC.csv', index = False)
X_test=test[Keep]
ensemble_voting = VotingClassifier(estimators=[('lg', LR), ('svm', svc), ('dc', DTC)], voting='hard')
ensemble_voting.fit(X, y)
pred = ensemble_voting.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })
submission.to_csv('submission_ensemble_voting.csv', index=False)