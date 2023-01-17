import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
print('train shape : {}'.format(train.shape))

print('-'*40)

print('train shape : {}'.format(test.shape))
print(train.info())

print('-'*40)

print(test.info())
train.describe()
train.describe(include=['O'])
cor = train.corr()

plt.figure(figsize=(8,6))

sns.heatmap(cor, vmax=1, annot=True)

plt.show()
print(train.isnull().sum())

print('-'*40)

print(test.isnull().sum())
gender_survived = train.groupby('Sex').Survived.mean().reset_index()

gender_survived
class_survived = train.groupby('Pclass').Survived.mean().reset_index()

class_survived
gender_class_survived = train.groupby(['Sex','Pclass']).Survived.mean().reset_index()

gender_class_survived
embarked_survived = train.groupby('Embarked').Survived.mean().reset_index()

embarked_survived 
parch_survived = train.groupby('Parch').Survived.mean().reset_index()

parch_survived 
table = pd.pivot_table(train, values='Survived', index=['SibSp', 'Parch'], aggfunc= 'mean').head()

table 
#set background and style of graph

sns.set_style('darkgrid')

sns.set_palette('pastel')



#figure size

plt.figure(figsize=(12,8))



#create 1st graph: Train data

colors = ['gold','yellowgreen','lightcoral']

plt.subplot(1,2,1)

plt.pie(train['Embarked'].value_counts(), labels = ['Southampton','Cherbourg','Queenstown'], 

        colors=colors, autopct = '%1.1f%%',startangle = 10)

plt.title('Proportion of Embarked in Train data')



#create 2nd graph: Test data

colors = ['silver','turquoise','wheat']

plt.subplot(1,2,2)

plt.pie(test['Embarked'].value_counts(), labels = ['Southampton','Cherbourg','Queenstown'], 

        colors=colors, autopct = '%1.1f%%',startangle = 10)

plt.title('Proportion of Embarked in Test data')



#show the data

plt.show()
#figure size

plt.figure(figsize=(12,8))



#create 1st graph: Train data

color = ['lightskyblue','orange']

plt.subplot(1,2,1)

plt.pie(train['Sex'].value_counts(),labels=['Male','Female'],colors=color, autopct = '%1.1f%%',startangle = 10)

plt.title('Gender Proportion in Titanic in Train Data')



#create 2nd graph: Test data

color2 = ['lime','yellow']

plt.subplot(1,2,2)

plt.pie(test['Sex'].value_counts(),labels=['Male','Female'],colors=color2, autopct = '%1.1f%%',startangle = 10)

plt.title('Gender Proportion in Titanic in Test Data')



#adjust the distance between plot

plt.subplots_adjust(wspace=0.5)



#show the data

plt.show()
#create histogram in the graph

plt.hist(train['SibSp'], color= 'red',alpha = 0.5)

plt.hist(test['SibSp'], color= 'blue', alpha = 0.5)



#Title

plt.title('Distribution of Siblings (SibSp) in Train (Red) and Test (Blue) Set')

#Legend

plt.legend(['Train','Test'])



#show the data

plt.show()
#create histogram in the graph

plt.hist(train['Parch'], color= 'red',alpha = 0.5)

plt.hist(test['Parch'], color= 'blue', alpha = 0.3)



#title

plt.title('Distribution of Parent (Parch) Passenger in Train (Red) and Test (Blue) Set')



#legend

plt.legend(['Train','Test'])



#show the data

plt.show()
#figure size

plt.figure(figsize=(12,8))



#Create 1st graph

plt.subplot(1,2,1)

ax = sns.boxplot(x="Fare",data=train)

plt.title('Fare Price in Train')



#Create 2nd graph

plt.subplot(1,2,2)

ax = sns.boxplot(x="Fare",data=test, color='red')

plt.title('Fare Price in Test')



#show the data

plt.show()
g = sns.FacetGrid(train,col='Survived')

g.map(plt.hist,'Age',bins=15)

plt.show()
g1 = sns.FacetGrid(train,col='Survived',row='Sex')

g1.map(plt.hist,'Age',bins=15)

plt.show()
g2 = sns.FacetGrid(train,col='Survived',row='Pclass')

g2.map(plt.hist,'Age',bins=20)

plt.show()
g3 = sns.FacetGrid(train,col='Embarked')

g3.map(sns.pointplot,'Pclass','Survived','Sex',c='red')

g3.add_legend()

plt.show()
g4 = sns.FacetGrid(train,col='Survived',row='Embarked')

g4.map(sns.barplot, 'Sex','Fare')

g4.add_legend()

plt.show()
train['Sex'] = train['Sex'].replace('female',0)

train['Sex'] = train['Sex'].replace('male',1)

test['Sex'] = test['Sex'].replace('female',0)

test['Sex'] = test['Sex'].replace('male',1)
print(train.Age.median())

print(test.Age.median())
train['Age'] = train['Age'].fillna(28)

test['Age'] = test['Age'].fillna(27)
train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].replace('S',int(0))

train['Embarked'] = train['Embarked'].replace('Q',int(1))

train['Embarked'] = train['Embarked'].replace('C',int(2))



test['Embarked'] = test['Embarked'].replace('S',int(0))

test['Embarked'] = test['Embarked'].replace('Q',int(1))

test['Embarked'] = test['Embarked'].replace('C',int(2))
print(train.SibSp.unique())

print(test.SibSp.unique())
train['SibSp'] = train['SibSp'].replace([1,2,3,4,5,8],1)

test['SibSp'] = train['SibSp'].replace([1,2,3,4,5,8],1)
print('train unique:{}'.format(train.SibSp.unique()))

print('test unique:{}'.format(test.SibSp.unique()))
print(train.Parch.unique())

print(test.Parch.unique())
train['Parch'] = train['Parch'].replace([1,2,3,4,5,6,9],1)

test['Parch'] = test['Parch'].replace([1,2,3,4,5,6,9],1)
print('train unique:{}'.format(train.Parch.unique()))

print('test unique:{}'.format(test.Parch.unique()))
train['AgeRange'] = pd.cut(train['Age'], 5)

train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
train['AgeCategory'] = train['AgeRange'].cat.codes

train.head()
#Make a new label based on Age Category

bins = [-0.01,16.336,32.252,48.168,64.084,80.0]

labels = [0,1,2,3,4]

test['AgeCategory'] = pd.cut(test['Age'],bins,labels=labels)

test.head()
print(test.Fare.mode())

test['Fare'] = test['Fare'].fillna(7.75)
print(train.Fare.describe())

print('-'*40)

print(test.Fare.describe())
bins = [-0.01,7.9104,14.4542,31.00,512.3292]

labels = [0,1,2,3]

train['FareCategory'] = pd.cut(train['Fare'],bins,labels=labels)

test['FareCategory'] = pd.cut(test['Fare'],bins,labels=labels)
test.head(3)
train2 = train.drop(['Ticket','Fare','Cabin','AgeRange','Age','Name','PassengerId'],axis=1)

train2.head()
test2 = test.drop(['Ticket','Fare','Cabin','Age','Name','PassengerId'],axis=1)

test2.head(100)
X_train = train2.drop('Survived',axis=1)

Y_train = train2['Survived']

X_test = test2

features = ['Pclass', 'Sex','SibSp','Parch','Embarked','AgeCategory','FareCategory']

print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)
#import logistic regression

from sklearn.linear_model import LogisticRegression



#correlates logistic regression with a variables

logreg = LogisticRegression()



#fit the logistic regression with X and Y train

logreg.fit(X_train, Y_train)



#Find accuracy and prediction

Y1_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



print('Accuracy : {}'.format(acc_log))
coeff = pd.DataFrame(X_train.columns)

coeff.columns = ['Feature']

coeff["Correlation"] = pd.Series(logreg.coef_[0])



coeff.sort_values(by='Correlation', ascending=True)
# Support Vector Machines

from sklearn.svm import SVC, LinearSVC



svc = SVC()

svc.fit(X_train, Y_train)

Y2_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)



print('Accuracy : {}'.format(acc_svc))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(X_train, Y_train)

Y3_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



print('Accuracy : {}'.format(acc_knn))
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y4_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



print('Accuracy : {}'.format(acc_gaussian))
# Random Forest

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y5_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



print('Accuracy : {}'.format(acc_random_forest))
feature_imp = pd.Series(random_forest.feature_importances_,index=features).sort_values(ascending=False)

feature_imp
from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y6_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree



print('Accuracy : {}'.format(acc_decision_tree))
feature_imp = pd.Series(decision_tree.feature_importances_,index=features).sort_values(ascending=False)

feature_imp
test3 = test.copy()

test3.head()
test3['Survived'] = Y5_pred
test3.head()
test4 = test3.drop(['Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','AgeCategory','FareCategory'],axis=1)
test4.head()
#Make the file into CSV

test4.to_csv('submission.csv',index = False)