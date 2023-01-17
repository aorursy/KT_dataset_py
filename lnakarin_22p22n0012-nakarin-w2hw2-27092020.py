import pandas as pd
import numpy as np
import random

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


train = pd.read_csv('//kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
# dataset = train.copy()

train.head(10)
train.tail(10)
train.describe()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.groupby('Pclass').mean()
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 6))
PC1 = train[train['Pclass']==1]
PC2 = train[train['Pclass']==2]
PC3 = train[train['Pclass']==3]
ax = sns.distplot(PC1[PC1['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(PC1[PC1['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('PClass = 1')
ax = sns.distplot(PC2[PC2['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(PC2[PC2['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('PClass = 3')
ax = sns.distplot(PC3[PC3['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[2], kde = False)
ax = sns.distplot(PC3[PC3['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[2], kde = False)
ax.legend()
_ = ax.set_title('PClass = 3')
# grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=10)
# grid.add_legend();
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.groupby('Sex').mean()
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Sex']=='female']
men = train[train['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
train = train.drop(['PassengerId','Name','Cabin','Embarked'], axis=1)
train.info()
train['Age'].fillna(train['Age'].mean(),inplace=True)

train.loc[ train['Age'] <= 11, 'Age'] = 0
train.loc[(train['Age'] > 11) & (train['Age'] <= 18), 'Age'] = 1
train.loc[(train['Age'] > 18) & (train['Age'] <= 22), 'Age'] = 2
train.loc[(train['Age'] > 22) & (train['Age'] <= 27), 'Age'] = 3
train.loc[(train['Age'] > 27) & (train['Age'] <= 33), 'Age'] = 4
train.loc[(train['Age'] > 33) & (train['Age'] <= 40), 'Age'] = 5
train.loc[(train['Age'] > 40) & (train['Age'] <= 66), 'Age'] = 6
train.loc[ train['Age'] > 66, 'Age'] = 6

train['Age'] = train['Age'].astype(int)
train.info()
genders = {"male": 0, "female": 1}
Sex = []

for index,row in train.iterrows():
    Sex.append(genders[row['Sex']])
    
train.drop(['Sex','Ticket','Fare'],axis=1,inplace=True)
train['Sex'] = Sex

train.info()
train.head()
test = test.drop(['PassengerId','Name','Cabin','Embarked'], axis=1)
test['Age'].fillna(train['Age'].mean(),inplace=True)
test.loc[ test['Age'] <= 11, 'Age'] = 0
test.loc[(test['Age'] > 11) & (test['Age'] <= 18), 'Age'] = 1
test.loc[(test['Age'] > 18) & (test['Age'] <= 22), 'Age'] = 2
test.loc[(test['Age'] > 22) & (test['Age'] <= 27), 'Age'] = 3
test.loc[(test['Age'] > 27) & (test['Age'] <= 33), 'Age'] = 4
test.loc[(test['Age'] > 33) & (test['Age'] <= 40), 'Age'] = 5
test.loc[(test['Age'] > 40) & (test['Age'] <= 66), 'Age'] = 6
test.loc[ test['Age'] > 66, 'Age'] = 6

test['Age'] = test['Age'].astype(int)

genders = {"male": 0, "female": 1}
Sex = []

for index,row in test.iterrows():
    Sex.append(genders[row['Sex']])
    
test.drop(['Sex','Ticket','Fare'],axis=1,inplace=True)
test['Sex'] = Sex
test.info()
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.copy()
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB() 
gaussian.fit(X_train, Y_train)  

Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
nn = Perceptron(max_iter=5)
nn.fit(X_train, Y_train)

Y_pred = nn.predict(X_test)
acc_nn = round(nn.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Naive Bayes', 'Neural Network'],
    'Score': [acc_decision_tree,acc_gaussian, acc_nn]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
scores = cross_val_score(decision_tree, X_train, Y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
predictions_decision_tree = cross_val_predict(decision_tree, X_train, Y_train, cv=5)
confusion_matrix(Y_train, predictions_decision_tree)
print("Precision:", precision_score(Y_train, predictions_decision_tree))
print("Recall:",recall_score(Y_train, predictions_decision_tree))
f1_score(Y_train, predictions_decision_tree)
print(metrics.confusion_matrix(Y_train, predictions_decision_tree))

print(metrics.classification_report(Y_train, predictions_decision_tree, digits=3))

scores = cross_val_score(gaussian, X_train, Y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
predictions_gaussian = cross_val_predict(gaussian, X_train, Y_train, cv=5)
confusion_matrix(Y_train, predictions_gaussian)
print("Precision:", precision_score(Y_train, predictions_gaussian))
print("Recall:",recall_score(Y_train, predictions_gaussian))
f1_score(Y_train, predictions_gaussian)
print(metrics.confusion_matrix(Y_train, predictions_gaussian))

print(metrics.classification_report(Y_train, predictions_gaussian, digits=3))

scores = cross_val_score(nn, X_train, Y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
predictions_nn = cross_val_predict(nn, X_train, Y_train, cv=5)
confusion_matrix(Y_train, predictions_nn)
print("Precision:", precision_score(Y_train, predictions_nn))
print("Recall:",recall_score(Y_train, predictions_nn))
f1_score(Y_train, predictions_nn)
print(metrics.confusion_matrix(Y_train, predictions_nn))

print(metrics.classification_report(Y_train, predictions_nn, digits=3))

print("Decision Tree :")
print(precision_recall_fscore_support(Y_train, predictions_decision_tree,average='micro'))
print("Naive Bayse")
print(precision_recall_fscore_support(Y_train, predictions_gaussian,average='micro'))
print("Nueral Network")
print(precision_recall_fscore_support(Y_train, predictions_nn,average='micro'))