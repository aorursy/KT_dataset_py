#import data packages
import math as m
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns #(sam seaborn!!)
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#Input csv filed into two data fram object: train_df and test_df
#train = 891 entries of passengers, features, with labels if survived or died 
#test = 418 entries of titanic passengers, features, no labels
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
#Examine first 10 rows to take a look at the data 
#note: passengerID randomly assigned by Kaggle (correlates with index)
train_df[:10]
#print info
train_df.info()
#lookin at test set
print(test_df.info())
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False) 
#sns.barplot(x = "Family", y = "Survived", data = train_df)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
#grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
#grid.map(plt.hist, 'Age', alpha=.5, bins=20)
#grid.add_legend();
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# changing males to 0 and females to 1 , changing the type to an integer 
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
#combining Sibling and spouses with parents and children to create a new family column 
train_df["Family"] = train_df["SibSp"] + train_df["Parch"]
test_df["Family"] = test_df["SibSp"] + test_df["Parch"] 

train_df.head()
#drop columns that Family replaces 
train_df.drop('Parch', axis = 1, inplace = True)
train_df.drop('SibSp', axis = 1, inplace = True)
test_df.drop('Parch', axis = 1, inplace = True)
test_df.drop('SibSp', axis = 1, inplace = True)

train_df.head()
test_df["IsAlone"] = 0
train_df["IsAlone"] = 0
train_df.loc[train_df['Family'] == 1, 'IsAlone'] = 1
test_df.loc[test_df['Family'] == 1, 'IsAlone'] = 1

#This was Manav's code (which I modified)
#for dataset in combine:
    #dataset['IsAlone'] = 0
    #dataset.loc[dataset['Family'] == 1, 'IsAlone'] = 1

train_df.info()
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
for val in train_df.columns:
    if val != 'IsAlone' and val != 'PassengerId' and val != 'Pclass' and val != 'Sex' and val != 'Survived':
        train_df = train_df.drop([val], axis=1)
for val in test_df.columns:
    if val != 'IsAlone' and val != 'PassengerId' and val != 'Pclass' and val != 'Sex':
        test_df = test_df.drop([val], axis=1)
train_df.head()
test_df.head() 
#split training data into the feature list w/o passenger id and labels (survived or not)
# features - x
# labels - y
X_train = train_df.drop(["Survived", "PassengerId"], axis=1)
Y_train = train_df["Survived"]

X_test = test_df.drop('PassengerId', axis=1)

X_train.head()
Y_train.head()
#from sklearn.cross_validation import train_test_split
#train_test_split splits our data for us into training and testing! (in a cross validation)
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

#this means a 70-30 split, pulling out 30% of data for test size (what percent of the data goes into the test)
#4 variables are returned from the method call

valid_X_train, valid_X_test, valid_Y_train, valid_Y_test = train_test_split(X_train, Y_train, train_size = .7, test_size = .3)

valid_X_train.shape, valid_X_test.shape, valid_Y_train.shape, valid_Y_test.shape 
 
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(valid_X_train, valid_Y_train)
Y_pred = logreg.predict(valid_X_test)

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
#this uses the method of the used classifier
# I got this specific code from Ms. Sconyers!
print((logreg.score(X_train, Y_train)*100), logreg.score(valid_X_train, valid_Y_train)*100)
svc = SVC()
svc.fit(valid_X_train, valid_Y_train)
Y_pred = svc.predict(valid_X_test)
#acc_svc = round(svc.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_svc

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((svc.score(X_train, Y_train)*100), svc.score(valid_X_train, valid_Y_train)*100)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(valid_X_train, valid_Y_train)
Y_pred = knn.predict(valid_X_test)
#acc_knn = round(knn.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_knn

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((knn.score(X_train, Y_train)*100), knn.score(valid_X_train, valid_Y_train)*100)
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(valid_X_train, valid_Y_train)
Y_pred = gaussian.predict(valid_X_test)
#acc_gaussian = round(gaussian.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_gaussian

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((gaussian.score(X_train, Y_train)*100), gaussian.score(valid_X_train, valid_Y_train)*100)
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(valid_X_train, valid_Y_train)
Y_pred = decision_tree.predict(valid_X_test)
#acc_decision_tree = round(decision_tree.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_decision_tree

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((decision_tree.score(X_train, Y_train)*100), decision_tree.score(valid_X_train, valid_Y_train)*100)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(valid_X_train, valid_Y_train)
Y_pred = random_forest.predict(valid_X_test)
random_forest.score(valid_X_train, valid_Y_train)
#acc_random_forest = round(random_forest.score(valid_X_train, valid_Y_train) * 100, 2)
#acc_random_forest

print(accuracy_score(valid_Y_test, Y_pred)* 100)

#a check to see if data overfitted- if these numbers are close together, then the data is not overfit 
print((random_forest.score(X_train, Y_train)*100), random_forest.score(valid_X_train, valid_Y_train)*100)
tree_pred = decision_tree.predict(X_test)
final_pred = tree_pred
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': final_pred
})
submission.to_csv('submission.csv', index=False)
submission.info()