# data analysis
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Reading files:
train_df= pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

combine = [train_df, test_df]
train_df
# Droped column
train_df.drop(['Cabin'], axis=1, inplace=True)
test_df.drop(['Cabin'], axis=1, inplace=True)
combine = [train_df, test_df]
# visualization
sns.countplot(train_df.Survived)
plt.show()


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#
#
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


sns.countplot(train_df.Pclass)
plt.show()


sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.show()


sns.countplot(train_df.Sex)
plt.show()

sns.barplot(x='Sex', y='Survived', data=train_df)
plt.show()


plt.hist(train_df.Age)
plt.xlabel('Age')
plt.ylabel('count')
plt.show()

sns.countplot(train_df.SibSp)
plt.show()

sns.barplot(x='SibSp', y='Survived', data=train_df)
plt.show()

sns.countplot(train_df.Embarked)
plt.show()

sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.show()
#Clining and combine DATA 
for dataset in combine:
        # Family: 0,1,2
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)
    dataset.loc[ dataset['Family'] >=  2, 'Family'] = 2
    
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)
    
            # Convert text to number.
    dataset["Sex"] = dataset["Sex"].map({'male':0, 'female':1})
    dataset["Embarked"] = dataset["Embarked"].map({'Q':0, 'S':1, "C":2})
    
    dataset.loc[(dataset['Age'] <=20), 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 40), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age']   = 2
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 3
    dataset['Age'] = dataset['Age'].fillna(4)
    
            # Fare
    dataset.loc[(dataset['Fare'] <= 7.854), 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].fillna(5)
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df.head()
train_df.drop(['PassengerId', 'Name', 'Ticket','Parch', 'SibSp'], axis=1, inplace=True)
test_df.drop([ 'Name', 'Ticket', 'Parch','SibSp'], axis=1, inplace=True)
train_df.head()
test_df.head()
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
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
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Decision Tree

decision_tree = DecisionTreeClassifier(min_weight_fraction_leaf =0.0005)
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
models = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'KNN', 
               'Naive Bayes','Linear SVC','Decision Tree','Random Forest'],
    
    'Score': [acc_log, 
              acc_svc,
              acc_knn,
              acc_gaussian,
               acc_linear_svc,
               acc_decision_tree,
              acc_random_forest        ]})
models.sort_values(by='Score', ascending=False)


#submission.to_csv("submission.csv", index = False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv("submission.csv", index = False)