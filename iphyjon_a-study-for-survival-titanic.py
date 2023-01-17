# Import important libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline()
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
pid = test["PassengerId"]
train.info()
train.describe()
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', data = train)
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Pclass', data = train)
sns.barplot("Pclass", "Survived", data = train)

sns.barplot("Embarked", "Survived", data = train)

sns.distplot(train['Age'].dropna(), kde = False, bins = 30)
sns.countplot(x = 'SibSp', data = train)
sns.barplot("SibSp", "Survived", data = train)

sns.barplot("Parch", "Survived", data = train)
train['Fare'].iplot(kind = 'hist', bins = 30, color = 'blue')
sns.boxplot(x = 'Pclass', y = 'Age', data = train)
# We create a function to get the avg. age based on Pclass

def pred_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(pred_age, axis = 1)
# We create a function to get the avg. age based on Pclass

def pred_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 40

        elif Pclass == 2:
            return 28

        else:
            return 24

    else:
        return Age
    
test['Age'] = test[['Age','Pclass']].apply(pred_age, axis = 1)
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')

family = train["SibSp"] + train["Parch"]      # combining the siblings and family data
train["Alone"] = np.where(family >  0, 0, 1)  # If a passenger is traveling alone or with a family member

fam = test["SibSp"] + test["Parch"]
test["Alone"] = np.where(fam >  0, 0, 1)
train.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis = 1, inplace = True)
test.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis = 1, inplace = True)
train.info()
sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.countplot(x = 'Embarked', data = train)
train.fillna({"Embarked": "S"}, inplace = True)
sns.heatmap(test.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
test["Fare"].fillna(test["Fare"].median(), inplace = True)
train = pd.get_dummies(train, columns = ['Embarked', 'Sex', 'Pclass'], drop_first = True)
test = pd.get_dummies(test, columns = ['Embarked', 'Sex', 'Pclass'], drop_first = True)
train.head()
test.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = train.drop("Survived",axis = 1)
Y = train["Survived"]
#X_test = test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
# Logistic Regression

logmod = LogisticRegression(max_iter = 200)

logmod.fit(X_train,Y_train)

pred = logmod.predict(X_test)

acc_logmod = round(accuracy_score(Y_test, pred) * 100, 2)
print(acc_logmod)
# K Nearest Neighbors

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, Y_train)

knn_pred = knn.predict(X_test)

acc_knn = round(accuracy_score(Y_test, knn_pred) * 100, 2)

print(acc_knn)
#knn.score(X_train, Y_train)
# Support Vector Machines


param_grid = {"C": [0.1, 1, 10, 100, 1000], "gamma": [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose = 3)

grid.fit(X_train, Y_train)

grid_pred = grid.predict(X_test)

acc_svc = round(accuracy_score(Y_test, grid_pred) * 100, 2)
print(acc_svc)
#print(grid.score(X_train, Y_train))
# Random Forest


random_forest = RandomForestClassifier(n_estimators = 150, random_state = 5)

random_forest.fit(X_train, Y_train)

RF_pred = random_forest.predict(X_test)

acc_RF = round(accuracy_score(Y_test, RF_pred) * 100, 2)
print(acc_RF)
#random_forest.score(X_train, Y_train)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Support Vector Machines', 'Random Forest'],
    'Score': [acc_logmod, acc_knn, acc_svc, acc_RF]})

models.sort_values(by='Score', ascending = False)
predict = random_forest.predict(test)

submit = pd.DataFrame({
        "PassengerId": pid,
        "Survived": predict})

submit.to_csv('submit.csv', index=False)