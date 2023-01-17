# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Importing Training & Testing Dataset
train_set = pd.read_csv("../input/titanic/train.csv")
test_set = pd.read_csv("../input/titanic/test.csv")

# Preview the data
train_set.head()
print(train_set.isnull().sum())
# Survival/ dead count

sns.countplot(x = "Survived", data = train_set)
# Survival/ dead count based on gender

sns.countplot(x="Survived",hue="Sex", data=train_set)
# dropping unnecessary columns, these columns won't be useful for prediction

train_set.drop(["PassengerId","Cabin", "Name", "Ticket",], axis = 1, inplace= True)

# Using most occured value fill missing values in Embarked column

train_set["Embarked"].fillna(train_set.Embarked.mode()[0], inplace = True)

# Fill missing values with median value for Age column

train_set["Age"].fillna(train_set.Age.median(), inplace = True)
# Encoading dummy variables for categorical data( "Sex", "Pclass", "Embarked" columns)

sex = pd.get_dummies(train_set["Sex"], drop_first = True,)
pclass = pd.get_dummies(train_set["Pclass"], drop_first = True)
embark = pd.get_dummies(train_set["Embarked"], drop_first= True)

# drop "Sex", "Pclass", "Embarked"  columns 

train_set.drop(["Pclass", "Sex", "Embarked"], axis = 1, inplace= True)

# concat the dummy variables into training set

train_set = pd.concat([train_set,sex,pclass,embark],axis = 1)
print(test_set.isnull().sum())
# dropping unnecessary columns, these columns won't be useful for prediction

X_test = test_set.drop(["PassengerId","Cabin", "Name", "Ticket",], axis = 1)

# Using most occured value fill missing values in Fare column

X_test['Fare'].fillna(X_test.Fare.median(), inplace=True)

# Fill missing values with median value for Age column

X_test["Age"].fillna(X_test.Age.median(), inplace = True)
# Encoading dummy variables for categorical data( "Sex", "Pclass", "Embarked" columns)

sex = pd.get_dummies(X_test["Sex"], drop_first = True,)
pclass = pd.get_dummies(X_test["Pclass"], drop_first = True)
embark = pd.get_dummies(X_test["Embarked"], drop_first= True)

# drop "Sex", "Pclass", "Embarked"  columns 

X_test.drop(["Pclass", "Sex", "Embarked"], axis = 1, inplace= True)

# concat the dummy variables into training set

X_test = pd.concat([X_test,sex,pclass,embark],axis = 1)
# Define training sets

X_train = train_set.drop("Survived",axis=1)
y_train = train_set["Survived"]
# Logistic Regression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
round(log_reg.score(X_train,y_train)*100,2)
# K-Nearest Neighbours

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
round(knn.score(X_train,y_train)*100,2)
# Decision Tree Classification

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
y_pred = decision_tree.predict(X_test)
round(decision_tree.score(X_train,y_train)*100,2)
#  Random Forest Classification

random_forest = RandomForestClassifier(n_estimators=90)
random_forest.fit(X_train,y_train)
y_pred = random_forest.predict(X_test)
round(random_forest.score(X_train,y_train)*100,2)
submission = pd.DataFrame({
        "PassengerId": test_set["PassengerId"],
        "Survived": y_pred})
submission.to_csv('Titanic_output.csv', index =False)
