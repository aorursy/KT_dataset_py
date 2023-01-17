#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing datasets to the dataframe 
train_df =  pd.read_csv("../input/train.csv")
test_df =  pd.read_csv("../input/test.csv")
#let's look at the basic structure and data of the training dataframe 

train_df.head()
#let's look at the basic structure and data of the test dataframe
test_df.head()
#getting to know about the size of the training data

train_df.shape
#getting to know about the size of the test data
test_df.shape
#Understanding the datatype & number of non-null values of each column of Train and Test dataframe
train_df.info()
test_df.info()
train_df.describe()
#Since we are predicting the survival of passengers in the ship -- 
#the Name, Passsenger ID & Fare does not directly/indirectly impact the predicive models. 
#Hence, dropping of these 3 columns.

train_df = train_df.drop((['PassengerId','Fare','Name']),axis = 1)
test_df = test_df.drop(['Fare','Name'],axis = 1)
#Sex is categorical data. Converting this column to numerical values by assigning 0 -> Male and 1 -> Female

train_df['Sex'] = train_df['Sex'].map({'male':0,'female':1})
test_df['Sex'] = test_df['Sex'].map({'male':0,'female':1})
#Creating a new Family column with Sibsp and Parch columns
train_df['Family'] = train_df['SibSp']+train_df['Parch']
test_df['Family'] = test_df['SibSp']+test_df['Parch']
train_df = train_df.drop((['SibSp','Parch']),axis = 1)
test_df = test_df.drop(['SibSp','Parch'],axis = 1)
train_df.Family.value_counts()
test_df.Family.value_counts()
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0
train_df.Embarked.value_counts(dropna = False)
train_df['Embarked']= train_df.Embarked.fillna('S')
train_df['Embarked']= train_df['Embarked'].map({'S':0,'C':1,'Q':2})
test_df['Embarked']= test_df['Embarked'].map({'S':0,'C':1,'Q':2})
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
train_df = train_df.drop(['Ticket','Cabin'],axis = 1)
test_df = test_df.drop(['Ticket','Cabin'],axis = 1)
average_age_titanic   = train_df["Age"].mean()
std_age_titanic       = train_df["Age"].std()
count_nan_age_titanic = train_df["Age"].isnull().sum()
rand= np.random.randint(average_age_titanic-std_age_titanic,average_age_titanic+std_age_titanic,count_nan_age_titanic)
train_df["Age"][np.isnan(train_df["Age"])] = rand
# Normalising the data for missing AGE values. 

average_age_titanic   = test_df["Age"].mean()
std_age_titanic       = test_df["Age"].std()
count_nan_age_titanic = test_df["Age"].isnull().sum()
rand1= np.random.randint(average_age_titanic-std_age_titanic,average_age_titanic+std_age_titanic,count_nan_age_titanic)
test_df["Age"][np.isnan(test_df["Age"])] = rand1
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)
train_df.head()
test_df.head()
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
Y_train = train_df['Survived']
X_train = train_df.drop(['Survived'],axis = 1)
train_x,test_x,train_y,test_y =  train_test_split(X_train,Y_train,random_state = 3, stratify = Y_train)
models = []
models.append(('logistic', LogisticRegression()))
models.append(('Gaussian', GaussianNB()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=2)
    cv_results = model_selection.cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train_x, train_y)

Y_pred = random_forest.predict(test_x)
#Accuracy score for Random Forest Classifier
accuracy_score(test_y,Y_pred)
#Confusion Matrix for Random Forest Classifier
confusion_matrix(test_y,Y_pred)
logicR = LogisticRegression()

logicR.fit(train_x, train_y)

Y_pred1 = logicR.predict(test_x)
# Logistic regression accuracy score
accuracy_score(test_y,Y_pred1)
# Confusion matrix for Logistic Regression
confusion_matrix(test_y,Y_pred1)
GNB = GaussianNB()

GNB.fit(train_x, train_y)

Y_pred2 = GNB.predict(test_x)
# Accuracy score for Gaussian Naive Bay model
accuracy_score(test_y,Y_pred2)
# Confusion matrix for Gaussian Naive Bay
confusion_matrix(test_y,Y_pred2)
dtc = DecisionTreeClassifier()

dtc.fit(train_x, train_y)

Y_pred3 = dtc.predict(test_x)
# Accuracy score for Decision Tree Classifier
accuracy_score(test_y,Y_pred3)
# confusion matrix for Decision Tree Classifier
confusion_matrix(test_y,Y_pred3)
# Original test dataset
X_test = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Pred_Y = random_forest.predict(X_test)
# For submitting the Titanic survival prediction which obtained for test data
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Pred_Y
    })
submission.to_csv('submission.csv', index=False)