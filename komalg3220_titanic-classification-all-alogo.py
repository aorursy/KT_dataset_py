

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

from sklearn import preprocessing  # to normalisation

from sklearn.model_selection import train_test_split as dsplit

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

import warnings #to ignore warnings





print(os.listdir("../input"))

warnings.filterwarnings("ignore")





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        





df = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')

df.drop(['PassengerId','Cabin','Age'],axis=1,inplace=True) #to drop column

df.head(6)
#null value

df.isnull().any()
#to check how manu null values are there



df.isnull().sum()
#To check unique values 



df['Embarked'].value_counts()
#To fill null value

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0]) # mode
#null value

df.isnull().any()
#To check the data type of coloumn



df.dtypes
#To check wheater column is contnious or categorical



for column in df.columns:

    print(column,len(df[column].unique()))
# To check Correlation

#df.corr() #df.corr(method='spearman')

df.corr().abs().unstack().sort_values()['Survived']
#To define x and y

x = df.loc[:, df.columns != 'Survived'] #to  select multiple column except one data point may be that we want to predict

#x=df.loc[:, ~df.columns.isin(['price','symboling','stroke','compression-ratio','peak-rpm','gas'])] # to  select multiple column except all the data point that we dont need.

y=df['Survived'].values #.values = to get the numpy array and dataset dont return index value and column with selected column



#convert categorical values (either text or integer) 

#df = pd.get_dummies(df, columns=['type'])

x=pd.get_dummies(x,columns=['Name','Pclass','Sex','SibSp','Parch','Embarked','Fare','Ticket'])

print(x.columns)



#To Normalise the equation

#x=preprocessing.normalize(x)

#print(x.head())

#print(y)
#logestic regression

x_train, x_test, y_train, y_test = dsplit(x, y, random_state = 1)

reg = LogisticRegression()

reg.fit(x_train, y_train)

predicted = reg.predict(x_test)

from sklearn.metrics import accuracy_score

print(reg.score(x_train, y_train))

print(reg.score(x_test, y_test))







'''# Logistic Regression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)



y_pred = logreg.predict(x_test)



acc_log = round(logreg.score(x_train, y_train) * 100, 2)

print(round(acc_log,2,), "%")'''
# stochastic gradient descent (SGD) learning

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)



sgd.score(x_train, y_train)

print(sgd.score(x_train, y_train))

print(sgd.score(x_test, y_test))



# Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)



y_prediction = random_forest.predict(x_test)



random_forest.score(x_train, y_train)

print(random_forest.score(x_train, y_train))

print(random_forest.score(x_test, y_test))

# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)



y_pred = knn.predict(x_test)



knn.score(x_train, y_train)

print(knn.score(x_train, y_train))

print(knn.score(x_test, y_test))
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)



y_pred = gaussian.predict(x_test)



gaussian.score(x_train, y_train)

print(gaussian.score(x_train, y_train))

print(gaussian.score(x_test, y_test))
# Perceptron

perceptron = Perceptron(max_iter=5)

perceptron.fit(x_train, y_train)



y_pred = perceptron.predict(x_test)



perceptron.score(x_train, y_train)

print(perceptron.score(x_train, y_train))

print(perceptron.score(x_test, y_test))
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)



y_pred = linear_svc.predict(x_test)



perceptron.score(x_train, y_train)

print(linear_svc.score(x_train, y_train))

print(linear_svc.score(x_test, y_test))
# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(x_train, y_train)



y_pred = decision_tree.predict(x_test)



decision_tree.score(x_train, y_train)

print(decision_tree.score(x_train, y_train))

print(decision_tree.score(x_test, y_test))
#which is the best model

results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [linear_svc, knn, reg, 

              random_forest, gaussian, perceptron, 

              sgd, decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
#Feature Importance

importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(random_forest.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)