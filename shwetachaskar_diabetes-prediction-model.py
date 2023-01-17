import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
diabetes='../input/diabetes.csv'

data=pd.read_csv(diabetes)

data.head()
data.isnull().sum()
data.info()
X=data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values

X
Y=data['Outcome'].values

Y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=0)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)

predictions
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)

predictions
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.svm import SVC

classifier=SVC(kernel='linear',random_state=0)  

classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)

predictions
predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestRegressor

classifier=RandomForestRegressor(n_estimators=100,random_state=0,oob_score=True)

classifier.fit(X_train,y_train)
classifier.score(X_train,y_train)

acc_random_forest=round(classifier.score(X_train,y_train)*100,2)

print(round(acc_random_forest,2,), '%')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train, y_train)
knn.score(X_train,y_train)

acc_knn=round(knn.score(X_train,y_train)*100,2)

print(round(acc_knn,2,), '%')
Classification_Types=['Logistic Regression','Decision Tree','Support Vector Machine',

                      'Random Forest','k-Nearest Neighbors']

Score=['80%','70%','79%','89%','78%']
df=pd.DataFrame({'Classification_Types':Classification_Types,'Score':Score})

df