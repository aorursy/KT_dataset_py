import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
#Importing Dataset

dataset = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
#Visualising top 10 records

dataset.head()
# Basic info of columnsabs

dataset.info()
dataset.describe()
plt.figure(figsize=(10,10))

sns.heatmap(dataset.corr(), annot=True, fmt='.1f')

plt.show()
#age analysis

dataset.Age.value_counts()[:10]
sns.barplot(x= dataset.Age.value_counts()[:10].index, y= dataset.Age.value_counts()[:10].values  )

plt.xlabel('Age')

plt.ylabel("Age counter")

plt.title("Age Analysis")

plt.show
dataset.Outcome.value_counts()
countNoDisease = len(dataset[dataset.Outcome == 0])

countHaveDisease = len(dataset[dataset.Outcome == 1])

print("Percentage of patients dont have Diabetes: {:.2f}%".format((countNoDisease/(len(dataset.Outcome)))*100))

print("Percentage of patients have Diabetes: {:.2f}%".format((countHaveDisease/(len(dataset.Outcome)))*100))
young_ages = dataset[(dataset.Age>=29)&(dataset.Age<40)]

middle_ages =  dataset[(dataset.Age>=40)&(dataset.Age<55)]

elderly_ages =  dataset[(dataset.Age>=55)]



print("Young Ages", len(young_ages))

print("Middle Ages", len(middle_ages))

print("Elderly Ages", len(elderly_ages))
colors = ['blue','green','red']

explode= [1,1,1]

plt.figure(figsize=(8,8))

plt.pie([len(young_ages),len(middle_ages),len(elderly_ages)],labels=['Young Ages','Middle Ages','Elderly Ages'])

plt.show()
sns.countplot(dataset.Outcome)

plt.xlabel('Outcome')

plt.ylabel('Count')

plt.title('Outcome 1 & 0')

plt.show()
dataset.corr()
Data = dataset.drop(['Outcome'],axis =1)

Outcome = dataset.Outcome.values
x_train,x_test,y_train,y_test = train_test_split(Data,Outcome, test_size=0.2, random_state=1)
# Logistic Regression 

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

regressor = LogisticRegression()

regressor.fit(x_train,y_train)

print('Test Accuracy {:.2f}%'.format(regressor.score(x_test, y_test)*100))
# KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

print('KNN Accuracy {:.2f}%'.format(knn.score(x_test,y_test)*100))
# Support Vactor 

from sklearn.svm import SVC

svm = SVC(random_state=1)

svm1 = SVC(kernel='linear',gamma='scale',random_state=0)

svm2 = SVC(kernel='rbf',gamma='scale',random_state=0)

svm3 = SVC(kernel='poly',gamma='scale',random_state=0)

svm4 = SVC(kernel='sigmoid',gamma='scale',random_state=0)



svm.fit(x_train,y_train)

svm1.fit(x_train,y_train)

svm2.fit(x_train,y_train)

svm3.fit(x_train,y_train)

svm4.fit(x_train,y_train)



print('SVC Accuracy : {:,.2f}%'.format(svm.score(x_test,y_test)*100))



print('SVC Liner Accuracy : {:,.2f}%'.format(svm1.score(x_test,y_test)*100))



print('SVC RBF Accuracy : {:,.2f}%'.format(svm2.score(x_test,y_test)*100))



print('SVC Ploy Accuracy : {:,.2f}%'.format(svm3.score(x_test,y_test)*100))



print('SVC Sigmoid Accuracy : {:,.2f}%'.format(svm4.score(x_test,y_test)*100))











# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Naive Bayes Accuracy : {:,.2f}%".format(nb.score(x_test,y_test)*100))
# Random Forest Classifier



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, max_depth=100,random_state=1)

rf.fit(x_train,y_train)

print("Random Forest Accuracy : {:,.2f}%".format(rf.score(x_test,y_test)*100))
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=0)

dt.fit(x_train,y_train)

print("Decision Tree Accuracy : {:,.2f}%".format(dt.score(x_test,y_test)*100))
# XGboost

import xgboost

xg = xgboost.XGBClassifier()

xg.fit(x_train,y_train)

print("XGboost accuracy : {:.2f}%".format(xg.score(x_test,y_test)*100))