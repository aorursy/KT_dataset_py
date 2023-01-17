import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import GridSearchCV
datas = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
datas.head()
datas.shape
datas.columns
datas.diagnosis.value_counts()
datas.isnull().sum()
datas.drop(['Unnamed: 32','id'],axis=1,inplace=True)
datas[datas.isnull().any(axis=1)]
datas.describe()
corr=datas.corr()

corr.shape
plt.figure(figsize=(20,20))

sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True)
sns.countplot(datas['diagnosis'])

plt.xlabel('M/B')

plt.ylabel('Tumour Count')

plt.show()
plt.figure(figsize=(15,10))

sns.boxplot(x="diagnosis",y="radius_mean",data=datas)

plt.show()
sns.FacetGrid(datas,hue='diagnosis',height=10).map(sns.kdeplot,"radius_mean").add_legend()

plt.show()
X = datas.drop(['diagnosis'],axis=1)

y = datas['diagnosis']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
accuracy_lr = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("Logistic Regression Accuracy:",accuracy_lr)
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy_gnb = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("Gaussian NB Accuracy:",accuracy_gnb)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

parameters = {'max_features':['log2','sqrt','auto'],

             'criterion':['entropy','gini'],

             'max_depth':[2,3,5,10,50],

             'min_samples_split':[2,3,50,100],

             'min_samples_leaf':[1,5,8,10]}

grid_obj = GridSearchCV(clf,parameters)

grid_obj = grid_obj.fit(X_train,y_train)

clf = grid_obj.best_estimator_

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy_dt = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("Decision Tree Accuracy:",accuracy_dt)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9, 10, 15], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1, 5, 8]

             }

grid_obj = GridSearchCV(rf,parameters)

grid_obj = grid_obj.fit(X_train,y_train)

rf = grid_obj.best_estimator_

rf.fit(X_train,y_train)
accuracy_rf = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("Random Forest Classifier Accuracy:",accuracy_rf)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn import svm

svc = svm.SVC()

parameters = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

]

grid_obj = GridSearchCV(svc,parameters)

grid_obj = grid_obj.fit(X_train,y_train)

svc = grid_obj.best_estimator_

svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
accuracy_svm = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("SVM Accuracy:",accuracy_svm)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

parameters = { 'n_neighbors' :[3,4,5,10],

              'weights' : ['uniform','distance'],

             'algorithm': ['auto','ball_tree','kd_tree','brute'],

             'leaf_size' : [10,20,30,50]}

grid_obj = GridSearchCV(knn,parameters)

grid_obj = grid_obj.fit(X_train,y_train)

knn = grid_obj.best_estimator_

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
accuracy_knn = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("KNN Accuracy:",accuracy_knn)
models_used = pd.DataFrame({

    'Model': ['Logistic Regression','Naive Bayes','Decision Tree','Random Forest',

              'Support Vector Machines','K-Nearest Neighbors'],

    'Score' : [accuracy_lr,accuracy_gnb,accuracy_dt,accuracy_rf,accuracy_svm,accuracy_knn]})

models_used.sort_values(by='Score')