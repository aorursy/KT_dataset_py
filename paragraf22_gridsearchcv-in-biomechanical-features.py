import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
plt.style.use('fivethirtyeight')
data.head()
data.describe()
data.info()
data["class"].value_counts()
plt.figure(figsize=(8,6))

fig_class =  data['class'].value_counts(normalize=True).plot(kind="bar",color=['red','green'],width=0.3)

fig_class.set_title("Class count")
abnormal = data[data['class'] == "Abnormal"]

normal = data[data['class'] == "Normal"]

plt.scatter(abnormal['lumbar_lordosis_angle'], abnormal['degree_spondylolisthesis'], color = "red",label = "Abnormal")

plt.scatter(normal['lumbar_lordosis_angle'], normal['degree_spondylolisthesis'], color = "green",label = "Normal")

plt.legend()

plt.xlabel("Lumbar Lordosis")

plt.ylabel("Degree Spondylolisthesis")

plt.show()
data['class'] = [1 if each == "Abnormal" else 0 for each in data['class']]
Y = data["class"]

X = data.drop(["class"],axis=1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



param_grid = {'n_neighbors':np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv=GridSearchCV(knn,param_grid,cv=3)

knn_cv.fit(X,Y)

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

knn = KNeighborsClassifier(n_neighbors = 4)

knn.fit(x_train,y_train)

prediciton = knn.predict(x_test)

print("With KNN(3) accuracy is: ",knn.score(x_test,y_test))
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier



param_grid = { 

    'n_estimators': [200,300,400,500],

    'max_depth' : [4,5,6,7,8],

    'criterion' :['gini', 'entropy']

}

rdf = RandomForestClassifier(random_state=42)

rdf_cv=GridSearchCV(rdf,param_grid,cv=3)

rdf_cv.fit(X,Y)

print("Tuned hyperparameter k: {}".format(rdf_cv.best_params_)) 

print("Best score: {}".format(rdf_cv.best_score_))
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state = 42, test_size = 0.25)

rf=RandomForestClassifier(criterion='entropy', max_depth=7, max_features='auto', n_estimators=500,random_state=42)

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)

cm=confusion_matrix(y_test,y_pred)

print("With RandomForestClassifier accuracy is: ",rf.score(x_test,y_test))
sns.heatmap(cm,annot=True,fmt='d')

plt.show()
from sklearn.linear_model import LogisticRegression

param_grid={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}

logreg=LogisticRegression()

logreg_cv=GridSearchCV(logreg,param_grid,cv=3)

logreg_cv.fit(X,Y)

print("Tuned hyperparameter k: {}".format(logreg_cv.best_params_)) 

print("Best score: {}".format(logreg_cv.best_score_))
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state = 42, test_size = 0.25)

logreg=LogisticRegression(C=100,penalty='l2',random_state=42)

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

print("With LogisticRegression accuracy is: ",logreg.score(x_test,y_test))
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

steps=[('scalar',StandardScaler()),('SVM',SVC())]

pipeline=Pipeline(steps)

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}

cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)

cv.fit(X,Y)



y_pred = cv.predict(X)

print("Tuned Model Parameters: {}".format(cv.best_params_)) 

print("Accuracy: {}".format(cv.best_score_))
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state = 42, test_size = 0.25)

steps=[('scalar',StandardScaler()),('SVM',SVC(C=100,gamma=0.01))]

pipeline = Pipeline(steps)

pipeline.fit(x_train,y_train)

y_pred=pipeline.predict(x_test)

print("With SVC accuracy is: ",pipeline.score(x_test,y_test))