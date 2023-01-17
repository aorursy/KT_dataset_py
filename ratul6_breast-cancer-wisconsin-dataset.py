# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

%matplotlib inline 

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import GridSearchCV
cancer=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

cancer.head()
cancer.shape
cancer.columns
cancer.diagnosis.value_counts()
cancer[cancer.isnull().any(axis=1)]
cancer = cancer.drop(['Unnamed: 32','id'],axis = 1)
cancer.head()
cancer.describe().T
corr=cancer.corr()

corr.shape
plt.figure(figsize=(40,40))

sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='coolwarm')
sns.FacetGrid(cancer,hue='diagnosis',height=10).map(sns.kdeplot,"radius_mean").add_legend()

plt.show()
from sklearn.metrics import classification_report, confusion_matrix

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

Diagnosis = label_encoder.fit_transform(cancer['diagnosis'])

Diagnosis = pd.DataFrame({'diagnosis': Diagnosis})

cancer.diagnosis = Diagnosis

cancer
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

y= cancer['diagnosis']

X = cancer.drop(columns =['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
accuracy_lr = round(metrics.accuracy_score(y_test,y_pred)*100,2)

print("Logistic Regression Accuracy:",accuracy_lr)
from sklearn.metrics import classification_report, confusion_matrix

cm = np.array(confusion_matrix(y_test, y_pred))

confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],

                         columns=['predicted_cancer','predicted_healthy'])

confusion
sns.heatmap(confusion, annot=True)
print(classification_report(y_test, y_pred))
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
models_used = pd.DataFrame({

    'Model': ['Logistic Regression','Naive Bayes','Decision Tree','Random Forest',

              'Support Vector Machines','K-Nearest Neighbors'],

    'Score' : [accuracy_lr,accuracy_gnb,accuracy_dt,accuracy_rf,accuracy_svm,accuracy_knn]})

models_used.sort_values(by='Score')