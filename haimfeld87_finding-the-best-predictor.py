import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/pml-training.csv", error_bad_lines=False, index_col=False, dtype='unicode')
train.head()
#train.isnull().sum()
train.dropna(axis=1,thresh=int(0.20*train.shape[0]),inplace=True)

train.isnull().sum()
train.columns
sns.factorplot(x="user_name", data=train, kind="count", palette="Set1")
sns.factorplot(x="classe", data=train, kind="count", palette="Set2")
train['new_window'] = train['new_window'].map( {'no': 0, 'yes': 1} ).astype(int)



train = train.drop(['Unnamed: 0','cvtd_timestamp'], axis=1)



train['user_name'] =  train['user_name'].map( {'carlitos': 0, 'pedro': 1, 'adelmo': 2, 'charles': 3,

     'eurico': 4, 'jeremy': 5} ).astype(int)



#train['classe'] =  train['classe'].map( {'A': 0, 'B': 1, 'C': 2, 'D': 3,

#     'E': 4} ).astype(int)
X=train.drop(['classe'], axis=1)

Y=train['classe']
X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.1, random_state=1)
from sklearn.metrics import classification_report
from sklearn.svm import SVC

svc=SVC()

svc.fit(X_train, Y_train)

Y_pred_svc=svc.predict(X_test)

print("SVC report \n", classification_report(Y_pred_svc,Y_test))
from sklearn.neighbors import KNeighborsClassifier

KNN=KNeighborsClassifier()

KNN.fit(X_train, Y_train)

Y_pred_KNN=KNN.predict(X_test)

print("K-nearest neighbors Classifier report \n", classification_report(Y_pred_KNN,Y_test))
from sklearn.naive_bayes import GaussianNB

GNB=GaussianNB()

GNB.fit(X_train, Y_train)

Y_pred_GNB=GNB.predict(X_test)

print("Gaussian Naive Bayes report \n", classification_report(Y_pred_GNB,Y_test))
from sklearn.ensemble import AdaBoostClassifier

AdaB=AdaBoostClassifier()

AdaB.fit(X_train, Y_train)

Y_pred_AdaB=AdaB.predict(X_test)

print("AdaBoost Classifier report \n", classification_report(Y_pred_AdaB,Y_test))
from sklearn.tree import ExtraTreeClassifier

ETC=ExtraTreeClassifier()

ETC.fit(X_train, Y_train)

Y_pred_ETC=ETC.predict(X_test)

print("Extremely Randomized Trees Classifier report \n", classification_report(Y_pred_ETC,Y_test))
from sklearn.ensemble import BaggingClassifier

BC=BaggingClassifier()

BC.fit(X_train, Y_train)

Y_pred_BC=BC.predict(X_test)

print("Bagging Classifier report \n", classification_report(Y_pred_BC,Y_test))
from sklearn.ensemble import GradientBoostingClassifier

GBC=GradientBoostingClassifier()

GBC.fit(X_train, Y_train)

Y_pred_GBC=GBC.predict(X_test)

print("Gradient Boosting Classifier report \n", classification_report(Y_pred_GBC,Y_test))
from sklearn.tree import DecisionTreeClassifier

DTC=DecisionTreeClassifier()

DTC.fit(X_train, Y_train)

Y_pred_DTC=DTC.predict(X_test)

print("Decision Tree Classifier report \n", classification_report(Y_pred_DTC,Y_test))
from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier()

RF.fit(X_train, Y_train)

Y_pred_RF=RF.predict(X_test)

print("Random Forest Classifier report \n", classification_report(Y_pred_RF,Y_test))
from sklearn.metrics import confusion_matrix



cfm_GBC = confusion_matrix(Y_test, Y_pred_GBC)

cfm_GBC=cfm_GBC / cfm_GBC.astype(np.float).sum(axis=1)



cfm_BC = confusion_matrix(Y_test, Y_pred_BC)

cfm_BC=cfm_BC / cfm_BC.astype(np.float).sum(axis=1)



cfm_DTC = confusion_matrix(Y_test, Y_pred_DTC)

cfm_DTC=cfm_DTC / cfm_DTC.astype(np.float).sum(axis=1)



cfm_RF = confusion_matrix(Y_test, Y_pred_RF)

cfm_RF=cfm_RF / cfm_RF.astype(np.float).sum(axis=1)



f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10,8))

sns.heatmap(cfm_GBC, annot = True,  linewidths=.5, ax=ax1, cbar =None)

ax1.set_title('Gradient Boosting Classifier')

plt.ylabel('True label')

plt.xlabel('Predicted label')

sns.heatmap(cfm_BC, annot = True,  linewidths=.5, ax=ax2, cbar =None)

ax2.set_title('Bagging Classifier')

plt.ylabel('True label')

plt.xlabel('Predicted label')

sns.heatmap(cfm_DTC, annot = True, linewidths=.5, ax=ax3, cbar =None)

ax3.set_title('Decision Tree Classifier')

plt.ylabel('True label')

plt.xlabel('Predicted label')

sns.heatmap(cfm_RF, annot = True, linewidths=.5, ax=ax4, cbar =None)

ax4.set_title('Random Forest Classifier')

plt.ylabel('True label')

plt.xlabel('Predicted label')
from sklearn.grid_search import GridSearchCV



BaggingClassifier_parameters = {"n_estimators":list(range(50,60))}



clf_BC = GridSearchCV(BaggingClassifier(), BaggingClassifier_parameters ,cv=3, verbose=1)

clf_BC.fit(X_train, Y_train)
print(clf_BC.best_params_)

print(clf_BC.grid_scores_)
Y_pred_BC=clf_BC.predict(X_test)

print("Bagging Classifier report \n", classification_report(Y_pred_BC,Y_test))
#I need to choose better n_estimators

RandomForestClassifier_parameters = {"n_estimators":list(range(10,20)),

#              "max_features":["sqrt","log2","auto"],

#              "max_depth": [1000, None],

#              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}

#              "min_samples_leaf": [1, 2, 100],

#              "class_weight":["balanced", "auto"]}



clf_RF = GridSearchCV(RandomForestClassifier(), RandomForestClassifier_parameters, cv=3, verbose=1)

clf_RF.fit(X_train, Y_train)
print(clf_RF.best_params_)

print(clf_RF.grid_scores_)
Y_pred_RF=clf_RF.predict(X_test)

print("Random Forest Classifier report \n", classification_report(Y_pred_RF,Y_test))
cfm_BC = confusion_matrix(Y_test, Y_pred_BC)

cfm_BC=cfm_BC / cfm_BC.astype(np.float).sum(axis=1)



cfm_RF = confusion_matrix(Y_test, Y_pred_RF)

cfm_RF=cfm_RF / cfm_RF.astype(np.float).sum(axis=1)



f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8,4))

sns.heatmap(cfm_BC, annot = True,  linewidths=.5, ax=ax1, cbar =None)

ax1.set_title('Bagging Classifier')

plt.ylabel('True label')

plt.xlabel('Predicted label')

sns.heatmap(cfm_RF, annot = True, linewidths=.5, ax=ax2, cbar =None)

ax2.set_title('Random Forest Classifier')

plt.ylabel('True label')

plt.xlabel('Predicted label')
BC_RF=BaggingClassifier(RandomForestClassifier())

BC_RF.fit(X_train, Y_train)

Y_pred_BC_RF=BC_RF.predict(X_test)

print("Random Forest and Bagging Classifier report \n", classification_report(Y_pred_BC_RF,Y_test))
cfm_BC_RF = confusion_matrix(Y_test, Y_pred_BC_RF)

cfm_BC_RF=cfm_BC_RF / cfm_BC_RF.astype(np.float).sum(axis=1)



sns.heatmap(cfm_BC_RF, annot = True,  linewidths=.5)

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.title('Random Forest and Bagging Classifier confusion matrix')