import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Model Selection

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix



# Model Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

data.head()
data.info()
f, ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(),annot=True, linewidths=.5, ax=ax)

plt.show()
cor = data.corr() 

corr_target = abs(cor["DEATH_EVENT"])

relevant_features = corr_target[corr_target>0.1]

relevant_features
accuracy_list = []

algorithm = []

predict_list = []



X_train,X_test,Y_train,Y_test = train_test_split(data.loc[:,{"age","ejection_fraction","serum_creatinine","serum_sodium","time"}]

                                                 ,data["DEATH_EVENT"],test_size=0.2)

print("X_train shape :",X_train.shape)

print("Y_train shape :",Y_train.shape)

print("X_test shape :",X_test.shape)

print("Y_test shape :",Y_test.shape)
reg = LogisticRegression(max_iter=1000)

reg.fit(X_train,Y_train)

accuracy_list.append(reg.score(X_test,Y_test))

algorithm.append("Logistic Regression")

print("test accuracy ",reg.score(X_test,Y_test))



cm = confusion_matrix(Y_test,reg.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
knn = KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1, 25)}

knn_gscv = GridSearchCV(knn, param_grid, cv=4)

knn_gscv.fit(X_train, Y_train)

print("Best K Value is ",knn_gscv.best_params_)



accuracy_list.append(knn_gscv.score(X_test,Y_test))

print("test accuracy ",knn_gscv.score(X_test,Y_test))

algorithm.append("K Nearest Neighbors Classifier")



cm = confusion_matrix(Y_test,knn_gscv.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
svm = SVC()

svm.fit(X_train,Y_train)

print("test accuracy: ",svm.score(X_test,Y_test))

accuracy_list.append(svm.score(X_test,Y_test))

algorithm.append("Support Vector Machine")



cm = confusion_matrix(Y_test,svm.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
nb = GaussianNB()

nb.fit(X_train,Y_train)

print("test accuracy: ",nb.score(X_test,Y_test))

accuracy_list.append(nb.score(X_test,Y_test))

algorithm.append("Native Bayes Classifier")



cm = confusion_matrix(Y_test,nb.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

print("test accuracy: ",dt.score(X_test,Y_test))

accuracy_list.append(dt.score(X_test,Y_test))

algorithm.append("Decision Tree Classifier")



cm = confusion_matrix(Y_test,dt.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
param_grid = {'n_estimators': np.arange(10, 100, 10)}

rf = RandomForestClassifier(random_state = 42)

rf_gscv = GridSearchCV(rf, param_grid, cv=4)

rf_gscv.fit(X_train, Y_train)

print("Best K Value is ",rf_gscv.best_params_)



print("test accuracy: ",rf_gscv.score(X_test,Y_test))

accuracy_list.append(rf_gscv.score(X_test,Y_test))

algorithm.append("Random Forest Classifier")



cm = confusion_matrix(Y_test,rf_gscv.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
param_grid = {'n_estimators': [10,20,50],'learning_rate': [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],'max_features': [2],'max_depth': [2]}

gb = GradientBoostingClassifier()

gb_gscv = GridSearchCV(gb, param_grid, cv=4)

gb_gscv.fit(X_train,Y_train)

print("The best parameters are ",gb_gscv.best_params_)

print("------------------------------------------------------")

print("test accuracy is ",gb_gscv.score(X_test,Y_test))

accuracy_list.append(gb_gscv.score(X_test,Y_test))

algorithm.append("Gradient Boosting Classifier")



cm = confusion_matrix(Y_test,gb_gscv.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
xgb_clf = XGBClassifier()

xgb_clf.fit(X_train, Y_train)

print("test accuracy is ",xgb_clf.score(X_test,Y_test))

accuracy_list.append(xgb_clf.score(X_test,Y_test))

algorithm.append("XGBClassifier")



cm = confusion_matrix(Y_test,xgb_clf.predict(X_test))

predict_list.append(cm.item(0)+cm.item(2))

sns.heatmap(cm,annot=True, linewidths=.5)

plt.show()
#Classifier Accuracy

f,ax = plt.subplots(figsize = (15,7))

sns.barplot(x=accuracy_list,y=algorithm,palette = sns.cubehelix_palette(len(accuracy_list)))

plt.xlabel("Accuracy")

plt.ylabel("Classifier")

plt.title('Classifier Accuracy')

plt.show()
#Classifier Predict Death Event Count

f,ax = plt.subplots(figsize = (15,7))

sns.barplot(x=predict_list,y=algorithm,palette = sns.cubehelix_palette(len(accuracy_list)))

plt.xlabel("Predict Death Event Count")

plt.ylabel("Classifier")

plt.title('Classifier Predict Death Event Count')

plt.show()