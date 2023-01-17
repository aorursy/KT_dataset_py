import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
wine = pd.read_csv('../input/winequality-red.csv')
wine.head(3)
wine.info()
wine[wine.isnull()].count()
plt.figure(figsize=(10,8))
sns.boxplot(wine['quality'],wine['fixed acidity'])
plt.figure(figsize=(10,8))
plt.scatter(wine['fixed acidity'],wine['pH'])
plt.xlabel('Acidity').set_size(20)
plt.ylabel('pH').set_size(20)
plt.figure(figsize=(10,8))
sns.pointplot(wine['quality'],wine['pH'], color='grey')
plt.xlabel('Quality').set_size(20)
plt.ylabel('pH').set_size(20)
plt.figure(figsize=(10,8))
sns.countplot(wine['citric acid'] > 0)
plt.xlabel('Citric Acid Content').set_size(20)
plt.ylabel('Count').set_size(20)
plt.figure(figsize=(10,8))
sns.barplot(wine['quality'],wine['total sulfur dioxide'])
plt.xlabel('Quality').set_size(20)
plt.ylabel('Total Sulfur Dioxide').set_size(20)
plt.figure(figsize=(10,8))
sns.pointplot(wine['pH'].round(1),wine['residual sugar'], color='green')
plt.xlabel('pH').set_size(15)
plt.ylabel('Residual Sugar').set_size(15)
plt.figure(figsize=(20,8))
sns.countplot(x=wine['chlorides'].round(2))
plt.xlabel('Chlorides').set_size(15)
plt.ylabel('Count').set_size(15)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(wine.drop(['quality'],axis=1))
scaled_feat = scaler.transform(wine.drop(['quality'],axis=1))
wine_scaled = pd.DataFrame(scaled_feat,columns=wine.columns[:-1])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
X = wine_scaled
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)
lr_conf_matrix = pd.DataFrame(data=metrics.confusion_matrix(y_test,lr_pred),index=[[3,4,5,6,7,8]], columns=[[3,4,5,6,7,8]])
print("Confusion Matrix:")
lr_conf_matrix
lr_as = metrics.accuracy_score(y_test,lr_pred)
print("Accuracy Score: {}\n\n".format(lr_as))
print("Classification Report: \n{}".format(metrics.classification_report(y_test,lr_pred)))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_pred = dtc.predict(X_test)
dtc_conf_matrix = pd.DataFrame(data=metrics.confusion_matrix(y_test,dtc_pred),index=[[3,4,5,6,7,8]], columns=[[3,4,5,6,7,8]])
print("Confusion Matrix: ")
dtc_conf_matrix
dtc_as = metrics.accuracy_score(y_test,dtc_pred)
print("Accuracy Score: {}\n\n".format(dtc_as))
print("Classification Report: \n{}".format(metrics.classification_report(y_test,dtc_pred)))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
rfc_conf_matrix = pd.DataFrame(data=metrics.confusion_matrix(y_test,rfc_pred),index=[[3,4,5,6,7,8]], columns=[[3,4,5,6,7,8]])
print("Confusion Matrix: ")
rfc_conf_matrix
rfc_as = metrics.accuracy_score(y_test,rfc_pred)
print("Accuracy Score: {}\n\n".format(rfc_as))
print("Classification Report: \n{}".format(metrics.classification_report(y_test,rfc_pred)))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
knn_conf_matrix = pd.DataFrame(data=metrics.confusion_matrix(y_test,knn_pred),index=[[3,4,5,6,7,8]], columns=[[3,4,5,6,7,8]])
print("Confusion Matrix: ")
knn_conf_matrix
knn_as = metrics.accuracy_score(y_test,knn_pred)
print("Accuracy Score: {}\n\n".format(knn_as))
print("Classification Report: \n{}".format(metrics.classification_report(y_test,knn_pred)))
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)
svm_conf_matrix = pd.DataFrame(data=metrics.confusion_matrix(y_test,svm_pred),index=[[3,4,5,6,7,8]], columns=[[3,4,5,6,7,8]])
print("Confusion Matrix: ")
svm_conf_matrix
svm_as = metrics.accuracy_score(y_test,svm_pred)
print("Accuracy Score: {}\n\n".format(svm_as))
print("Classification Report: \n{}".format(metrics.classification_report(y_test,svm_pred)))
summary_table_1_1 = pd.DataFrame([lr_as,dtc_as,rfc_as,knn_as,svm_as],index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM'], columns=['Accuracy Score'])
print("Summary Table for Section 1.1")
summary_table_1_1
plt.figure(figsize=(10,5))
sns.pointplot(summary_table_1_1.index,summary_table_1_1['Accuracy Score'])
from sklearn.cross_validation import cross_val_score
lr_cross = LogisticRegression()
dtc_cross = DecisionTreeClassifier()
rfc_cross = RandomForestClassifier(n_estimators=20)
knn_cross = KNeighborsClassifier(n_neighbors=1)
svm_grid = SVC()
lr_scores = cross_val_score(lr_cross,X,y,cv=10,scoring='accuracy')
dtc_scores = cross_val_score(dtc_cross,X,y,cv=10,scoring='accuracy')
rfc_scores = cross_val_score(rfc_cross,X,y,cv=10,scoring='accuracy')
knn_scores = cross_val_score(knn_cross,X,y,cv=10,scoring='accuracy')
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[0.1,1,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(svm_grid,param_grid)
grid.fit(X_train,y_train)
grid_predicitons = grid.predict(X_test)
svm_grid_score = metrics.accuracy_score(y_test,grid_predicitons)
summary_table_1_2 = pd.DataFrame([lr_scores.mean(),dtc_scores.mean(),rfc_scores.mean(),knn_scores.mean(),svm_grid_score],index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM'], columns=['Accuracy Score'])
print("Summary Table for Section 1.2")
summary_table_1_2
plt.figure(figsize=(10,5))
sns.pointplot(summary_table_1_2.index,summary_table_1_2['Accuracy Score'])
def add_encode(quality):
    if quality == 3 or quality == 4:
        return 0
    elif quality == 5 or quality == 6:
        return 1
    else:
        return 2
wine['quality_encoding'] = wine['quality'].apply(add_encode)
wine['quality_remarks'] = wine['quality_encoding'].map({0:'Poor',1:'Average',2:'Good'})
wine.head(3)
scaler = StandardScaler()
scaler.fit(wine.drop(['quality','quality_encoding', 'quality_remarks'],axis=1))
scaled_feat = scaler.transform(wine.drop(['quality','quality_encoding', 'quality_remarks'],axis=1))
wine_scaled_enc = pd.DataFrame(scaled_feat,columns=wine.columns[:-3])
X_enc = wine_scaled_enc
y_enc = wine['quality_encoding']
lr_cross_enc = LogisticRegression()
dtc_cross_enc = DecisionTreeClassifier()
rfc_cross_enc = RandomForestClassifier(n_estimators=20)
knn_cross_enc = KNeighborsClassifier(n_neighbors=1)
svm_grid_enc = SVC()
lr_scores_enc = cross_val_score(lr_cross,X_enc,y_enc,cv=10,scoring='accuracy')
dtc_scores_enc = cross_val_score(dtc_cross,X_enc,y_enc,cv=10,scoring='accuracy')
rfc_scores_enc = cross_val_score(rfc_cross,X_enc,y_enc,cv=10,scoring='accuracy')
knn_scores_enc = cross_val_score(knn_cross,X_enc,y_enc,cv=10,scoring='accuracy')
X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.30)
grid_enc = GridSearchCV(svm_grid_enc,param_grid)
grid_enc.fit(X_train,y_train)
grid_predicitons_enc = grid_enc.predict(X_test)
svm_grid_score_enc = metrics.accuracy_score(y_test,grid_predicitons_enc)
summary_table_2 = pd.DataFrame([lr_scores_enc.mean(),dtc_scores_enc.mean(),rfc_scores_enc.mean(),knn_scores_enc.mean(),svm_grid_score_enc],index=['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'SVM'], columns=['Accuracy Score'])
print("Summary Table for Section 2")
summary_table_2
plt.figure(figsize=(10,5))
sns.pointplot(summary_table_2.index,summary_table_2['Accuracy Score'])
overall_summary = pd.concat([summary_table_1_1,summary_table_1_2,summary_table_2],axis=1)
overall_summary.columns = ['Without Encoding, Hold-Out','Without Encoding, K-fold','With Encoding']
overall_summary
plt.figure(figsize=(10,5))
ax = sns.pointplot(overall_summary.index,overall_summary['Without Encoding, Hold-Out'],color='red')
ax = sns.pointplot(overall_summary.index,overall_summary['Without Encoding, K-fold'],color='green')
ax = sns.pointplot(overall_summary.index,overall_summary['With Encoding'],color='blue')
ax.legend(handles=ax.lines[::len(overall_summary)+1], labels=["Without Encoding, Hold-Out","Without Encoding, K-fold","With Encoding"])