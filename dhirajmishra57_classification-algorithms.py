import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,accuracy_score,roc_auc_score,roc_curve

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

import numpy as np

%matplotlib inline

from sklearn.model_selection import train_test_split

plt.style.use('seaborn-colorblind')



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/bank-additional-full.csv",sep=';')

data =  df.copy()

df.head()
X_label = LabelEncoder()



df['job'] = X_label.fit_transform(df['job'])

df['marital'] = X_label.fit_transform(df['marital'])

df['education'] = X_label.fit_transform(df['education'])

df['default'] = X_label.fit_transform(df['default'])

df['housing'] = X_label.fit_transform(df['housing'])

df['loan'] = X_label.fit_transform(df['loan'])

df['contact'] = X_label.fit_transform(df['contact'])

df['month'] = X_label.fit_transform(df['month'])

df['day_of_week'] = X_label.fit_transform(df['day_of_week'])

df['poutcome'] = X_label.fit_transform(df['poutcome'])

df.head()


y = (pd.get_dummies(data['y'], columns = ['y'], prefix = 'y', drop_first = True)).values

type(y)

y[:5]

y = np.ravel(y)

y[:4]
df.drop(columns=['y'],inplace=True)

df.head()
df.dtypes
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.3,random_state=42)
print('Shape of Training set : ' , [X_train.shape,y_train.shape])

print('Shape of Validation set : ' , [X_valid.shape,y_valid.shape])
log_clf = LogisticRegression().fit(X_train,y_train)



log_pred = log_clf.predict(X_valid)



print('Training score : ' , log_clf.score(X_train,y_train))

Training_score_log = log_clf.score(X_train,y_train)

 

print('Validation score : ' , round(log_clf.score(X_valid,y_valid),2))

Validation_score_log = log_clf.score(X_valid,y_valid)



print("Accuracy:",accuracy_score(y_valid, log_pred))

Accuracy_log = accuracy_score(y_valid, log_pred)

print("Precision:",precision_score(y_valid, log_pred))

Precision_log = precision_score(y_valid, log_pred)

print("Recall:",recall_score(y_valid, log_pred))

Recall_log = recall_score(y_valid, log_pred)
CONFMTX = confusion_matrix(y_valid,log_pred)

CONFMTX
%matplotlib inline

y_pred_proba  = log_clf.predict_proba(X_valid)[::,1]

FPR,TPR,threshold = roc_curve(y_valid,y_pred_proba)

auc_log = roc_auc_score(y_valid,y_pred_proba)

plt.plot([0,1],[0,1],'o--');

plt.plot(FPR,TPR,label='data 1, auc='+str(auc_log));

plt.show()
knn_clf = KNeighborsClassifier(n_neighbors=20).fit(X_train,y_train)



knn_pred = knn_clf.predict(X_valid)



print('Training score : ' , knn_clf.score(X_train,y_train))

Training_score_knn = knn_clf.score(X_train,y_train)



print('Validation score : ' ,(knn_clf.score(X_valid,y_valid)))

Validation_score_knn = knn_clf.score(X_valid,y_valid)



print("Accuracy:",accuracy_score(y_valid, knn_pred))

Accuracy_knn = accuracy_score(y_valid, knn_pred)



print("Precision:",precision_score(y_valid, knn_pred))

Precision_knn = precision_score(y_valid, knn_pred)



print("Recall:",recall_score(y_valid, knn_pred))

Recall_knn = recall_score(y_valid, knn_pred)
y_pred_proba  = knn_clf.predict_proba(X_valid)[::,1]

auc_knn = roc_auc_score(y_valid,y_pred_proba)
tree_clf = DecisionTreeClassifier().fit(X_train,y_train)



tree_pred = tree_clf.predict(X_valid)



print('Training score : ' , tree_clf.score(X_train,y_train))

Training_score_tree = tree_clf.score(X_train,y_train)



print('Validation score : ' ,(tree_clf.score(X_valid,y_valid)))

Validation_score_tree = tree_clf.score(X_valid,y_valid)



print("Accuracy:",accuracy_score(y_valid, tree_pred))

Accuracy_tree = accuracy_score(y_valid, tree_pred)



print("Precision:",precision_score(y_valid, tree_pred))

Precision_tree = precision_score(y_valid, tree_pred)



print("Recall:",recall_score(y_valid, tree_pred))

Recall_tree = recall_score(y_valid, tree_pred)



y_pred_proba  = tree_clf.predict_proba(X_valid)[::,1]

auc_tree = roc_auc_score(y_valid,y_pred_proba)
rf_clf = RandomForestClassifier().fit(X_train,y_train)



rf_pred = rf_clf.predict(X_valid)



print('Training score : ' , rf_clf.score(X_train,y_train))

Training_score_rf = rf_clf.score(X_train,y_train)



print('Validation score : ' ,(rf_clf.score(X_valid,y_valid)))

Validation_score_rf = rf_clf.score(X_valid,y_valid)



print("Accuracy:",accuracy_score(y_valid, rf_pred))

Accuracy_rf = accuracy_score(y_valid, rf_pred)



print("Precision:",precision_score(y_valid, rf_pred))

Precision_rf = precision_score(y_valid, rf_pred)



print("Recall:",recall_score(y_valid, rf_pred))

Recall_rf = recall_score(y_valid, rf_pred)



y_pred_proba  = rf_clf.predict_proba(X_valid)[::,1]

auc_rf = roc_auc_score(y_valid,y_pred_proba)
#lets create a dataframe

Models= pd.DataFrame({'Training_Score':[Training_score_log,Training_score_knn,Training_score_tree,Training_score_rf]

,'Validation_score' : [Validation_score_log,Validation_score_knn,Validation_score_tree,Validation_score_rf]

,'Accuracy' : [Accuracy_log,Accuracy_knn,Accuracy_tree,Accuracy_rf]

,'Precision' : [Precision_log,Precision_knn,Precision_tree,Precision_rf]

,'Recall' : [Recall_log,Recall_knn,Recall_tree,Recall_rf]

,'AUC' : [auc_log,auc_knn,auc_tree,auc_rf]},index=['Logistic Regression','KNN','Decision Tree','Random Forest'])

Models