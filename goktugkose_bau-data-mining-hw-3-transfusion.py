import numpy as np 

import pandas as pd 

import os

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,r2_score,precision_score,recall_score,f1_score

print(os.listdir("../input"))
df_trans = pd.read_csv("../input/transfusion.data")
df_trans.head()
print('There are', df_trans.shape[0], 'rows and', df_trans.shape[1], 'columns in the dataset.')
X = df_trans.iloc[:,:-1]

y = df_trans.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)
dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)

dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred ))

print('Precision: %.3f' % precision_score(y_test,y_pred ,average='micro'))

print('Recall: %.3f' % recall_score(y_test,y_pred ,average='micro'))

print('f1_score: %.3f' % f1_score(y_test,y_pred ,average='micro'))

print("Confusion Matrix:\n",cm)
lr = LogisticRegression(solver='liblinear')

lr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)
cm = confusion_matrix(y_test,y_pred_lr)

print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_lr ))

print('Precision: %.3f' % precision_score(y_test,y_pred_lr ,average='micro'))

print('Recall: %.3f' % recall_score(y_test,y_pred_lr ,average='micro'))

print('f1_score: %.3f' % f1_score(y_test,y_pred_lr ,average='micro'))

print("Confusion Matrix:\n",cm)