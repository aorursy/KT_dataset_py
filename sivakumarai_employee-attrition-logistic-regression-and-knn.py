# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



emp_df = pd.read_csv("../input/HR-Employee-Attrition.csv")

emp_df1 = emp_df.copy()

emp_df1.head()
emp_df1.info()
emp_df1['Attrition'].replace(to_replace=['Yes'],value=1,inplace=True)

emp_df1['Attrition'].replace(to_replace=['No'],value=0,inplace=True)

emp_df1.head()

emp_df_duplicate = emp_df1[emp_df1.duplicated()]

emp_df_duplicate
emp_df1['Attrition'].value_counts()
emp_df1.corr()
x = emp_df1.drop('Attrition',axis=1)

y = emp_df1['Attrition']

x = pd.get_dummies(x)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=12)

model = LogisticRegression()

model.fit(X_train,Y_train)

print("Coeff:", model.coef_)

print("Intercept:", model.intercept_)
Y_train_predict = model.predict(X_train)

Y_test_predict = model.predict(X_test)

Y_train_predict_df = pd.DataFrame(Y_train_predict)

emp_df_ytrain = pd.concat([X_train,Y_train_predict_df],axis=1)

emp_df_ytrain.head()

results = confusion_matrix(Y_train,Y_train_predict)

print("confusion matrix - Train:\n", results)

print("accuracy score - Train:\n", accuracy_score(Y_train,Y_train_predict))

print("classifcation report - Train :\n",classification_report(Y_train,Y_train_predict))
results = confusion_matrix(Y_test,Y_test_predict)

print("confusion matrix - Test:\n", results)

#Accuracy score is to measure the accuracy of overall model")

print("Accuracy score - Test:", (results[0][0] + results[1][1]) / sum(sum(results)))

#Precision is to measure the correctly predicted positive from overall predicted positive.

#print("accuracy score - Test:\n", accuracy_score(Y_test,Y_test_predict))

print("precision (TP / TP + FP):", (results[0][0] / (results[0][0] + results[1][0])))

#Recall is to measure the correctly predicted positive from positive classifier

print("recall (TP / TP + FN):", (results[0][0] / (results[0][0] + results[0][1])))

print("classifcation report - Test :\n",classification_report(Y_test,Y_test_predict))

print("False Positive Rate - Test:",results[1][0] / (results[1][0] + results[1][1]))

print("True Positive Rate - Test:",results[0][0] / (results[0][0] + results[0][1]))
plt.rc("font",size=18)

auc = roc_auc_score(Y_test, model.predict(X_test))

print("Area Under Curve:",auc)

false_positive, true_positive, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:,1])

plt.figure(figsize=(10,6))

plt.plot(false_positive, true_positive, label='Logistic Regression (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.savefig('Log_ROC')

plt.show()

print(false_positive)

print(true_positive)

print(Y_test.shape)
for K in range(25):

    K_value = K+1

    knn_neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    knn_neigh.fit(X_train, Y_train) 

    y_test_pred = knn_neigh.predict(X_test)

    print ("Accuracy is ", accuracy_score(Y_test,y_test_pred) , "for k-value",K_value)