#loading dataset

import pandas as pd

import numpy as np

#visualisation

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# data preprocessing

from sklearn.preprocessing import StandardScaler

# data splitting

from sklearn.model_selection import train_test_split

# data modeling

from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

data.head()
data.info()
y = data["Outcome"]

X = data.drop('Outcome',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
m1 = 'Logistic Regression'

lr = LogisticRegression()

model = lr.fit(X_train, y_train)

lr_predict = lr.predict(X_test)

lr_conf_matrix = confusion_matrix(y_test, lr_predict)

lr_acc_score = accuracy_score(y_test, lr_predict)

print("confussion matrix")

print(lr_conf_matrix)

print("\n")

print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')

print(classification_report(y_test,lr_predict))
m3 = 'Random Forest Classfier'

rf = RandomForestClassifier(n_estimators=20, random_state=12,max_depth=5)

rf.fit(X_train,y_train)

rf_predicted = rf.predict(X_test)

rf_conf_matrix = confusion_matrix(y_test, rf_predicted)

rf_acc_score = accuracy_score(y_test, rf_predicted)

print("confussion matrix")

print(rf_conf_matrix)

print("\n")

print("Accuracy of Random Forest:",rf_acc_score*100,'\n')

print(classification_report(y_test,rf_predicted))
m7 = 'Support Vector Classifier'

svc =  SVC(kernel='rbf', C=2)

svc.fit(X_train, y_train)

svc_predicted = svc.predict(X_test)

svc_conf_matrix = confusion_matrix(y_test, svc_predicted)

svc_acc_score = accuracy_score(y_test, svc_predicted)

print("confussion matrix")

print(svc_conf_matrix)

print("\n")

print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')

print(classification_report(y_test,svc_predicted))
model_ev = pd.DataFrame({'Model': ['Logistic Regression','Random Forest','Support Vector Machine'], 'Accuracy': [lr_acc_score*100,

                    rf_acc_score*100,svc_acc_score*100]})

model_ev
colors = ['red','green','blue','gold','silver','yellow','orange',]

plt.figure(figsize=(12,5))

plt.title("barplot Represent Accuracy of different models")

plt.xlabel("Accuracy %")

plt.ylabel("Algorithms")

plt.bar(model_ev['Model'],model_ev['Accuracy'],color = colors)

plt.show()