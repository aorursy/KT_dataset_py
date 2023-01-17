import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
column_labels = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]

data = pd.read_csv("../input/processed-cleveland/processed_cleveland.csv", names=column_labels)

data.head()
data.dtypes

total_missing_data = 0

for col in data.columns:

  total_missing_data += list(data[col].values).count("?")



print(f'total no. of missing data = {total_missing_data}')
data['thal'].replace('3.0',1,inplace=True)

data['thal'].replace('6.0',2,inplace=True)

data['thal'].replace('7.0',3,inplace=True)

data['thal'].replace('?',1,inplace=True)

data['ca'].replace('?',0.0,inplace=True)

data['ca'].replace('0.0',0.0,inplace=True)

data['ca'].replace('1.0',1.0,inplace=True)

data['ca'].replace('2.0',2.0,inplace=True)

data['ca'].replace('3.0',3.0,inplace=True)



data.dtypes
data.nunique()
total_missing_data = 0

for col in data.columns:

  total_missing_data += list(data[col].values).count("?")



print(f'total no. of missing data = {total_missing_data}')
y = data["target"]

X = data.drop("target",axis=1)

from sklearn.model_selection import train_test_split, cross_validate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
linear_regrs = LinearRegression()

lm_model = linear_regrs.fit(X_train, y_train)

y_lm_pred = np.around(lm_model.predict(X_test))

print("Confusion Matrix for Linear Regression\n",confusion_matrix(y_test,y_lm_pred))

print("\nClassification Report for Linear Regression\n",classification_report(y_test,y_lm_pred))

lm_acc = accuracy_score(y_test,y_lm_pred)

print("\nAccuracy score of Linear Regression =",lm_acc)

print()
logis_regrs = LogisticRegression(penalty = 'l1', C = .01, solver="liblinear")

lo_model = logis_regrs.fit(X_train, y_train)

y_lo_pred = np.around(lo_model.predict(X_test))

print("Confusion Matrix for Logistic Regression\n",confusion_matrix(y_test,y_lo_pred))

print("\nClassification Report for Logistic Regression\n",classification_report(y_test,y_lo_pred))

lo_acc = accuracy_score(y_test,y_lo_pred)

print("\nAccuracy score of Logistic Regression =",lo_acc)

print()
knn = KNeighborsClassifier()

knn_model = knn.fit(X_train, y_train)

y_knn_pred = np.around(knn_model.predict(X_test))

print("Confusion Matrix for KNN\n",confusion_matrix(y_test,y_knn_pred))

print("\nClassification Report for KNN\n",classification_report(y_test,y_knn_pred))

knn_acc = accuracy_score(y_test,y_knn_pred)

print("\nAccuracy score of KNN =",knn_acc)

print()
svc = SVC(C = 0.1, kernel = "linear" )

svc.fit( X_train, y_train )

y_svc_pred = svc.predict( X_test )

print("Confusion Matrix for SVM\n",confusion_matrix(y_test,y_svc_pred))

print("\nClassification Report for SVM\n",classification_report(y_test,y_svc_pred))

svc_acc = accuracy_score(y_test, y_svc_pred)

print("\nAccuracy score of SVM =",svc_acc)
dt = DecisionTreeClassifier()

dt_model = dt.fit(X_train, y_train)

y_dt_pred = dt_model.predict(X_test)

print("Confusion Matrix for Decision Tree\n",confusion_matrix(y_test,y_dt_pred))

print("\nClassification Report for Decision Tree\n",classification_report(y_test,y_dt_pred))

dt_acc = accuracy_score(y_test,y_dt_pred)

print("\nAccuracy score of Decision Tree =",dt_acc)

print()
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

fig = plt.figure(figsize=(16,5))

ax = fig.add_axes([0,0,1,1])

model_labels = ["Linear Regression", "Logistic Regression", "K-nearest neighbours", "Support Vector Classifier", "Decision Tree"]

accuracies = np.array([lm_acc,lo_acc,knn_acc,svc_acc,dt_acc])*100

ax.bar(model_labels,accuracies,color = colors)

ax.set_xlabel('Algorithms')

ax.set_ylabel('Accuracy %')

plt.yticks(np.arange(0,100,10))

plt.show()
print("Following are the accuracy of different models")

ac_df = pd.DataFrame()

ac_df["Algorithms"] = model_labels

ac_df["Accuracy"] = accuracies

ac_df