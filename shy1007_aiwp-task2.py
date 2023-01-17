import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, cross_validate

from sklearn import preprocessing, svm

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.cluster import KMeans

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
column_labels = ['age','workclass','fnlgwt','education','education_num','marital_status',

           'occupation','relationship','race','sex','capital_gain','capital_loss',

           'hours_week','native_country','income']

print("Total Labels = ",len(column_labels))
dataset = pd.read_csv("../input/income-data/income_data.csv",names=column_labels)

dataset.head()
dataset.shape
income = dataset["income"]

income
for i in range(len(income)):

  if income[i] == " <=50K":

    income[i] = 0

  else:

    income[i] = 1

dataset["income"] = income
for col in dataset.columns:

    if type(dataset[col][0]) == str:

        for i in range(len(dataset[col])):

          dataset[col][i] = dataset[col][i].strip()
dataset.replace(' ?', np.nan, inplace=True)
dataset = pd.concat([dataset, pd.get_dummies(dataset['workclass'],prefix='workclass',prefix_sep=':')], axis=1)

dataset.drop('workclass',axis=1,inplace=True)



dataset = pd.concat([dataset, pd.get_dummies(dataset['marital_status'],prefix='marital_status',prefix_sep=':')], axis=1)

dataset.drop('marital_status',axis=1,inplace=True)



dataset = pd.concat([dataset, pd.get_dummies(dataset['occupation'],prefix='occupation',prefix_sep=':')], axis=1)

dataset.drop('occupation',axis=1,inplace=True)



dataset = pd.concat([dataset, pd.get_dummies(dataset['relationship'],prefix='relationship',prefix_sep=':')], axis=1)

dataset.drop('relationship',axis=1,inplace=True)



dataset = pd.concat([dataset, pd.get_dummies(dataset['race'],prefix='race',prefix_sep=':')], axis=1)

dataset.drop('race',axis=1,inplace=True)



dataset = pd.concat([dataset, pd.get_dummies(dataset['sex'],prefix='sex',prefix_sep=':')], axis=1)

dataset.drop('sex',axis=1,inplace=True)



dataset = pd.concat([dataset, pd.get_dummies(dataset['native_country'],prefix='native_country',prefix_sep=':')], axis=1)

dataset.drop('native_country',axis=1,inplace=True)



dataset.drop('education', axis=1,inplace=True)



dataset.drop("fnlgwt",axis=1,inplace=True)



dataset.head()
dataset.shape
X = dataset.drop(['income'], 1)

X = preprocessing.scale(X)

y = dataset['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
temp = dataset.drop(["income"],axis=1)

temp.head()
linear_regrs = LinearRegression()

lm_model = linear_regrs.fit(X_train, y_train)

y_lm_pred = np.around(lm_model.predict(X_test))

print("Confusion Matrix for Linear Regression\n",confusion_matrix(y_test,y_lm_pred))

print("\nClassification Report for Linear Regression\n",classification_report(y_test,y_lm_pred))

lm_acc = accuracy_score(y_test,y_lm_pred)

print("\nAccuracy score of Linear Regression =",lm_acc)

print()
logis_regrs = LogisticRegression(penalty = 'l1', C = .001, solver="liblinear")

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
k_means = KMeans(n_clusters=2)

k_means.fit(X_train)

labels = k_means.labels_

y_k_means_label = k_means.predict(X_test)



print("Confusion Matrix for K-means\n",confusion_matrix(y_test,y_k_means_label))

print("\nClassification Report for K-means\n",classification_report(y_test,y_k_means_label))

k_mean_acc = accuracy_score(y_test, y_k_means_label)

print("\nAccuracy score of KNN =",k_mean_acc)

print()
svc = SVC(C = 0.01, kernel = "linear" )

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