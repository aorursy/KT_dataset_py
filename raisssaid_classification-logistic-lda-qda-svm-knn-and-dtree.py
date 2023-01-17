import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import metrics
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Loading data
data = pd.read_csv("../input/iris/Iris.csv")
data.info()
data.sample(5)
# Drop Id column
data.drop(['Id'], axis=1, inplace=True)
# Iris types
data['Species'].unique()
# The dataset is balanced
classes = data['Species']
ax = sns.countplot(x=classes, data=data)
sns.pairplot(data, hue='Species')
# Correlation between variables
corr = data.corr()
sns.heatmap(corr, annot= True);
# How length and width vary by species?
plt.figure(figsize=(12,7))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=data)
X = data.drop(['Species'], axis=1)
y = data['Species'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=22
    )
classes_test = pd.DataFrame(y_test.reshape(-1, 1))
classes_test[0].value_counts()
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train.ravel())
log_reg_pred = log_reg.predict(X_test)
metrics.accuracy_score(log_reg_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, log_reg_pred)
confusion_matrix
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train.ravel())
lda_pred = lda.predict(X_test)
metrics.accuracy_score(lda_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, lda_pred)
confusion_matrix
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train.ravel())
qda_pred = qda.predict(X_test)
metrics.accuracy_score(lda_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, qda_pred)
confusion_matrix
svc = svm.SVC()
svc.fit(X_train, y_train.ravel())
svc_pred = svc.predict(X_test)
metrics.accuracy_score(svc_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, svc_pred)
confusion_matrix
# Choose k
a_index=list(range(1,15))
a=pd.Series(dtype='float64')
x=[1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14]
for i in list(range(1,15)):
    KNNmodel=KNeighborsClassifier(n_neighbors=i) 
    KNNmodel.fit(X_train,y_train.ravel())
    KNNprediction=KNNmodel.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(KNNprediction,y_test)))
plt.plot(a_index, a)
plt.xticks(x)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train.ravel())
knn_pred = knn.predict(X_test)
metrics.accuracy_score(knn_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, svc_pred)
confusion_matrix
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train.ravel())
dtc_pred = dtc.predict(X_test)
metrics.accuracy_score(dtc_pred, y_test)
confusion_matrix = metrics.confusion_matrix(y_test, dtc_pred)
confusion_matrix