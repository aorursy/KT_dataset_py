

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))

heart1=pd.read_csv('../input/heart.csv')
heart1.info()
heart1.describe()
heart1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(heart1.drop('target',axis=1))

scaled_features = scaler.transform(heart1.drop('target',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=heart1.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,heart1['target'], test_size=0.30)

X_train

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))

error_rate = []

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')    







knn = KNeighborsClassifier(n_neighbors=6)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=6')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
# Checking the performance od Decision Trees on the same Data set 
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)

cm

print(classification_report(y_test,pred))
# Checking the performance of Random forests on the same Data set 
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,rfc_pred)
cm
print(classification_report(y_test,pred))
# Working with SVM on the same dataset
from sklearn.svm import SVC

classifier = SVC()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test,pred))