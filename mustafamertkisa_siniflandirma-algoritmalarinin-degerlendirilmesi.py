#kutuphaneler

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#veri onisleme

#veri yukleme



veriler = pd.read_excel("../input/irisxls/iris.xls")



x = veriler.iloc[:,1:4].values #bağımsız değişkenler

y = veriler.iloc[:,4:].values #bağımlı değişken



#verilerin egitim ve test icin bolunmesi



from sklearn.model_selection import train_test_split



x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)



#verilerin olceklenmesi



from sklearn.preprocessing import StandardScaler



sc=StandardScaler()



X_train = sc.fit_transform(x_train)

X_test = sc.transform(x_test)
#lojistik regresyon



from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)

logr.fit(X_train,y_train)



y_pred = logr.predict(X_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



print("LR")

print(cm)
#en yakın komşu



from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train,y_train)



y_pred = knn.predict(X_test)



cm = confusion_matrix(y_test,y_pred)



print("KNN")

print(cm)
#destek vektör



from sklearn.svm import SVC

svc = SVC(kernel='rbf')

svc.fit(X_train,y_train)



y_pred = svc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)



print("SVC")

print(cm)
#gaus



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)



y_pred = gnb.predict(X_test)



cm = confusion_matrix(y_test,y_pred)



print("GNB")

print(cm)
#karar agaci



from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion = 'entropy')



dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)



cm = confusion_matrix(y_test,y_pred)



print("DTC")

print(cm)
#rassal ormanlar



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')

rfc.fit(X_train,y_train)



y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)



print("RFC")

print(cm)
#ROC , TPR, FPR değerleri 



y_proba = rfc.predict_proba(X_test)

print(y_proba[:,0])



from sklearn import metrics

fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')

print(fpr)

print(tpr)