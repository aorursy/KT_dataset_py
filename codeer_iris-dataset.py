import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = pd.read_csv('../input/iris/Iris.csv')
data.describe(include='all')
data.columns
data.head()
data.tail()
x = data.iloc[:,2:5].values
y = data.iloc[:,5:].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# 1 Linear Legression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) 
y_pred = logr.predict(X_test) 

cm = confusion_matrix(y_test,y_pred)
print('LR')
print(cm)
acc_Linear = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_Linear)
# 2. KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)
acc_KNN = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_KNN)
# 3. SVC (SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SupportVector')
print(cm)

acc_SVC = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_SVC)
# 4. NAive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)
acc_NaiB = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_NaiB)
# 5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DecisionTree')
print(cm)
acc_DT = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_DT)
# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RandomForest')
print(cm)
acc_RF = round(accuracy_score(y_pred, y_test) * 100, 2)
print("accuracy score is ",acc_RF)
