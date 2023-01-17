import pandas as pd
import numpy as np
df1 = pd.read_csv('.../input/sensor_readings_2.csv', header = None)
df2 = pd.read_csv('.../input/sensor_readings_4.csv',header = None)

df1 = df1.drop(labels = 2,axis = 1)
df2 = df2.drop(labels = 4,axis = 1)

data1 = pd.DataFrame.as_matrix(df1)
data2 = pd.DataFrame.as_matrix(df2)

df = pd.read_csv('.../input/sensor_readings_24.csv',header = None)

df = df.replace('Move-Forward',0)
df = df.replace('Slight-Right-Turn',1)
df = df.replace('Sharp-Right-Turn',2)
df = df.replace('Slight-Left-Turn',3)

data3 = pd.DataFrame.as_matrix(df)

data = np.concatenate([data1,data2,data3],axis=1)
s = int(0.7*data.shape[0])

X_train = data[0:s,0:data.shape[1]-1]
X_test = data[s:data.shape[0],0:data.shape[1]-1]
y = data[0:s,data.shape[1]-1]

y_train = y.astype(int)
y_test = data[s:data.shape[0],data.shape[1]-1].astype(int)


from sklearn import linear_model
model = linear_model.LogisticRegression(penalty ='l2',max_iter=500,multi_class= 'ovr')

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

from sklearn import metrics
acc1 = metrics.accuracy_score(y_test,y_pred_test)
acc = metrics.accuracy_score(y_train,y_pred_train)

print ("Accuracy for train set:"),
print (acc)
print ("Accuracy for test set:"),
print (acc1)


from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X_train, y_train)

y_pred_test = classif.predict(X_test)
y_pred_train = classif.predict(X_train)

from sklearn import metrics
acc1 = metrics.accuracy_score(y_test,y_pred_test)
acc = metrics.accuracy_score(y_train,y_pred_train)

print ("Accuracy for train set:"),
print (acc)
print ("Accuracy for test set:"),
print (acc1)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred_test = gnb.predict(X_test)
y_pred_train = gnb.predict(X_train)

from sklearn import metrics
acc1 = metrics.accuracy_score(y_test,y_pred_test)
acc = metrics.accuracy_score(y_train,y_pred_train)

print ("Accuracy for train set:"),
print (acc)
print ("Accuracy for test set:"),
print (acc1)
