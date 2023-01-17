import numpy as np

import pandas as pd

df=pd.read_csv('../input/dataset.csv', na_values = "?")

df.dropna(inplace=True)

df.index = np.arange(1, len(df)+1)
df
df.corr()
df1=df.drop(['date','time','username','wrist'],axis=1)

df1
import sklearn

from sklearn import metrics

# import metrics we'll need

from sklearn.metrics import accuracy_score  

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve 

from sklearn.metrics import auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

data = df1.values

X = data[:, 1:]  

y = data[:, 0]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier



# instantiate the estimator

knn = KNeighborsClassifier()



# fit the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# accuracy score

pred_knn = metrics.accuracy_score(y_test, y_pred)

print ("Accuracy for KNN: {}".format(pred_knn))
from sklearn.svm import SVC



# instantiate the estimator

svm = SVC()



# fit the model

svm.fit(X_train, y_train)



# predict the response

y_pred = svm.predict(X_test)



# accuracy score

pred_sv = metrics.accuracy_score(y_test, y_pred)

print ("Accuracy for SVM: {}".format(pred_sv))
from sklearn.ensemble import RandomForestClassifier



clf=RandomForestClassifier(random_state=1)

clf.fit(X_train,y_train)
actual=y_test

predictions=clf.predict(X_test)
clf.score(X_test, y_test)
confusion_matrix(actual,predictions)

from sklearn.naive_bayes import GaussianNB



# instantiate the estimator

nb = GaussianNB()



# fit the model

nb.fit(X_train, y_train)



# predict the response

y_pred = nb.predict(X_test)



# accuracy score

pred_nb = metrics.accuracy_score(y_test, y_pred)

print ("Accuracy for Gaussian Naive Bayes: {}".format(pred_nb))