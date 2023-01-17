# Importing libraries



import pandas as pd

import numpy as np
# Importing dataset 

data =pd.read_csv('../input/Website Phishing.csv')

x = data.iloc[:, :-1]

y = data.iloc[:, : 1]

z = data.iloc[:, : 0]
x.head()
y.head()
# Normalization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

xScaler = scaler.fit_transform(x)
# Holdout

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(xScaler,y, test_size = 0.4)
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
k =1

knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_predict, cross_val_score
score = cross_val_score(knn, xScaler, y, cv = 8)

print(score)
y_pred = cross_val_predict(knn, xScaler, y, cv = 10)

conf_mat = metrics.confusion_matrix(y , y_pred)

print(conf_mat)
f1 = metrics.f1_score(y,y_pred,average="weighted")

print(f1)
acc = metrics.accuracy_score(y, y_pred)

print(acc)