import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print(iris["DESCR"])
X = iris["data"]
y = iris["target"]
plt.scatter(X[:,0], X[:,1], c=y, cmap = plt.cm.Set2)
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
from sklearn.svm import SVC
model = SVC()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("SVC Classification Report")
print(classification_report(y_test, y_predict))
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("kMeans Classification Report")
print(classification_report(y_test, y_predict))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("LogisticRegression Classification Report")
print(classification_report(y_test, y_predict))