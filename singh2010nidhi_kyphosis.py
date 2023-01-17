import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/kyphosis-dataset/kyphosis.csv')
df.head()
df.info()
sns.pairplot(df,hue="Kyphosis")
from sklearn.model_selection import train_test_split
X = df.drop("Kyphosis",axis=1)
y = df["Kyphosis"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dt = dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rf = rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
from sklearn import tree

plt.figure(figsize=(20,20))

tree.plot_tree(dt)

plt.show()