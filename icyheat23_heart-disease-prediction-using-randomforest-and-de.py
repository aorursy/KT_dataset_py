import pandas as pd

import numpy as np
df = pd.read_csv('../input/heart.csv')
df.head()
df.info()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.drop('target',axis=1),df['target'],test_size=0.3,random_state=101)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
pred_tree = dtree.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix

print(classification_report(y_test,pred_tree))

print("\n")

print(confusion_matrix(y_test,pred_tree))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=400)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
from sklearn.metrics import classification_report , confusion_matrix

print(classification_report(y_test,pred))

print("\n")

print(confusion_matrix(y_test,pred))