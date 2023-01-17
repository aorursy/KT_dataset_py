import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
iris = pd.read_csv('../input/iris/Iris.csv')
iris.head()
sns.pairplot(iris,hue='Species')
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(iris.drop('Species',axis=1),iris['Species'],test_size=0.33)
dt.fit(X_train,y_train)
pred = dt.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
# Almost perfect result because of small dataset. Let's try random forest method now.
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
pred2 = rfc.predict(X_test)
print(confusion_matrix(y_test,pred2))

print('\n')

print(classification_report(y_test,pred2))
# The Random Forests method was able to classify perfectly.