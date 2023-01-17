import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
wine=pd.read_csv('../input/winequality-red.csv')
wine.head()
wine.info()
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
sns.barplot(x = 'quality', y = 'citric acid', data = wine)
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
sns.barplot(x = 'quality', y = 'chlorides', data = wine)
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
sns.barplot(x = 'quality', y = 'density', data = wine)
sns.barplot(x = 'quality', y = 'pH', data = wine)
sns.barplot(x = 'quality', y = 'sulphates', data = wine)
sns.barplot(x = 'quality', y = 'alcohol', data = wine)
X = wine.drop('quality', axis = 1)

y = wine['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.head()
X_test.head()
y_train.head()
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
cnf=pd.DataFrame(lm.coef_,X_train.columns)

cnf
predictions=lm.predict(X_test)
y_test.head()
predictions
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=3)
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from IPython.display import Image  

from sklearn.externals.six import StringIO  

from sklearn.tree import export_graphviz

import pydot 



features = list(wine.columns[:-1])

features
dot_data = StringIO()  

export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)



graph = pydot.graph_from_dot_data(dot_data.getvalue())  

Image(graph[0].create_png(),width=5000000,height=100)  
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))