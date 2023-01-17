import numpy as np

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split

from sklearn import metrics

from IPython.display import SVG

from graphviz import Source

from IPython.display import display

from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv('../input/titanic/train.csv')

data.head()
data=data.dropna()
col=data.columns

col1=col[[2,5]]

# col2=col[4:12]

# col1=col1.append(col2)
x=data[col1]

x.head()
y=data.Survived
y.isna()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=None)

clf = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

y_score = clf.score(x,y)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
graph = Source(export_graphviz(clf, out_file=None,  

                filled=True, rounded=True,

                special_characters=True,feature_names = col1,class_names=['0','1','2']))



display(SVG(graph.pipe(format='svg')))
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
gnb = GaussianNB()

y_pred = gnb.fit(x_train, y_train).predict(x_test)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)