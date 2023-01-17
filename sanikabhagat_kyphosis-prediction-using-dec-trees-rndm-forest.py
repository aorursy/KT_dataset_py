import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
data=pd.read_csv('../input/kyphosis.csv')
data.head()
data.info()
data.shape
data.describe()
sns.pairplot(data,hue='Kyphosis',palette='Set1')
X = data.drop('Kyphosis',axis=1)

y = data['Kyphosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
# Instantiating Decision Tree model (basically creating a decision tree object)
dtree = DecisionTreeClassifier()
# Training or fitting the model on training data
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
features = list(data.columns[1:])

features
!pip install pydotplus
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True, feature_names = features,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('kyphosis.png')

Image(graph.create_png())
from sklearn.ensemble import RandomForestClassifier
# Instantiating Random Forest model (basically creating a random forest object)
rfc = RandomForestClassifier(n_estimators=200)
# Training or fitting the model on training data
rfc.fit(X_train,y_train)
rfc_predictions = rfc.predict(X_test)
print(classification_report(y_test,rfc_predictions))
print(confusion_matrix(y_test,rfc_predictions))