# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO 

from sklearn.preprocessing import LabelEncoder

from pydot import graph_from_dot_data

from sklearn import metrics

from IPython.display import Image  

import pydotplus



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/carseats/carseats.csv')

data.head(6)
data.isnull().sum()
lb_make = LabelEncoder()

df=data

df["ShelveLoc"] = lb_make.fit_transform(data["ShelveLoc"])

df["Urban"] = lb_make.fit_transform(data["Urban"])

df.head(6)
X = df.drop('US', axis=1)

y = df['US']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Metrics

print("Classification Report:\n",metrics.classification_report(y_test, y_pred))

print("Cinfusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))
feature_cols = X.columns



dot_data = StringIO()

export_graphviz(dt, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = feature_cols,class_names=['No','Yes'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('Cars.png')

Image(graph.create_png())
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True, feature_names = feature_cols,class_names=['No','Yes'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('Cars1.png')

Image(graph.create_png())
#Metrics

print("Classification Report:\n",metrics.classification_report(y_test, y_pred))

print("Cinfusion Matrix:\n",metrics.confusion_matrix(y_test, y_pred))