# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
haberman=pd.read_csv('../input/habermans-survival-data-set/haberman.csv')
haberman.head()
col_names = ['age', 'year', 'nodes', 'survival_status']

haberman.columns = col_names

col_names
haberman.head()
y = haberman['survival_status']
X = haberman.drop(['survival_status'], axis=1)
from sklearn.model_selection import train_test_split
#creating the train and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 101, stratify=y, test_size=0.25)
# distribution in training set
y_train.value_counts(normalize=True)
# distribution in validation set
y_valid.value_counts(normalize=True)
#shape of training set
X_train.shape, y_train.shape
#shape of validation set
X_valid.shape, y_valid.shape
#importing decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
# how to import decision tree regressor
from sklearn.tree import DecisionTreeRegressor
#creating the decision tree function
dt_model = DecisionTreeClassifier()
#fitting the model
dt_model.fit(X_train, y_train)
#checking the training score
dt_model.score(X_train, y_train)
#checking the validation score
dt_model.score(X_valid, y_valid)
#predictions on validation set
dt_model.predict(X_valid)
y_pred = dt_model.predict_proba(X_valid)[:,1]
y_new = []
for i in range(len(y_pred)):
    if y_pred[i]<=0.7:
        y_new.append(0)
    else:
        y_new.append(1)
from sklearn.metrics import accuracy_score
accuracy_score(y_valid, y_new)
train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    dt_model= DecisionTreeClassifier(max_depth=depth)
    dt_model.fit(X_train, y_train)
    train_accuracy.append(dt_model.score(X_train, y_train))
    validation_accuracy.append(dt_model.score(X_valid, y_valid))
frame = pd.DataFrame({'max_depth':range(1,10), 'train_acc':train_accuracy, 'valid_acc':validation_accuracy})
frame.head()
dt_model = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=25, random_state=10)

#importing decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
#fitting the model
dt_model.fit(X_train, y_train)

plt.figure(figsize=(12,8))

from sklearn import tree
import graphviz 
dot_data = tree.export_graphviz(dt_model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("dt_model")
dot_data = tree.export_graphviz(dt_model, out_file=None,   
                              filled=True, rounded=True,  
                              special_characters=True)

graph = graphviz.Source(dot_data) 

graph 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_valid, y_new))
print(classification_report(y_valid, y_new))