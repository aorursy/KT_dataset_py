# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# getting the data
ccf = "../input/creditcard.csv"
df = pd.read_csv(ccf, nrows=1000)
df.head()

# setting up the random forest
from sklearn.ensemble import RandomForestRegressor
nestimators = 10000
rf = RandomForestRegressor(n_estimators = nestimators, random_state = 42)
# creating train labels and features
train_labels = np.array(df["Class"])
train_features = np.array(df.drop("Class", 1))

# fitting the model
rf.fit(train_features, train_labels);
# creating test labels and features
df_test = pd.read_csv(ccf, skiprows=range(1, 1000))
df_test.head()
test_labels = np.array(df_test["Class"])
test_features = np.array(df_test.drop("Class", 1))
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
# printing Mean Absolute Error
print('MAE:', round(np.mean(errors), 2))

# plotting some trees from the forest
from IPython.display import Image, display

def viewPydot(pdot):
    plt = Image(pdot.create_png())
    display(plt)
    
from sklearn.tree import export_graphviz\

import pydot
startestimator = 800
endestimator = 810
for i in range(startestimator, endestimator):    
    tree = rf.estimators_[i]
    export_graphviz(tree, out_file = 'tree.dot', feature_names = list(df.drop("Class", 1).columns), rounded = True, precision = 2)
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    viewPydot(graph)
    png_str = graph.create_png(prog='dot')

# feature importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
X = df.drop("Class", 1)
for f in range(X.shape[1]):
    print("%s. feature %d (%f)" % (X.columns[f], indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(20, 6))
plt.title("Feature importances")
plt.bar(X.columns, importances[indices],
       color="r", align="center")
plt.show()