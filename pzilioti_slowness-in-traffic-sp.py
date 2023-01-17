#Dataset:

#   https://archive.ics.uci.edu/ml/datasets/Behavior+of+the+urban+traffic+of+the+city+of+Sao+Paulo+in+Brazil
!pip install pydotplus
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv", sep=";", decimal=",")



print(data.dtypes)

data.head()
data.apply(lambda x: x.isnull().value_counts())
y = data["Slowness in traffic (%)"]

X = data.drop("Slowness in traffic (%)", axis = 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.tree import DecisionTreeRegressor



dt_model = DecisionTreeRegressor(random_state=100)

dt_model.fit(X_train,y_train)



y_pred = dt_model.predict(X_test)



print("Decision Tree Score:",dt_model.score(X_test, y_test))
plt.figure()

plt.scatter(y_test, y_pred, s=20, edgecolor="black",

            c="darkorange", label="data")

plt.xlabel("test")

plt.ylabel("pred")

#plt.title("Decision Tree Regression")

#plt.legend()

#plt.show()
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(dt_model, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
dt_model2 = DecisionTreeRegressor(random_state=100, criterion='friedman_mse', splitter='random', max_depth=3, min_samples_split=4, min_samples_leaf=2)

dt_model2.fit(X_train,y_train)



y_pred2 = dt_model2.predict(X_test)



print("Decision Tree Score:",dt_model2.score(X_test, y_test))
plt.figure()

plt.scatter(y_test, y_pred2, s=20, edgecolor="black",

            c="darkorange", label="data")

plt.xlabel("test")

plt.ylabel("pred")
fig, axs = plt.subplots(1, 1, figsize=(20, 10))

tree.plot_tree(dt_model2, filled=True, feature_names =X.columns)

plt.show()