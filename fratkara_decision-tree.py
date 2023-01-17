# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO 

from IPython.display import Image 

from pydot import graph_from_dot_data

import pandas as pd

import numpy as np
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)

y = pd.Categorical.from_codes(iris.target, iris.target_names)
X.head()
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)
dot_data = StringIO()

export_graphviz(dt, out_file=dot_data, feature_names=iris.feature_names)

(graph, ) = graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())
y_pred = dt.predict(X_test)
species = np.array(y_test).argmax(axis=1)

predictions = np.array(y_pred).argmax(axis=1)

cm=confusion_matrix(species, predictions)