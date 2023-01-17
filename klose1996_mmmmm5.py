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
import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

import graphviz

data = pd.read_excel("../input/cooknum/cook.xlsx")

X = data.iloc[:, 1:4]

y = data.iloc[:,-1]

tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42,criterion='entropy')

tree_clf.fit(X, y)

export_graphviz(tree_clf,out_file='tree.dot',class_names=['NO','YES'],feature_names=['温度','口味','尺寸'],impurity=False,rounded=True,filled=True)

with open('tree.dot') as f:

    dot_graph=f.read()

graphviz.Source(dot_graph)