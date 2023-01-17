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
import pandas as pd

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']

# load dataset

pima = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
pima.head()
Y=pima['Outcome']

X=pima.drop(['Outcome'], axis=1)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)
!pip install pydotplus
from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO  

from IPython.display import Image  

import pydotplus



dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True,feature_names = col_names,class_names=['0','1'], max_depth=3)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')

Image(graph.create_png())