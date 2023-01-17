# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data

y = iris.target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=3)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_predict = clf.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,target_names=iris.target_names))
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))

plot_tree(clf,feature_names=iris.feature_names, class_names=iris.target_names, filled = True)
!pip install dtreeviz
from dtreeviz.trees import dtreeviz
viz = dtreeviz(

    clf,

    X_train, 

    y_train,

    target_name="Species",

    feature_names=iris.feature_names,

    class_names=list(iris.target_names),

    scale=2

) 

viz
from yellowbrick.model_selection import FeatureImportances

fi_viz = FeatureImportances(clf, labels=iris.feature_names)

fi_viz.fit(X_test, y_test)

fi_viz.show()
from yellowbrick.classifier import ConfusionMatrix

iris_cm = ConfusionMatrix(

    clf, classes=iris.target_names,

    label_encoder={0: 'setosa', 1: 'versicolor', 2: 'virginica'}

)

iris_cm.score(X_test, y_test)

iris_cm.show()

y_probas = clf.predict_proba(X_test)
!pip install scikit-plot
import matplotlib.pyplot as plt

import scikitplot as skplt

skplt.metrics.plot_roc(y_test, y_probas)

plt.show()
skplt.metrics.plot_precision_recall_curve(y_test, y_probas)

plt.show()
skplt.estimators.plot_learning_curve(clf, X, y)

plt.show()