# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
iris = pd.read_csv('../input/iris/Iris.csv')
iris.info()
Iris_PreProcess=iris.drop('Id',axis=1)
Iris_PreProcess.info()
iris['Species'].unique()
Iris_PreProcess.columns
Iris_data=Iris_PreProcess[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']].copy()
Iris_data.head()
Iris_PreProcess.Species=pd.get_dummies(Iris_PreProcess.Species)
Iris_target=Iris_PreProcess['Species']
Iris_data_final=Iris_data.drop('Species',axis=1)
Iris_data_final
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
print(knn)
knn.fit(Iris_data_final,Iris_target)
new_x = [3 ,5 ,4 ,2]
knn.predict([new_x])
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(iris, test_size=0.2, random_state=42)
train_X, train_y = train_set.drop('Species', axis=1), train_set['Species']

test_X, test_y   = test_set.drop('Species', axis=1), test_set['Species']
import matplotlib.pyplot as plt



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.image as mpimg
prep_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler()),

])



train_X_prep = prep_pipeline.fit_transform(train_X)
tree_clf = DecisionTreeClassifier(max_depth=2)



tree_clf.fit(train_X, train_y)
predictions = tree_clf.predict(train_X)

target = train_y
acc = accuracy_score(predictions, target)

acc
conf_mx = confusion_matrix(target, predictions)

conf_mx
from sklearn.tree import export_graphviz



export_graphviz( tree_clf,

                out_file="iris_tree_train.dot", 

                feature_names=iris.columns[:-1], 

                class_names=iris['Species'].unique(), rounded=True,

                filled=True

        )
from sklearn import tree

tree.plot_tree(tree_clf)