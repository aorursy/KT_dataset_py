from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,

                                                 stratify=cancer.target,random_state=42)

d_tree = DecisionTreeClassifier(random_state=0)

d_tree.fit(X_train,y_train)
print("Training Set Accuracy: {}".format(d_tree.score(X_train,y_train)))

print("Test Set Accuracy: {}".format(d_tree.score(X_test,y_test)))
d_tree_1 = DecisionTreeClassifier(max_depth=4,random_state=0)

d_tree_1.fit(X_train,y_train)
print("Training Set Accuracy: {}".format(d_tree_1.score(X_train,y_train)))

print("Test Set Accuracy: {}".format(d_tree_1.score(X_test,y_test)))
from sklearn.tree import export_graphviz

export_graphviz(d_tree_1,out_file = 'tree.dot',class_names=["malignant","benign"],

                feature_names=cancer.feature_names,impurity=False,filled=True)
import graphviz

with open ("tree.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)
print("Feature Importances:\n{}".format(d_tree_1.feature_importances_))
import matplotlib.pyplot as plt

import numpy as np

plt.figure(figsize=(15,10))

def plot_feature_importances_cancer(model):

    n_features = cancer.data.shape[1]

    plt.barh(range(n_features),model.feature_importances_,align='center')

    plt.yticks(np.arange(n_features),cancer.feature_names)

    plt.xlabel("Feature Importance")

    plt.ylabel("Feature")

plot_feature_importances_cancer(d_tree_1)
!pip install mglearn
import mglearn

tree = mglearn.plots.plot_tree_not_monotone()