import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # sklearn lib for decision tree
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
clf=DecisionTreeClassifier() 
clf.fit(x_train,y_train)
print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(x_test, y_test)))
clf2 = DecisionTreeClassifier(max_depth=4, random_state=0)  # Notice max_depth parameter is set to 4 
clf2.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(clf2.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf2.score(x_test, y_test)))
from sklearn import tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf2,class_names=['Malignant','Benign'])
def plot_feature_importances_cancer(model):
    plt.figure(figsize=(10,8))
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances_cancer(tree)