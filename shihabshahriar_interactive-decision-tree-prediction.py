import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeClassifier, plot_tree

from random import randint

import matplotlib.pyplot as plt
data = load_iris()

clf = DecisionTreeClassifier(max_depth=3).fit(data['data'],data['target'])
plt.figure(figsize=(10,8))

plot_tree(clf,feature_names=data['feature_names'],class_names=data['target_names'],filled=True);
tree = clf.tree_

node = 0      #Index of root node

while True:

    feat,thres = tree.feature[node],tree.threshold[node]

    print(feat,thres)

    v = randint(0,5) #Uncomment the line below, remove this line, it's only to Commit in kaggle

    print(v)

    #v = float(input(f"The value of {data['feature_names'][feat]}: "))

    if v<=thres:

        node = tree.children_left[node]

    else:

        node = tree.children_right[node]

    if tree.children_left[node] == tree.children_right[node]: #Check for leaf

        label = np.argmax(tree.value[node])

        print("We've reached a leaf")

        print(f"Predicted Label is: {data['target_names'][label]}")

        break