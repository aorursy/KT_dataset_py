# Import all modules
import numpy as np # linear algebra
import pandas as pd # Data exploration
import matplotlib.pyplot as plt # plotting
from sklearn.datasets import load_iris # dataset
from sklearn.tree import DecisionTreeClassifier , export_graphviz # Model and export
from sklearn.model_selection import train_test_split # train split
# Load data and exploration
iris = load_iris()
print("Shape:", iris.data.shape)
pd.DataFrame(iris.data, columns=iris.feature_names).describe()
# Iterate on each feature
graficas = []
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):    
    currentFeatures = [iris.feature_names[pair[0]], iris.feature_names[pair[1]]]
    X, X_test, y, y_test = train_test_split(iris.data[:, pair], iris.target, random_state=100)
    
    n_classes = np.unique(y).shape[0]

    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
                                 min_samples_leaf=1, max_features=None, random_state=None, max_leaf_nodes=None)
    clf.fit(X, y)

    print ("Precision:", clf.score(X_test, y_test))
    print ("Features importance:")
    for c, v in zip(currentFeatures, clf.feature_importances_):
        print (" "*4, c, "->", round(v,2))
    print ("***"*20)
    
    
    
    # tree graph
    with open("tree.dot", 'w') as f:
        export_graphviz(clf, out_file=f, 
                             feature_names=currentFeatures,    
                             filled=True, rounded=True)
    with open("tree.dot", 'r') as f:
        graficas.append(f.read())
    
    # decision limits graph
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")
    
    for i, color in zip(range(n_classes), "bry"):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

plt.legend()
plt.show()
for g in graficas:
    print (g)
    print ("*"*60)
# Visualize the graphs pasting the graph info in http://webgraphviz.com/    