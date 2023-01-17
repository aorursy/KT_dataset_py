# Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

# SciKit Learn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, train_test_split

# For visualizing the tree
from graphviz import Source
from IPython.display import SVG
data = pd.read_csv("../input/Iris.csv", index_col="Id")
data.head()
iris_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
iris_targets = "Species"
iris_train, iris_test = train_test_split(data, test_size=0.2) # We want a 80%/20% split for training/testing
print(f"Train: {len(iris_train)} rows\nTest: {len(iris_test)} rows")
iris_classifier = DecisionTreeClassifier(random_state=0)
model = iris_classifier.fit(X=iris_train[iris_features], y=iris_train[iris_targets])
print(model)
model.score(X=iris_test[iris_features], y=iris_test[iris_targets])
graph = Source(export_graphviz(model, out_file=None, feature_names=iris_features, filled=True, class_names=model.classes_))
SVG(graph.pipe(format='svg'))
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=2)
forest_model = forest.fit(X=iris_train[iris_features], y=iris_train[iris_targets])
size, index = min((estimator.tree_.node_count, idx) for (idx,estimator) in enumerate(forest.estimators_))
print(f'The smallest tree has {size} nodes!')
smallest_tree = forest_model.estimators_[index]
smallest_tree = smallest_tree.fit(X=iris_train[iris_features], y=iris_train[iris_targets])
smallest_tree.score(X=iris_test[iris_features], y=iris_test[iris_targets])
graph = Source(export_graphviz(smallest_tree, out_file=None, feature_names=iris_features, filled=True, class_names=model.classes_))
SVG(graph.pipe(format='svg'))