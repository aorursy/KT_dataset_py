import graphviz

import numpy as np

import pandas as pd



from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn.tree import export_graphviz

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
np.random.seed(42)
animal_class = pd.read_csv('../input/class.csv')

animal_class.head()
animal_zoo = pd.read_csv('../input/zoo.csv')

animal_zoo.head()
features = animal_zoo.columns[1:-1]



X = animal_zoo[features]

y = animal_zoo.class_type



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
dummy_classifier = DummyClassifier(strategy='prior')

dummy_classifier.fit(X_train, y_train)

dummy_classifier.score(X_test, y_test)
model = DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)
prediction = model.predict(X_test)

accuracy_score(prediction, y_test)
dot_model_graph = export_graphviz(model, feature_names=features, class_names=animal_class['Class_Type'])

graphviz.Source(dot_model_graph)