import graphviz

import numpy as np

import pandas as pd

from sklearn import tree

from sklearn.externals import joblib

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import shuffle
flowers = pd.read_csv('../input/Iris.csv')

del flowers['Id']

flowers = shuffle(flowers, random_state=1).reset_index(drop=True)
flowers.head(5)
flowers.groupby('Species').size()
flowers.head(5)
features = flowers.iloc[:, :4].values
features[:5]
labels = flowers.iloc[:, 4:].values
labels[:5]
classifier = DecisionTreeClassifier()
classifier.fit(features, labels)
flowers.head(1)
classifier.predict(np.array([[5.9, 4.1, 1.4, 0.3]]))[0]
joblib.dump(classifier, 'flower-classifier.pkl')
dot_data = tree.export_graphviz(

    classifier,

    out_file=None,

    feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],

    class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],

    filled=True,

    special_characters=True)

graph = graphviz.Source(dot_data)

graph