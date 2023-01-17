import numpy as np
from sklearn.datasets import make_classification

np.random.seed(42)
X, y = make_classification(n_samples=100000, n_features=2, n_informative=2, n_redundant=0)
n_train_samples = 1000

X_train, y_train = X[:n_train_samples], y[:n_train_samples]
X_test, y_test = X[n_train_samples:], y[n_train_samples:]
import matplotlib.pyplot as plt
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.axis('off')
import tensorflow as tf


def input_fn(X, y): # returns x, y (where y represents label's class index).
    dataset = tf.data.Dataset.from_tensor_slices(({'X': X[:, 0], 'Y': X[:, 1]}, y))
    dataset = dataset.shuffle(1000).batch(1000)
    return dataset


from tensorflow.estimator import BaselineClassifier
clf = BaselineClassifier(n_classes=2)


clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))

# Convert object-wrapped prediction iterator to a list.
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from tensorflow.estimator import LinearClassifier
feature_columns = [
    tf.feature_column.numeric_column(key='X', dtype=tf.float32),
    tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
]
clf = LinearClassifier(n_classes=2, feature_columns=feature_columns)


clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from tensorflow.estimator import DNNClassifier
feature_columns = [
    tf.feature_column.numeric_column(key='X', dtype=tf.float32),
    tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
]
clf = DNNClassifier(n_classes=2, feature_columns=feature_columns, hidden_units=[32, 32])


clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10000)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# from tensorflow.estimator import BoostedTreesClassifier
# feature_columns = [
#     tf.feature_column.numeric_column(key='X', dtype=tf.float32),
#     tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
# ]
# clf = BoostedTreesClassifier(n_classes=2, feature_columns=feature_columns, n_trees=100, n_batches_per_layer=1)


# clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10000)
# y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
# y_pred = np.array([p['class_ids'][0] for p in y_pred])

# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_pred))
from tensorflow.estimator import DNNLinearCombinedClassifier
feature_columns = [
    tf.feature_column.numeric_column(key='X', dtype=tf.float32),
    tf.feature_column.numeric_column(key='Y', dtype=tf.float32)
]
clf = DNNLinearCombinedClassifier(n_classes=2, dnn_feature_columns=feature_columns, dnn_hidden_units=[32, 32], linear_feature_columns=feature_columns)

clf.train(input_fn=lambda: input_fn(X_train, y_train), max_steps=10000)
y_pred = clf.predict(input_fn=lambda: input_fn(X_test, y_test))
y_pred = np.array([p['class_ids'][0] for p in y_pred])

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))