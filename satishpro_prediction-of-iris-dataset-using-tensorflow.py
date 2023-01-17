import tensorflow as tf
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics
from sklearn.metrics import classification_report,confusion_matrix

iris = datasets.load_iris()

feature_columns = skflow.infer_real_valued_columns_from_input(iris.data)

classifier = skflow.DNNClassifier(feature_columns=feature_columns,hidden_units=[10, 20, 10], n_classes=3)

classifier.fit(x=iris.data, y=iris.target, steps=300,batch_size=32)

predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, predictions)

print ("Accuracy: %f" % score)

print(classification_report(iris.target,predictions))