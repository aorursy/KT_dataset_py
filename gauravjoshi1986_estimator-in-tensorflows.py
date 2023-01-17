import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import train_test_split

print(tf.__version__)
# Load datasets.
iris = pd.read_csv("../input/Iris.csv", skipinitialspace=True)
iris.head()
iris = iris.drop(["Id"], axis = 1)
# Visualize data with Seaborn
g=sns.pairplot(iris, hue="Species", size= 2.5)
# Convert categorical label to integers
iris["Species"] = iris["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})
FEATURES = iris.columns[0:4]
TARGET = "Species"
# labels and features
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
feature_cols
classifier = tf.estimator.DNNClassifier(feature_columns = feature_cols,
                                      hidden_units = [10, 10],
                                      n_classes = 3)
def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
              x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
              y = pd.Series(data_set[TARGET].values),
              num_epochs=num_epochs,
              shuffle=shuffle)
classifier.train(input_fn=get_input_fn(iris), steps=100)
# Evaluate accuracy.
ev = classifier.evaluate(input_fn = get_input_fn(iris, num_epochs=1, shuffle=False), steps=1)
print("accuracy : {0:f}, Loss : {0:f}".format(ev["accuracy"], ev['loss']))