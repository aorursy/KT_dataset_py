from __future__ import (

    absolute_import,

    division,

    print_function,

    unicode_literals

)



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import clear_output

from six.moves import urllib
import tensorflow as tf
from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/iris/Iris.csv").drop("Id", axis=1)

df
df.hist()
df.Species.value_counts().plot(kind="barh")
y = df.pop("Species")
COLUMNS = list(df.keys())

SPECIES = list(y.unique())
SPECIES
train, test, y_train, y_test = train_test_split(df, y, train_size=0.7, test_size=0.3)
train.shape
test.shape
sp_encoding = {}



for (i, v) in enumerate(SPECIES):

    sp_encoding[v] = i

    

y_train.replace(sp_encoding, inplace=True)

y_test.replace(sp_encoding, inplace=True)
def input_fn(features, labels, training=True, batch_size=256):

    ds = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:

        ds = ds.shuffle(1000).repeat()

    return ds.batch(batch_size)
feature_columns = []



for column in COLUMNS:

    feature_columns.append(tf.feature_column.numeric_column(column))



feature_columns
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[30, 10], n_classes=len(SPECIES))
classifier.train(input_fn=lambda: input_fn(train, y_train, training=True), steps=5000)
result = classifier.evaluate(input_fn=lambda: input_fn(test, y_test, training=False))

result
rev_sp_encoding= {}



for (i, v) in enumerate(SPECIES):

    rev_sp_encoding[i] = v

    

y_test.replace(rev_sp_encoding, inplace=True)

    

pd.concat([test.head(), y_test.head()], axis=1)
def predict_fn(features, batch_size=256):

    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)



predictions = classifier.predict(input_fn=lambda: predict_fn(test.head()))



for pred_dict in predictions:

    class_id = pred_dict['class_ids'][0]

    probability = pred_dict['probabilities'][class_id]

    print("Predicted: {} (Probability: {:.3f}%)".format(SPECIES[class_id], 100 * probability))