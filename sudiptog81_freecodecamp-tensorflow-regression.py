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
import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf
dftrain = pd.read_csv("/kaggle/input/train.csv")

dfeval = pd.read_csv("/kaggle/input/eval.csv")
dftrain.head()
dftrain.shape
y_train = dftrain.pop("survived")

y_eval = dfeval.pop("survived")



y_train.head()
dftrain.age.hist(bins = 20)
dftrain.sex.value_counts().plot(kind = "barh")
dftrain["class"].value_counts().plot(kind = "barh")
survival_distribution = pd.concat([dftrain, y_train], axis=1).groupby("sex").survived.mean() * 100

survival_distribution.plot(kind = "barh").set_xlabel("% survived")
dftrain.dtypes
CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]

NUMERIC_COLUMNS = ["age", "fare"]



feature_columns = []



for feature in CATEGORICAL_COLUMNS:

    vocabulary = dftrain[feature].unique()

    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))

    

for feature in NUMERIC_COLUMNS:

    feature_columns.append(tf.feature_column.numeric_column(feature, dtype = tf.float64))

    

feature_columns
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

    def input_fn():

        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:

            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    return input_fn
train_input_fn = make_input_fn(dftrain, y_train)

eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

result
predictions = list(linear_est.predict(eval_input_fn))

survival_probabilities = pd.Series([pred["probabilities"][1] for pred in predictions])
for i in range(len(dfeval.head())):

    print(dfeval.loc[i])

    print("survived: {}".format("yes" if (y_eval.loc[i] == 1) else "no"))

    print(f"predicted survival probability: {survival_probabilities[i]}")
survival_probabilities.plot(kind="hist", bins=20, title="Survival Probabilities")