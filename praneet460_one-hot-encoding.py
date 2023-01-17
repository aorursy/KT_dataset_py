# Import required libraries

import tensorflow as tf

import numpy as np

import pandas as pd
labels = [0, 1, 2]

result = tf.one_hot(indices= labels, depth= 3) # depth = N

print(result)
with tf.Session():

    print(result.eval())
new_result = tf.one_hot(indices= labels, depth= 3, on_value= 5.0, off_value= 0.0, axis= -1)

with tf.Session():

    print(new_result.eval())
# load the dataset

from sklearn.datasets import load_iris

data = load_iris()

iris_labels = data.target

print(iris_labels)
# convert array to tensor

tensor = tf.convert_to_tensor(iris_labels, dtype=tf.int32)

print(tensor)
final_result = tf.one_hot(indices= tensor, depth = 3)

with tf.Session():

    df = pd.DataFrame(data = final_result.eval(), columns = ["setosa", "versicolor", "virginica"])

    print(final_result.eval())
df.head()
# load iris data into the dataframe

iris_df = pd.DataFrame(data = data.data, columns= data.feature_names)

iris_df.head()
# combined both the dataframes

final_df = pd.concat([iris_df, df], axis= 1, join_axes=[iris_df.index])

final_df