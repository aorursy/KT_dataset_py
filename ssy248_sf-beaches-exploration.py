# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/beach.csv")

df.head
df.describe
df.isnull().sum()

data.info()
beachname = tf.feature_column.categorical_column_with_vocabulary_list("beachname", ["BAY#202.4_SL","OCEAN#21.1_SL"])
selected_columns = ["DATA"]



for column in selected_columns:

        df[column] = df[column].str.replace(">","")

        df[column] = df[column].str.replace("<","")

import matplotlib.pyplot as plt

import matplotlib.colors as colors

# create plot 



s = df.sort_values("DATA", ascending=False)

values = s["DATA"].tolist()

ind = np.arange(len(values))





# Creating new plot

fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111)

ax.yaxis.grid()

ax.xaxis.grid()

bars = ax.bar(ind, values)
d = df["DATA"]

X_train = d.sample(frac=0.8)
# X_test contains all the transaction not in X_train.

X_test = df.loc[~df.index.isin(X_train.index)]
from sklearn.utils import shuffle

#Shuffle the dataframes so that the training is done in a random order.

X_train = shuffle(X_train)

X_test = shuffle(X_test)
y_train = X_train

y_test = X_test
# binary classification problem

# 1 if > 10 

# 0 if not 

train_labels = (X_train.apply(lambda x: ">10" in x)).astype(int)

test_labels = (X_test.apply(lambda x: ">10" in x)).astype(int)
import tensorflow as tf

# continuous columns

numberdata = tf.feature_column.numeric_column("data")
number_buckets = tf.feature_column.bucketized_column(numberdata, boundaries = [10,20,25,30,35,40,50,60,70,80,90])

number_x_beachname = tf.feature_column.crossed_column([number_buckets,beachname],hash_bucket_size=1000)
base_columns = [number_buckets, beachname]

crossed_columns = [number_x_beachname]
import tempfile

import urllib

model_dir = tempfile.mkdtemp()

m = tf.estimator.LinearClassifier(

    model_dir=model_dir, feature_columns=base_columns + crossed_columns)
CSV_COLUMNS = [

    "ANALYTE", "COLOR", "CSO", "DATA", "POSTED", "SAMPLE_DATE", "SOURCE"]

num_epochs=None

def input_fn(df, shuffle):

  """Input builder function."""

  df_data = df

  # remove NaN elements

  df_data = df_data.dropna(how="any", axis=0)

  labels = df_data["DATA"].apply(lambda x: ">10" in x).astype(int)

  return tf.estimator.inputs.pandas_input_fn(

      x=df_data,

      y=labels,

      batch_size=100,

      num_epochs=num_epochs,

      shuffle=shuffle,

      num_threads=5)
m = tf.estimator.LinearClassifier(

    model_dir=model_dir, feature_columns=base_columns,

    optimizer=tf.train.FtrlOptimizer(

      learning_rate=0.1,

      l1_regularization_strength=1.0,

      l2_regularization_strength=1.0))
tf.initialize_all_variables()

# training and evaluating the model after adding all the features

tfname="../input/beach.csv"

train_steps =5

m.train( input_fn=input_fn(df, shuffle=True),steps=None)

input_fn = tf.contrib.learn.io.numpy_input_fn({"x":X_train}, y_train, batch_size=4, num_epochs=10)

m.train(input_fn=input_fn, steps=10)