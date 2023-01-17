import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np 

import pandas as pd 

import tensorflow as tf

import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

from tensorflow import feature_column
FlowerData = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")

FlowerData.head()
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)

sns.pairplot(FlowerData, hue="species", size=3)
FlowerData["species"] = pd.Categorical(FlowerData["species"])

print(pd.Categorical(FlowerData["species"]))

FlowerData["species"] = FlowerData.species.cat.codes

FlowerData.head()
train_df, test_df = train_test_split(FlowerData, test_size=30, shuffle=True)
def df_to_dataset(dataframe, shuffle=True, batch_size=20):

    dataframe = dataframe.copy()

    y = dataframe.pop("species")

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), y))

    if shuffle:

        ds = ds.shuffle(buffer_size=len(dataframe)*2)

    ds = ds.batch(batch_size)

    return ds
train_dataset = df_to_dataset(train_df)

test_dataset = df_to_dataset(test_df)
feature_columns = []



for header in FlowerData.columns:

    if header != "species":

      feature_columns.append(feature_column.numeric_column(header))



feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([

    feature_layer,

    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(3)

])

model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

model.fit(train_dataset, epochs=100)
test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
test_y = test_df.pop("species")
probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(dict(test_df.iloc[0:10]))
print(test_df.iloc[0:10])
print(predictions)

print(test_y)