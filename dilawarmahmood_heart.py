import tensorflow as tf

from tensorflow import keras



import pandas as pd
csv_file = "../input/heart-disease-dataset/heart.csv"
df = pd.read_csv(csv_file)
df.head()
df.dtypes
target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
train_dataset = dataset.shuffle(len(df)).batch(1)
model = keras.Sequential([

    keras.layers.Dense(10, activation='relu'),

    keras.layers.Dense(10, activation='relu'),

    keras.layers.Dense(1)

])



model.compile(optimizer='adam',

             loss=keras.losses.BinaryCrossentropy(from_logits=True),

             metrics=['accuracy'])
model.fit(train_dataset, epochs=15)