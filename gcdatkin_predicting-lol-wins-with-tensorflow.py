import numpy as np

import pandas as pd

import seaborn as sns



import tensorflow as tf

from tensorflow.keras import layers
# Load in dataset

data = pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data.head()
data.shape
data.drop('gameId', axis=1, inplace=True)
data = data.sample(frac=1).reset_index(drop=True)
y = data['blueWins']
X = data.drop('blueWins', axis=1, inplace=False)
sns.heatmap(X.corr())
train_test_split = 0.7



num_examples = X.shape[0]

num_train_examples = int(np.floor(num_examples*train_test_split))

num_test_examples = int(np.ceil(num_examples - num_train_examples))



print(num_examples)

print(num_train_examples)

print(num_test_examples)
X_train = X.iloc[0:num_train_examples, :]

y_train = y.iloc[0:num_train_examples]



X_test = X.iloc[num_train_examples:num_examples, :]

y_test = y.iloc[num_train_examples:num_examples]
optimizer = tf.keras.optimizers.Adam(

    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,

    name='Adam'

)
inputs = tf.keras.Input(shape=(38,))

x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(inputs)

x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)

outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()
model.compile(

    optimizer=optimizer,

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=['accuracy']

)
BATCH_SIZE = 32

EPOCHS = 300
model.fit(

    x=X_train,

    y=y_train,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    verbose=1,

    validation_split=0.2,

    shuffle=True

)
loss, accuracy = model.evaluate(X_test, y_test)