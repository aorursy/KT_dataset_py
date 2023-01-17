import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
plt.style.use('ggplot')
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.info()
df['performance'] = (df['math score'] + df['reading score'] + df['writing score']) // 3
df.head()
df['gender'].value_counts()
codes, uniques = pd.factorize(df['gender'], sort=True)
print(codes[:10], uniques)
df['gender'] = codes
df.head()
codes, uniques = pd.factorize(df['race/ethnicity'], sort=True)
print(codes[:10], uniques)
df['race/ethnicity'] = codes
df.head()
df['parental level of education'].value_counts()
df = df[df['parental level of education'] != "master's degree"]
df['parental level of education'].value_counts()
codes, uniques = pd.factorize(df['parental level of education'])
print(codes[: 10], uniques)
df['parental level of education'] = codes
df.head()
df.head()
df = df.drop(['lunch', 'math score', 'reading score', 'writing score'],axis=1)
df.head()
df['test preparation course'].value_counts()
codes, uniques = pd.factorize(df['test preparation course'])
print(codes[:10], uniques)
df['test preparation course'] = codes
df.head()
df = df.sample(frac=1, random_state=8).reset_index(drop=True)
df.head()
df.shape
labels = df['performance']
data = df.drop('performance', axis=1)
train_df = data[: 705]
train_labels = labels[: 705]

valid_df = data[: 845]
valid_labels = labels[: 845]

test_df = data[845: ]
test_labels = labels[845: ]
train_df.head()
# model = keras.models.Sequential([
#     keras.layers.Dense(64, activation="relu", input_shape=train_df.shape[1:]),
#     keras.layers.Dense(1)
# ])

model = keras.models.Sequential([
    keras.layers.Dense(80, activation="relu", input_shape=train_df.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
my_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

history = model.fit(train_df, train_labels, epochs=500, validation_data=(valid_df, valid_labels), callbacks=[my_cb])
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.xlabel('Epochs')
plt.ylabel('MAE')

plt.legend(['train', 'val'])

plt.tight_layout()
res = model.evaluate(test_df, test_labels)
predictions = model.predict(test_df[: 20])
preds = np.ndarray.flatten(predictions)
preds = np.rint(preds)
x = np.arange(0, 20)

plt.scatter(x, preds)
plt.scatter(x, test_labels[: 20])

plt.legend(['predictions', 'actual'])

plt.ylabel('Performance')

plt.tight_layout()