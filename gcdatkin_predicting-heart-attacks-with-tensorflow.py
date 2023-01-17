from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
heart_df = pd.read_csv("../input/health-care-data-set-on-heart-attack-possibility/heart.csv")
heart_df.describe()
heart_np = heart_df.to_numpy()

np.random.shuffle(heart_np)
train_test_split = .8

num_examples = heart_np.shape[0]
num_train_examples = int(np.floor(num_examples*train_test_split))
num_test_examples = int(np.ceil(num_examples*(1 - train_test_split)))

print("Training Examples:", num_train_examples)
print("Test Examples:", num_test_examples)
print("\nTotal Examples:", num_examples)
train_data = heart_np[0:num_train_examples, :]
test_data = heart_np[num_train_examples:len(heart_np), :]

X_train = train_data[:, 0:-1]
y_train = train_data[:, -1]

X_test = test_data[:, 0:-1]
y_test = test_data[:, -1]
print(X_train.shape)
print(y_train.shape)
inputs = keras.Input(shape=(13), name="features")
x = layers.Dense(16, activation="relu", name="dense_1")(inputs)
outputs = layers.Dense(2, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
BATCH_SIZE = 64
EPOCHS = 300
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
history = model.fit(
    X_train,
    y_train,
    shuffle=True,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)
predictions = model.predict(X_test)
def get_prediction_array(predictions, y):
    pred_arr = (predictions[:, 0] < 0.5)
    pred_arr = np.column_stack((pred_arr, y))
    return pred_arr
results = get_prediction_array(predictions, y_test)

num_correct = 0

for i in range(num_test_examples):
    if results[i, 0] == results[i, 1]:
        num_correct += 1

print("Accuracy:", num_correct/num_test_examples)
score = model.evaluate(X_test, y_test)
print("\nAccuracy:", score[1])
performance = (results[:, 0] == results[:, 1]).astype(int)
plt.title("Model Performance (Correct vs. Incorrect)")
plt.xlabel("1 = Correct  0 = Incorrect")
plt.ylabel("Count")
plt.xticks([1, 0])
plt.xlim(1.25, -0.25)

plt.hist(performance)
plt.show()