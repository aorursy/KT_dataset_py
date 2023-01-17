import json
import math
import os
from pprint import pprint

import numpy as np
import tensorflow as tf
print(tf.version.VERSION)
N_POINTS = 100
X = tf.constant(range(N_POINTS), dtype=tf.float32)
Y = 2 * X + 100
def create_dataset(X,Y,epochs,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    dataset = dataset.repeat(epochs).batch(batch_size,drop_remainder=True)
    return dataset
BATCH=32
EPOCH = 10

dataset = create_dataset(X,Y,EPOCH,BATCH)
for i, (x, y) in enumerate(dataset):
    print("x:", x.numpy(), "y:", y.numpy())
    assert len(x) == BATCH
    assert len(y) == BATCH
assert  EPOCH

def loss_mse(X, Y, w0, w1):
    Y_hat = w0 * X + w1
    errors = (Y_hat - Y)**2
    return tf.reduce_mean(errors)


def compute_gradients(X, Y, w0, w1):
    with tf.GradientTape() as tape:
        loss = loss_mse(X, Y, w0, w1)
    return tape.gradient(loss, [w0, w1])
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = .02
scale =100

MSG = "STEP {step} - loss: {loss}, w0: {w0}, w1: {w1}\n"

w0 = tf.Variable(0.0)
w1 = tf.Variable(0.0)

dataset = create_dataset(X/scale, Y/scale, epochs=EPOCHS, batch_size=BATCH_SIZE)

for step, (X_batch, Y_batch) in enumerate(dataset):

    dw0, dw1 = compute_gradients(X_batch, Y_batch, w0, w1)
    w0.assign_sub(dw0 * LEARNING_RATE)
    w1.assign_sub(dw1 * LEARNING_RATE)

    if step % 100 == 0:
        loss = loss_mse(X_batch, Y_batch, w0, w1)
        print(MSG.format(step=step, loss=loss, w0=w0.numpy(), w1=w1.numpy()))
        
assert loss < 1
assert abs(w0 - 2) < 1
assert abs(w1 - 10) < 10
!ls -l ../data/taxi*.csv
CSV_COLUMNS = [
    'fare_amount',
    'pickup_datetime',
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'passenger_count',
    'key'
]
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], ['na'], [0.0], [0.0], [0.0], [0.0], [0.0], ['na']]
def create_dataset(pattern):
    return tf.data.experimental.make_csv_dataset(
        pattern, 1, CSV_COLUMNS, DEFAULTS)


tempds = create_dataset('../data/taxi-train*')
print(tempds)

for data in tempds.take(2):
    pprint({k: v.numpy() for k, v in data.items()})
    print("\n")
UNWANTED_COLS = ['pickup_datetime', 'key']

# TODO 4a
def features_and_labels(row_data):
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    
    for unwanted_col in UNWANTED_COLS:
        features.pop(unwanted_col)

    return features, label
for row_data in tempds.take(2):
    features, label = features_and_labels(row_data)
    pprint(features)
    print(label, "\n")
    assert UNWANTED_COLS[0] not in features.keys()
    assert UNWANTED_COLS[1] not in features.keys()
    assert label.shape == [1]