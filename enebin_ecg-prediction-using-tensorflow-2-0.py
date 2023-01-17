import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, add, ReLU
from tensorflow.keras import Model
import matplotlib
import matplotlib.pyplot as plt


train_data = pd.read_csv("../input/heartbeat/mitbih_train.csv", header=None)
test_data = pd.read_csv("../input/heartbeat/mitbih_test.csv", header=None)


train_information = train_data.iloc[:, :-1].to_numpy()
train_label = train_data.iloc[:, -1].to_numpy()

test_information = test_data.iloc[:, :-1].to_numpy()
test_label = test_data.iloc[:, -1].to_numpy()

print(test_label)
gaussian_flag = 0

def add_gaussian_noise(signal):
    noise=np.random.normal(0,0.05,187)
    return (signal+noise)

tempo=train_information[20]
print(train_label[0])
bruiter=add_gaussian_noise(tempo)

plt.subplot(2,1,1)
plt.plot(tempo)

plt.subplot(2,1,2)
plt.plot(bruiter)

plt.show()

train_information = add_gaussian_noise(train_information)
gaussian_flag = not gaussian_flag
train_information = train_information[..., tf.newaxis]
test_information = test_information[..., tf.newaxis]

print(train_information.shape)
BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_information, train_label)).shuffle(10000).batch(BATCH_SIZE )
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_information, test_label)).batch(BATCH_SIZE )
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_in = Conv1D(32, input_shape=(None, 187, 1), kernel_size=(5), strides=1,
                              padding='same', activation='relu')
        self.conv_relu = Conv1D(32, kernel_size=(5), strides=1,
                           padding='same', activation='relu')
        self.conv_raw = Conv1D(32, kernel_size=(5), strides=1,
                              padding='same')
        self.maxpool = MaxPooling1D(pool_size=5, strides=2)
        self.flatten = Flatten()
        self.relu = ReLU()
        self.dense = Dense(32, activation='relu')
        self.dense_out = Dense(5, activation='softmax')

    def routine(self, x):
        input_param = x
        x = self.conv_relu(x)
        x = self.conv_raw(x)
        x = add([input_param, x])
        x = self.relu(x)
        return self.maxpool(x)

    def call(self, x):
        x = self.conv_in(x)
        for _ in range(5):
            x = self.routine(x)

        x = self.flatten(x)

        x = self.dense(x)
        return self.dense_out(x)
model = MyModel()

# Loss Func. SCC
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# Opimizer Adam
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
EPOCHS = 75

train_loss_results = []
train_accuracy_results = []

test_loss_results = []
test_accuracy_results = []


for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    train_loss_results.append(train_loss.result())
    train_accuracy_results.append(train_accuracy.result())

    test_loss_results.append(test_loss.result())
    test_accuracy_results.append(test_accuracy.result())

    template = 'EPOCH: {}, LOSS: {}, ACCURACY: {}, TEST LOSS: {}, TEST ACCURACY: {}'
    if (epoch + 1) % 5 == 0:
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))
COLOR = 'white'
matplotlib.rcParams['text.color'] = COLOR
matplotlib.rcParams['axes.labelcolor'] = COLOR
matplotlib.rcParams['xtick.color'] = COLOR
matplotlib.rcParams['ytick.color'] = COLOR

fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Results', fontsize=20)

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)
axes[0].grid()

axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].plot(train_accuracy_results)
axes[1].grid()


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Test Results', fontsize=20)

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(test_loss_results)
axes[0].grid()

axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].plot(test_accuracy_results)
axes[1].grid()

plt.show()
