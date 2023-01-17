!pip install tensorflow-gpu==2.0.0-alpha
# 

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns

import pandas as pd 

import numpy as np

import os



print(os.listdir("../input"))
data = pd.read_csv("../input/pmsm_temperature_data.csv")

data = data.drop("profile_id", axis=1)
data.head(5)
data.hist(figsize = (20,20))

plt.show()
column = ['coolant', 'motor_speed', 'torque']
fig, axes = plt.subplots(nrows=3, figsize=(5,20))

fig.subplots_adjust(hspace=0.2)



for (ax, i) in zip(axes, column):

    sns.violinplot(x=data[i], ax=ax)

    plt.plot()
fig, axes = plt.subplots(nrows=3, figsize=(5,20))

fig.subplots_adjust(hspace=0.2)



for (ax, i) in zip(axes, column):

    sns.distplot(data[i], ax=ax)

    plt.plot()
x = data.drop("motor_speed", axis=1).values

y = data["motor_speed"].values.reshape(-1, 1)



x = tf.cast(x, dtype=tf.float32)

y = tf.cast(y, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
loss_fn = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(0.001)
model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(128, activation='sigmoid', input_shape=(11,)),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(1, activation='tanh')

])
loss_metric = tf.keras.metrics.Mean(name='loss')
def train_step(data, labels):

    with tf.GradientTape() as tape:

        

        output = model(data, training=True)



        loss = loss_fn(labels, output)



    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    

    loss_metric.update_state(loss)
loss_history = []



for epoch in range(1, 11):

    

    loss_metric.reset_states()



    for (data, labels) in dataset:

        train_step(data, labels)

        

    mean_loss = loss_metric.result()

    

    loss_history.append(mean_loss)

    

    print('Epoch: {}  MSE: {:.3f}'.format(epoch, mean_loss))

h_loss = np.array(loss_history)
plt.plot(h_loss)

plt.title("loss")

plt.show()