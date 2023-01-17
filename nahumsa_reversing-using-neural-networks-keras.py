import numpy as np 

import pandas as pd 

import os

import tensorflow as tf

import matplotlib.pyplot as plt
train_df = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')

test_df = pd.read_csv("/kaggle/input/conways-reverse-game-of-life-2020/test.csv")

sample_submission = pd.read_csv("/kaggle/input/conways-reverse-game-of-life-2020/sample_submission.csv")
start_features = [f for f in train_df.columns if "start" in f]

stop_features = [f for f in train_df.columns if "stop" in f]



features_in = stop_features + ["delta"]
from sklearn.model_selection import train_test_split

delta_train, delta_validation, stop_train, stop_validation, Y_train, Y_valid = train_test_split(train_df["delta"].values, 

                                                                                                train_df[stop_features].values.reshape(-1, 25, 25, 1).astype(float),

                                                                                                train_df[start_features].values.reshape(-1, 25, 25, 1).astype(float),

                                                                                                test_size=0.33,

                                                                                                )
X_train = [delta_train, stop_train]

X_valid = [delta_validation, stop_validation]

X_test = [test_df["delta"].values, test_df[stop_features].values.reshape(-1, 25, 25, 1).astype(float)]



X_all_train = [train_df["delta"].values, train_df[stop_features].values.reshape(-1, 25, 25, 1).astype(float)]

Y_all_train = train_df[start_features].values.reshape(-1, 25, 25, 1).astype(float)
def conv_block(inputs, filters, index, activation='relu'):

    """ Creates a convolutional block with batch normalization.

    

    Args:

        inputs: input layer.

        filters: number of filters in the convolutional layer.

        index: Index for the name of the convolutional layer.

        activation: Activation of the convolutional block. 

        (default='relu')

    

    Out:

        keras.layer: Layer of the convolutional block.

        

    """

    x = layers.Conv2D(filters, kernel_size=(3,3), padding="SAME", name=f'conv{index}')(inputs)

    x = layers.BatchNormalization()(x)

    return layers.Activation(activation, name=activation + str(index))(x)
from tensorflow.keras import layers

from tensorflow.keras.models import Model

def create_model(dropout_prob=0.3):

    """ Creates the CNN model with the convolutional block.

    """

    input_delta = layers.Input(shape=(1,), name="input_delta")

    dense_delta = layers.Dense(25*25, name='dense_delta')(input_delta)

    dense_reshape = layers.Reshape((25,25,1), name='reshape_delta')(dense_delta)



    input_image = layers.Input(shape=(25,25,1), name="input_images")

    all_inputs = layers.Concatenate(axis=3, name='concatenate')([input_image, dense_reshape])



    x = conv_block(all_inputs, 32, index=1)    

    x = layers.Dropout(dropout_prob)(x)

    

    x = conv_block(x, 128, index=2)    

    x = layers.Dropout(dropout_prob)(x)

    

    x = conv_block(x, 256, index=3)

    x = layers.Dropout(dropout_prob)(x)

    

    x = conv_block(x, 64, index=4)

    x = layers.Dropout(dropout_prob)(x)

    

    out = conv_block(x, 1, index=5, activation='sigmoid')



    return Model(inputs=[input_delta, input_image], outputs=out)
model = create_model()

model.compile(loss="bce", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
history = model.fit(x=X_train,

                    y=Y_train, 

                    batch_size=128,

                    epochs=25,

                    validation_data=(X_valid, Y_valid))
loss = history.history['loss']

val_loss = history.history['val_loss']

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

epochs = range(len(loss))



fig = plt.figure(figsize=(12,6))

gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[0, 0])

ax2 = fig.add_subplot(gs[0, 1])



ax1.plot(epochs, loss, 'r', label='Training')

ax1.plot(epochs, val_loss, 'b', label='Validation')

ax1.set_xlabel('Epochs', size=16)

ax1.set_ylabel('Loss', size=16)

ax1.legend()



ax2.plot(epochs, accuracy, 'r', label='Training')

ax2.plot(epochs, val_accuracy, 'b', label='Validation')

ax2.set_xlabel('Epochs', size=16)

ax2.set_ylabel('Accuracy', size=16)

ax2.legend()

plt.show()
idx = 2



delta_sample = X_valid[0][idx]

img_sample = X_valid[1][idx]

out_sample = Y_valid[idx]

predicted = model.predict(X_valid)



fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

fig.suptitle("Delta: " + str(delta_sample))



ax1.imshow((img_sample.reshape(25, 25)), cmap="gray")

ax1.set_title("Stop Setting")



ax2.imshow((out_sample.reshape(25, 25)), cmap="gray")

ax2.set_title("Start Setting")



ax3.imshow((predicted[idx]>=0.5).reshape(25, 25), cmap="gray")

ax3.set_title("Predicted Setting")

plt.show()
model = create_model()

model.compile(loss="bce", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
history = model.fit(x=X_all_train,

                    y=Y_all_train, 

                    batch_size=128,

                    epochs=20,

                   )
test_prediction = model.predict(X_test)

test_prediction = (test_prediction > 0.5).astype(int).reshape(test_df.shape[0], -1)

sub = test_df[["id"]].copy()

tmp = pd.DataFrame(test_prediction, columns=start_features)

submission = sub.join(tmp)
submission.to_csv("submission.csv", index=False)