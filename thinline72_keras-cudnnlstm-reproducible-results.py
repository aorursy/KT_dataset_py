# Based on code from @rampeerr blog post: https://towardsdatascience.com/reproducible-model-training-deep-dive-2a4988d69031



# Just run this notebook and compare results from the cell output

# Sum of previous initial and trained model weights are already added as a dataset

# For more info, please read @rampeerr blog post, which would be updated soon with CuDNN results

# - https://towardsdatascience.com/reproducible-model-training-deep-dive-2a4988d69031



# Key takeaways:

# - It's possible to get reproducible results in Keras with CuDNNLSTM + Dense on GPU via torch import

# - Although, from session to session you can still get a bit different results ¯\_(ツ)_/¯

# - And it's not possible to get reproducible results with Conv + Dense

# - So it should be easier to just use PyTorch if you want to get the same results
import os

import struct

import random

import numpy as np

import torch

from keras.layers import *

from keras.losses import *

from keras.optimizers import *

from keras import initializers

from keras import Input, Model

import tensorflow as tf

import keras.backend as K



SEED = 239

print("SEED: {}".format(SEED))

np.random.seed(SEED)

random.seed(SEED)

tf.set_random_seed(SEED)



os.environ['PYTHONHASHSEED'] = str(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False



session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

K.set_session(sess)



def assert_same_across_runs(metric, value):

    filename = metric + ".bin"

    if os.path.exists(filename):

        with open(filename, "rb") as f:

            old_value, = struct.unpack("f", f.read())

        if abs(old_value - value) > 1e-8:

            print(f"{metric} is inconsistent! {old_value} != {value}")

        else:

            print(f"{metric} is consistent across runs: {value}")

    else:

        print(f"Cannot ensure consistency of {metric} between runs because it is the first run. Please run this script again.")



    with open(filename, "wb+") as f:

        f.write(struct.pack("f", value))
# dummy random data

Xs = np.random.normal(size=(500, 40, 300))

Ws = np.random.normal(size=(40*300, 1))

Ys = np.dot(Xs.reshape((500, 40*300)), Ws) + np.random.normal(size=(500, 1))
def create_lstm_net(dim):

    x_input = Input(shape=dim, name="inp")

    lstm = Bidirectional(CuDNNLSTM(128, return_sequences=True,

                                   kernel_initializer=initializers.glorot_uniform(seed=SEED),

                                   recurrent_initializer=initializers.Orthogonal(seed=SEED)), name="bi_lstm")(x_input)

    pool = GlobalMaxPool1D(name="global_max_pool")(lstm)



    x_out = Dense(1, activation="sigmoid", name="out",

                  kernel_initializer=initializers.glorot_uniform(seed=SEED))(pool)

    return Model(inputs=x_input, outputs=x_out, name="test")



model = create_lstm_net((40, 300))

init_weights = np.array(model.get_weights()[0]).sum()

model.compile(optimizer=RMSprop(lr=1e-2), loss=MSE)

model.fit(Xs, Ys, batch_size=10, epochs=10)

model_weights = model.get_weights()[0].sum()



print(f"Init  weights sum: {init_weights}")

print(f"Model weights sum: {model_weights}")

assert_same_across_runs("mixed model weight before training lstm", init_weights)

assert_same_across_runs("mixed model weight after training lstm", model_weights)



model = create_lstm_net((40, 300))

init_weights = np.array(model.get_weights()[0]).sum()

model.compile(optimizer=RMSprop(lr=1e-2), loss=MSE)

model.fit(Xs, Ys, batch_size=10, epochs=10)

model_weights = model.get_weights()[0].sum()



print(f"Init  weights sum: {init_weights}")

print(f"Model weights sum: {model_weights}")

assert_same_across_runs("mixed model weight before training lstm", init_weights)

assert_same_across_runs("mixed model weight after training lstm", model_weights)
def create_cnn_net(dim):

    x_input = Input(shape=dim, name="inp")

    conv = Conv1D(128, kernel_size=3, kernel_initializer=initializers.glorot_uniform(seed=SEED), name="conv")(x_input)

    pool = GlobalMaxPool1D(name="global_max_pool")(conv)



    x_out = Dense(1, activation="sigmoid", name="out",

                  kernel_initializer=initializers.glorot_uniform(seed=SEED))(pool)

    return Model(inputs=x_input, outputs=x_out, name="test")



model = create_cnn_net((40, 300))

init_weights = np.array(model.get_weights()[0]).sum()

model.compile(optimizer=RMSprop(lr=1e-2), loss=MSE)

model.fit(Xs, Ys, batch_size=10, epochs=10)

model_weights = model.get_weights()[0].sum()



print(f"Init  weights sum: {init_weights}")

print(f"Model weights sum: {model_weights}")

assert_same_across_runs("mixed model weight before training cnn", init_weights)

assert_same_across_runs("mixed model weight after training cnn", model_weights)



model = create_cnn_net((40, 300))

init_weights = np.array(model.get_weights()[0]).sum()

model.compile(optimizer=RMSprop(lr=1e-2), loss=MSE)

model.fit(Xs, Ys, batch_size=10, epochs=10)

model_weights = model.get_weights()[0].sum()



print(f"Init  weights sum: {init_weights}")

print(f"Model weights sum: {model_weights}")

assert_same_across_runs("mixed model weight before training cnn", init_weights)

assert_same_across_runs("mixed model weight after training cnn", model_weights)