# Imports and helper functions
import tensorflow as tf

def int_list_to_hex(l):
    return ''.join("{0:0{1}x}".format(x, 2) for x in l)

def int_list_to_string(l):
    return ''.join(chr(x) for x in l)
message_str = "Hello this is a secret message."
message = tf.constant([ord(c) for c in message_str], tf.uint8)

key_uint32 = tf.Variable(tf.random_uniform(message.shape, minval=0, maxval=2**8, dtype=tf.int32))
key = tf.cast(key_uint32, tf.uint8)

encrypt_xor = tf.bitwise.bitwise_xor(message, key)
decrypt_xor = tf.bitwise.bitwise_xor(encrypt_xor, key)

with tf.Session().as_default() as session:
    session.run(tf.global_variables_initializer())
    print('key:'.ljust(24), int_list_to_hex(key.eval()))
    print('message:'.ljust(24), int_list_to_string(message.eval()))

    ciphertext = encrypt_xor.eval()
    print('encrypted ciphertext:'.ljust(24), int_list_to_hex(ciphertext))

    plaintext = decrypt_xor.eval()
    print('decrypted plaintext:'.ljust(24), int_list_to_string(plaintext))
BLOCK_SIZE = 32
NUM_ROUNDS = 16

def feistel_network_encrypt_round(round_key, left_0, right_0):
    """Run one encryption round of a Feistel network.

    Args:
        round_key: The PRF is keyed with this round key.
        left_0: the left half of the input.
        right_0: the right half of the input.
    Returns:
        right n+1: the right half ouput.
        left n+1: the left half output.
    """
    # (Using bitwise inversion instead of a true PRF)
    f_ri_ki = tf.bitwise.invert(right_0)
    right_plusone = tf.bitwise.bitwise_xor(left_0, f_ri_ki)

    return right_0, right_plusone


def feistel_network_decrypt_round(round_key, left_plusone, right_plusone):
    """Run one decryption round of a Feistel network.

    Args:
        round_key: The PRF is keyed with this round key.
        left_plusone: the preceding left half of the input.
        right_plusone: the precedingright half of the input.
    Returns:
        left n-1: the decrypted left half.
        right n-1: the decrypted right half.
    """
    # (Using bitwise inversion instead of a true PRF)
    f_lip1_ki = tf.bitwise.invert(left_plusone)
    right_0 = tf.bitwise.bitwise_xor(right_plusone, f_lip1_ki)

    return right_0, right_plusone

def pkcs7_pad(text):
    # Not true PKCS #7 padding, only for demo purposes.
    val = BLOCK_SIZE - (len(text) % BLOCK_SIZE)
    return text + ('%d' % val) * val

def pkcs7_unpad(text):
    val = text[-1]
    return text[:(len(text) - int(text[-1]))]

message_str = pkcs7_pad("Hello this is a secret message.")
input_tensor = tf.constant([ord(c) for c in message_str], tf.uint8)

key_uint32 = tf.Variable(tf.random_uniform((NUM_ROUNDS,), minval=0, maxval=2**8, dtype=tf.int32))
key = tf.cast(key_uint32, tf.uint8)

with tf.Session().as_default() as session:
    session.run(tf.global_variables_initializer())

    # Keys here are used to seed the random shuffle.
    # Key is 16 bytes, one byte per round.
    # (Note: this does not follow the DES key scheduling algorithm).
    print('key:'.ljust(24), int_list_to_hex(key.eval()))
    print('padded message:'.ljust(24), int_list_to_string(input_tensor.eval()))
    
    # Encryption: split the input in half and run the network for 16 rounds.
    left, right = tf.split(input_tensor, num_or_size_splits=2)
    
    for round_num in range(NUM_ROUNDS):
        right, left = feistel_network_encrypt_round(key[round_num], left, right)
    
    print('encrypted ciphertext:'.ljust(24), int_list_to_hex(left.eval()) + int_list_to_hex(right.eval()))

    # Decryption: run the network in reverse.
    for round_num in range(NUM_ROUNDS):
        left, right = feistel_network_decrypt_round(key[round_num], left, right)
    
    print('decrypted plaintext:'.ljust(24), pkcs7_unpad(int_list_to_string(left.eval()) + int_list_to_string(right.eval())))
# Import libraries 

# To support both python 2 and python 3
# from __future__ import division, print_function, unicode_literals

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axis3d
from mpl_toolkits import mplot3d 
import os
import sys
import numpy.random as rnd

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 
rnd.seed(4)
m = 5
w1, w2 = 0.1, 0.3
noise = 0.1

angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
data = np.empty((m, 3))
data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m) / 2
data[:, 1] = np.sin(angles) * 0.7 + noise * rnd.randn(m) / 2
data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * rnd.randn(m)

print(f"np.array([data[:, 1]])\ndata is:\n{data}")
x=np.array([[i for i in range(1,11)]])
y=np.array([[i for i in range(5,15)]])
z=np.array([[i for i in range(4,14)]])

x = np.array([data[:, 0]])
y = np.array([data[:, 1]])
z = np.array([data[:, 2]])
fig = pyplot.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_wireframe(x, y, z)
# ax1.plot_wireframe(data[:, 0], data[:, 1], data[:, 2])
ax1.set_xlabel("X axis")
ax1.set_ylabel("Y axis")
ax1.set_zlabel("Z axis")
pyplot.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(data[:m//2])
X_test = scaler.transform(data[m//2:])
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    

reset_graph()

n_inputs = 3
n_hidden = 2  # codings
n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train})
    codings_val = codings.eval(feed_dict={X: X_test})
fig = plt.figure(figsize=(4,3))
plt.plot(codings_val[:,0], codings_val[:, 1], "b.")
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
# save_fig("linear_autoencoder_pca_plot")
plt.show()
