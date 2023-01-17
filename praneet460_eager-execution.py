import tensorflow as tf



print("Tensorflow Version ",tf.VERSION)

# Enable eager execution



tf.enable_eager_execution()
tf.executing_eagerly() # check if eager execution is enable or not
print(tf.add(1, 2))
x = tf.constant([[1, 2], [3, 4]])

print(x)

print("\n")

print(type(x))
y = tf.add(x, 1)

print(y)
# Use numpy values

import numpy as np



z = np.multiply(x, y) # numpy takes tensors as an argument

print(z)

print("\n")

print(type(z))
# Obtain numpy value from a tensor

print(x.numpy())
a = tf.constant(12)

counter = 0

while not tf.equal(a, 1):

    if tf.equal(a % 2, 0):

        a = a / 2

    else:

        a = 3 * a + 1

    print(a)