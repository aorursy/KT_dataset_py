# Import necessary packages

import tensorflow as tf

import numpy as np

np.random.seed(0)



print("TensorFlow Version ", tf.VERSION)

print("Numpy Version ", np.__version__)



# Take an array of random integer values

x = np.array(np.random.randint(low=100, size=(4, 4)))

y = np.array(np.random.randint(low=100, size=(5)))

z = np.array(np.random.randint(low=100, size=(5)))



# Take a look on arrays

print("X Array:\n ", x)

print("Y Array:\n ", y)

print("Z Array:\n ", z)
# Convert an array into the tensor

m1 = tf.convert_to_tensor(x)

m2 = tf.convert_to_tensor(y)

m3 = tf.convert_to_tensor(z)
arg_min = tf.argmin(m1, axis = 1) # row axis

with tf.Session():

    print(arg_min.eval())
arg_max = tf.argmax(m1, axis = 1) # row axis

with tf.Session():

    print(arg_max.eval())
unique_m2 = tf.unique(m2)



with tf.Session() as session:

    print("Unique value in m2 are ", session.run(unique_m2)[0])

    print("Unique index in m2 are ", session.run(unique_m2)[1])
diff = tf.setdiff1d(m2, m3)

with tf.Session() as session:

    print("Setdiff values are ", session.run(diff)[0])

    print("Setdiff indexes are ", session.run(diff)[1])
a = np.array(np.random.randint(low=10, size=[3, 3]))

m4 = tf.convert_to_tensor(a)

print(a)
red_sum = tf.reduce_sum(m4)

red_sum_0 = tf.reduce_sum(m4, axis=0)

red_sum_1 = tf.reduce_sum(m4, axis=1)





with tf.Session() as session:

    print("Sum of all the integer values present in m4 tensor :", session.run(red_sum))

    print("Sum of all the integer values present in individual columns of m4 tensor :",session.run(red_sum_0))

    print("Sum of all the integer values present in individual rows of m4 tensor :", session.run(red_sum_1))
red_prod = tf.reduce_prod(m4)

red_prod_0 = tf.reduce_prod(m4, axis=0)

red_prod_1 = tf.reduce_prod(m4, axis=1)

with tf.Session() as session:

    print("Product of all the integer values present in m4 tensor :", session.run(red_prod))

    print("Product of all the integer values present in individual columns of m4 tensor :", session.run(red_prod_0))

    print("Product of all the integer values present in individual rows of m4 tensor :", session.run(red_prod_1))
red_min = tf.reduce_min(m4)

red_min_0 = tf.reduce_min(m4, axis=0)

red_min_1 = tf.reduce_min(m4, axis=1)

with tf.Session() as session:

    print("Minimum value among all the integer values in m4 tensor :", session.run(red_min))

    print("Minimum value among all the integer values present in individual columns of m4 tensor :", session.run(red_min_0))

    print("Minimum value among all the integer values present in individual rows of m4 tensor :", session.run(red_min_1))
red_max = tf.reduce_max(m4)

red_max_0 = tf.reduce_max(m4, axis=0)

red_max_1 = tf.reduce_max(m4, axis=1)

with tf.Session() as session:

    print("Maximum value among all the integer values in m4 tensor :", session.run(red_max))

    print("Maximum value among all the integer values present in individual columns of m4 tensor :", session.run(red_max_0))

    print("Maximum value among all the integer values present in individual rows of m4 tensor", session.run(red_max_1))
red_mean = tf.reduce_mean(m4)

red_mean_0 = tf.reduce_mean(m4, axis=0)

red_mean_1 = tf.reduce_mean(m4, axis=1)

with tf.Session() as session:

    print("Mean of all the integer values present in m4 tensor :", session.run(red_mean))

    print("Mean of all the integer values present in individual columns of m4 tensor :", session.run(red_mean_0))

    print("Mean of all the integer values present in individual rows of m4 tensor :", session.run(red_mean_1))