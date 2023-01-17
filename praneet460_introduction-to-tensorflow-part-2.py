# Import required library

import tensorflow as tf

print("Tensorflow version ", tf.__version__)

# Define a hello tensor as a tf.constant

hello = tf.constant(value = "Hello, TensorFlow", name="hello")



# Tensorflow uses default Interactive Sesssion which help us use 'eval()' method



with tf.Session():

    print(hello.eval()) # Shorthand for session.run(hello)
# declare two matrices

matrix_1 = tf.constant(value = [[5, 6], [7, 8]], name="matrix_1")

matrix_2 = tf.placeholder(dtype = tf.int32, name="matrix_2")



# add two matrices

matrix_sum = tf.add(matrix_1, matrix_2, name="matrix_sum")



print(matrix_sum)

print("\n")

data = {matrix_2:[[1, 0], [0, 1]]}

with tf.Session():

    print("Matrix Sum:\n ",matrix_sum.eval(feed_dict=data)) # Shorthand of session.run(matrix_sum)
# subtract two matrices

matrix_subtract = tf.subtract(matrix_1, matrix_2, name = "matrix_subtract")

print(matrix_subtract)

print("\n")

with tf.Session():

    print("Matrix Substraction:\n ", matrix_subtract.eval(feed_dict=data))
# multiply two matrices

matrix_multiply = tf.multiply(matrix_1, matrix_2, name="matrix_multiply")

print(matrix_multiply)

print("\n")

with tf.Session():

    print("Matrix Multiplication:\n", matrix_multiply.eval(feed_dict=data))
# Import the required library

import numpy as np



# Creating a 2-d tensor, or matrix

tensor_2d = np.array(np.random.rand(4, 4), dtype="float32")

tensor_2d_1 = np.array(np.random.rand(4, 4), dtype="float32")

tensor_2d_2 = np.array(np.random.rand(4, 4), dtype="float32")





# look out the matrices

print("tensor_2d =\n", tensor_2d)

print("\n")

print("tensor_2d_1 =\n", tensor_2d_1)

print("\n")

print("tensor_2d_2 =\n", tensor_2d_2)
# converting the array into tensor

mat_1 = tf.convert_to_tensor(value = tensor_2d)

mat_2 = tf.convert_to_tensor(value = tensor_2d_1)

mat_3 = tf.convert_to_tensor(value = tensor_2d_2)



print("mat_1:\n", mat_1)

print("mat_2:\n", mat_2)

print("mat_3:\n", mat_3)
# transpose of the matrix

mat_transpose = tf.transpose(mat_1)

with tf.Session():

    print(mat_transpose.eval())
# matrix multiplication

mat_multiply = tf.matmul(mat_1, mat_2)

with tf.Session():

    print(mat_multiply.eval())
# Matrix determinant

mat_determinant = tf.matrix_determinant(mat_3)

with tf.Session():

    print(mat_determinant.eval())
# Matrix inverse

mat_inverse = tf.matrix_inverse(mat_3)

with tf.Session():

    print(mat_inverse.eval())
# Matrix solve

mat_solve = tf.matrix_solve(mat_3, [[1], [1], [1], [1]])

with tf.Session():

    print(mat_solve.eval())
result = tf.matmul(mat_1, mat_1) + tf.matmul(mat_1, tf.ones([4, 4], dtype="float32"))

with tf.Session():

    print(result.eval())