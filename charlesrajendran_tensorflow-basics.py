import tensorflow as tf
print(tf.__version__)
# tensorflow constant and operations
a = tf.constant(10)
b = tf.constant(20)

z = a + b

with tf.Session() as sess:
    print(sess.run(z))
# tensorflow utitlity functions
ten_matrix = tf.fill((3, 3), 20)
zero_matrix = tf.zeros((2, 2))
ones_matrix = tf.ones((4, 4))

operations = [ten_matrix, zero_matrix, ones_matrix]

# this is only valid for jupyter notebooks and in other cases you have to use tf.Session
sess = tf.InteractiveSession()
for op in operations:
    print(sess.run(op), '\n')
# distributions
uniform_dist = tf.random_uniform(shape=(4, 4), minval=0, maxval=1)
sess.run(uniform_dist)
# matrix multiplication
a_mat = tf.constant([[1, 2], [3, 4]])
b_mat = tf.constant([[1],[2]])

# 2 * 2 and 2 * 1 - 
#print(a_mat.get_shape())
result_mat = tf.matmul(a_mat, b_mat)

sess.run(result_mat)
# using a different graph
graphTwo = tf.Graph()
with graphTwo.as_default():
    print(graphTwo is tf.get_default_graph()) # return true

print(graphTwo is tf.get_default_graph()) # return false
# simplest perceptron type of model
# placeholder will have bucket to put the data, shape will mostly - (None, feature numbers) - the first argument is none becuase we dont know the number of records or records per batch

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = tf.add(a, b)
mul_op = tf.multiply(a, b)

# key of feed dict - placeholder value
# value of feed dict - numbers
add_result = sess.run(add_op, feed_dict={a:10, b: 20})
mul_result = sess.run(mul_op, feed_dict={a:10, b: 20})

print(add_result, mul_result)
import tensorflow as tf
import numpy as np
n_features = 10
n_dense_nuerons = 3
X = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features, n_dense_nuerons]))
b = tf.Variable(tf.ones([n_dense_nuerons]))
XW = tf.matmul(X, W)
Z = tf.add(XW, b)

#activation function
a = tf.sigmoid(Z)
# provide initial values to weight variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # input X is 10 records with number of feature 
    layer_out = sess.run(a, feed_dict={X: np.random.random([10, n_features])})
    
print(layer_out)