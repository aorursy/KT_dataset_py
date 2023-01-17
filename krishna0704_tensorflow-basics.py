import tensorflow as tf
hello_world = tf.constant('Hello World!', name='hello_world')

print(hello_world)
# Rank 0 tensor

A = tf.constant(37)



# Rank 1 tensor

B = tf.constant([23, 37, 73])



# Rank 2 tensor

C = tf.constant([[2, 3, 5], [5, 4, 1]])
A, B, C
x = tf.constant(5, name='x')

y = tf.Variable(x+5, name='y')

print(y)
# Addition

x = tf.add(1,2)



# Subtract

y = tf.subtract(2,4)



# Multiply

z = tf.multiply(3,7)
x, y, z
tf.compat.v1.truncated_normal((5,3))
# Softmax

'''Softmax converts its inputs to be between 0 and 1, 

and also normalizes the outputs so that they all sum up to 1.

'''

inputs = [2.0, 1.5, 1.0]

softmax = tf.nn.softmax(inputs)

print(softmax)
tf.reduce_sum([1,2,3])
softmax_data = [0.1,0.5,0.4]

onehot_data = [0.0,1.0,0.0]



cross_entropy = -tf.reduce_sum(tf.multiply(onehot_data, tf.math.log(softmax)))

print(cross_entropy)
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=tf.math.log(softmax), labels=onehot_data)

cross_entropy_loss