# minimize the function J = w^2 - 10*w + 25 using tensorflow

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# defining the parameter w
w = tf.Variable(0, dtype=tf.float32)

# defining the cost function
cost = w**2 - 10*w + 25 # tensorflow creates a computatitonal graph

# learning rate = 0.01
# goal is to optimize cost
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))   # evaluates w

# run one step of gradient descent and then print the value of w
session.run(train)
print(session.run(w))
# thousand iterations of gradient descent
for i in range(1000):
    session.run(train)
print(session.run(w)) # very close to 5

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

coefficients = np.array([[1],[-10],[25]]) # data that will be used by the cost function

# defining the parameter w
w = tf.Variable(0, dtype=tf.float32)

# a placeholder is a variable whose values are assigned later
x = tf.placeholder(tf.float32,[3,1]) # 3x1 array

#putting the coefficients of cost function in x
# defining the cost function
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0]

# learning rate = 0.01
# goal is to optimize cost
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
session = tf.Session()

#feeding data to function
session.run(init ,feed_dict = {x:coefficients})
print(session.run(w))   # evaluates w
# run one step of gradient descent and then print the value of w
session.run(train,feed_dict = {x:coefficients})
print(session.run(w))
# thousand iterations of gradient descent
for i in range(1000):
    session.run(train,feed_dict = {x:coefficients})
print(session.run(w)) # very close to 5
