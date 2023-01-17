# Import the TensorFlow library

import tensorflow as tf
# Build a simple Computational Graph

node1 = tf.constant(3.0, dtype=tf.float32)

node2 = tf.constant(4.0)



# Print the Computational Graph Built

print(node1, node2)
# Run the Computational Graph Built

sess = tf.Session() # Note: uppercase S



# Print the values

print(sess.run([node1, node2]))
# Build and Run an Operation Node

node3 = tf.add(node1, node2)

print("node3: ", node3)

print("sess.run(node3)", sess.run(node3))
# A Graph can also be parameterized to accept external inputs

# These external inputs are called placeholders

# A placeholder is a promise to provide a value later

a = tf.placeholder(tf.float32)

b = tf.placeholder(tf.float32)

adder_node = a + b # + is a shortcut which is equivalent to tf.add(a, b)



# These 3 lines are only a definition of an operation

# The actual operation happens when we provide the inputs and run it

# Below we feed in the inputs to the tensors using feed_dict

print(sess.run(adder_node, {a: 3, b:4.5}))

print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
# Adding another operation (multipley by 3) and running it

add_and_triple = adder_node * 3.

print(sess.run(add_and_triple, {a: 3, b: 4.5}))
# To have trainable inputs to a graph

# We have to provide a variable to the graph

# These variable allows us to provide trainable parameters to the graph

# The inputs given within the trainable variables will train the model we build

# A variable to the graph is delcared with an initial value and it's data type

W = tf.Variable([.3], dtype=tf.float32)

b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W * x + b



# In the previous lines we have declared variables and a placeholder

# How ever the variables are not yet initialized

# To initialize a variable we have to call an initializer method

init = tf.global_variables_initializer()

sess.run(init)



# Let us evalieate the linear model we have built

# Where x will be our input to the model

# We can supply several values of x and execute the model

print(sess.run(linear_model, {x:[1, 2, 3, 4]}))
# To evaluate the linear model on the training dataset we need a y placeholder

# The y placeholder will take in the desired values

# We will also write a loss function

# The loss function calculates how far apart the model is from the provided data

# The standard loss model for a linear regression is as below

# Sums the squares of deltas between current model and provided data

y = tf.placeholder(tf.float32)

squared_deltas = tf.square(linear_model - y)

loss = tf.reduce_sum(squared_deltas)

print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# Gradient Descent Optimizer in use to minimize loss

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)



sess.run(init)

for i in range(1000):

    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

    

print(sess.run([W, b]))