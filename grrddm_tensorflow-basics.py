import tensorflow as tf
import pandas as pd
import numpy as np
from math import ceil
from numpy.random import seed, shuffle
# Defining a graph (a default graph)
# Every declared node is automatically added 
# to the 'default-graph'
x = tf.Variable(3, name="x")
y = tf.Variable(5, name="y")
f = x * x * y + y + 2
# Run the graph inside a session.
# With the 'with' command, the session is set as the
# default sesion
with tf.Session() as sess:
    # We need to initialize the variables before
    # performing operations using them
    x.initializer.run() # Equivalent to: tf.get_default_session().run(x.initializer)
    y.initializer.run() # Equivalent to: tf.get_default_session().run(y.initializer)
    result = f.eval()   # Equivalent to: tf.get_default_session().run(f)
    
print(result)
# Removing every node inside the
# default graph
tf.reset_default_graph()

# **CONSTRUCTION PHASE**
x = tf.Variable(3, name="x")
y = tf.Variable(5, name="y")
f = x * x * y + y + 2
# Add to the graph a step to initialize all variables
# (we are not actually initializing the variables in this step)
init = tf.global_variables_initializer()

# **EXECUTION PHASE**
with tf.Session() as ses:
    init.run()
    result = f.eval()

print(result)
tf.reset_default_graph()
# Whenever we evaluate a node, Tensorflow automatically
# determines the set of nodes that it depends on.

w  = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

init = tf.global_variables_initializer()
# **The ineficient way to evaluate a set of nodes**
# By evaluating (y, z) (nodes) the following way, TensorFlow
# has to compute 'w' and 'x' twice in order to obtain (x, z)
with tf.Session() as sess:
    init.run()
    print(y.eval())
    print(z.eval())
    
tf.reset_default_graph()
w  = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

print()
init = tf.global_variables_initializer()
# **The proper way to evaluate a set of nodes**
with tf.Session() as sess:
    init.run()
    xres, yres = sess.run([y, z])
    print(xres)
    print(yres)
tf.reset_default_graph()
housing = pd.read_csv("../input/housing.csv").dropna()
m, n = housing.shape
housing_data_bias = np.c_[np.ones((m, 1)), housing.drop(["ocean_proximity", "median_house_value"], axis=1).values]
housing_target = housing.median_house_value.values.reshape(-1, 1)

# Defining the computation graph 
X = tf.constant(housing_data_bias, dtype=tf.float32, name="X")
y = tf.constant(housing_target, dtype=tf.float32, name="y")
Xtranspose = tf.transpose(X)
theta = tf.matrix_inverse(Xtranspose @ X) @ Xtranspose @ y

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #print(sess.list_devices())
    theta_star = theta.eval() # Equivalent to sess.run(theta)
from sklearn.preprocessing import StandardScaler
tf.reset_default_graph()

housing_data_bias_scaled = StandardScaler().fit_transform(housing_data_bias)

n_epochs = 1000
learning_rate = 0.1
X = tf.constant(housing_data_bias_scaled, dtype=tf.float32, name="X")
y = tf.constant(housing.median_house_value.values.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n - 1, 1], minval=-1, maxval=1), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="MSE")
# **Taking a step of gradient descent**
# ---------------- Manual Differentiation----------------
gradients = 2 / m *tf.matmul(tf.transpose(X), error)
# Updating the parameters
training_op = tf.assign(theta, theta - learning_rate * gradients)
# ---------------- Manual Differentiation----------------

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Step 1) Initialize all variables
    sess.run(init)
    # Step 2) Iterate over epochs
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            # Evaluate current mse variable
            print(f"@Epoch {epoch:03}, MSE: {mse.eval():0,.2f}")
        # update the training operation: compute gradients
        sess.run(training_op)
    theta_star = theta.eval()
from sklearn.preprocessing import StandardScaler
tf.reset_default_graph()

housing_data_bias_scaled = StandardScaler().fit_transform(housing_data_bias)

n_epochs = 1000
learning_rate = 0.1
X = tf.constant(housing_data_bias_scaled, dtype=tf.float32, name="X")
y = tf.constant(housing.median_house_value.values.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n - 1, 1], minval=-1, maxval=1), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="MSE")
# **Taking a step of gradient descent**
# ---------------- Automatic Differentiation----------------
gradients = tf.gradients(mse, [theta])[0]
# Updating the parameters
training_op = tf.assign(theta, theta - learning_rate * gradients)
# ---------------- Automatic Differentiation----------------

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Step 1) Initialize all variables
    sess.run(init)
    # Step 2) Iterate over epochs
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            # Evaluate current mse variable
            print(f"@Epoch {epoch:03}, MSE: {mse.eval():0,.2f}")
        # update the training operation: comppute gradients
        sess.run(training_op)
    theta_star = theta.eval()
from sklearn.preprocessing import StandardScaler
tf.reset_default_graph()

housing_data_bias_scaled = StandardScaler().fit_transform(housing_data_bias)

n_epochs = 1000
learning_rate = 0.1
X = tf.constant(housing_data_bias_scaled, dtype=tf.float32, name="X")
y = tf.constant(housing.median_house_value.values.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n - 1, 1], minval=-1, maxval=1), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="MSE")
# **Taking a step of gradient descent**
# ---------------- TF Optimizer ----------------
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
# ---------------- TF Optimizer ----------------

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Step 1) Initialize all variables
    sess.run(init)
    # Step 2) Iterate over epochs
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            # Evaluate current mse variable
            print(f"@Epoch {epoch:03}, MSE: {mse.eval():0,.2f}")
        # update the training operation: comppute gradients
        sess.run(training_op)
    theta_star = theta.eval()
from itertools import product
tf.reset_default_graph()
batch_size = 500
n_epochs = 1000
n_batches = ceil(m / batch_size)
def fetch_batch(epoch, batch_index, batch_size):
    """
    Retrieve the ith batch from a random shuffled
    training dataset. Each epoch the training
    dataset gets reshuffled.
    """
    seed(epoch)
    batches = np.c_[housing_data_bias_scaled, housing_target]
    shuffle(batches)
    batches = np.array_split(batches, n_batches)
    batch_ix = batches[batch_index]
    return batch_ix[:, :-1], batch_ix[:, -1].reshape(-1, 1)

X = tf.placeholder(tf.float32, shape=[None, n - 1], name="X")
y = tf.placeholder(tf.float32, shape=[None, 1], name="y")
theta = tf.Variable(tf.random_uniform([n - 1, 1], minval=-1, maxval=1), name="theta")

y_pred = tf.matmul(X, theta)
err = y_pred - y
mse = tf.reduce_mean(tf.square(err), name="MSE")
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_ix in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_ix, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 100 == 0:
            # Evaluate current mse variable
            print(f"@Epoch {epoch:03}, MSE: {mse.eval(feed_dict={X: X_batch, y: y_batch}):0,.2f}")
    theta_star_bgd = theta.eval()