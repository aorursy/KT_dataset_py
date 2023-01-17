import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
x_data = np.linspace(0,10,10000) # X values 
noise = np.random.randn(len(x_data))

y_true = .5 * x_data + 5 + noise # Added Noise, m is taken as .5 and b is 5, these are arbitraty values which can be changed.
x_df = pd.DataFrame(data = x_data, columns = ["X Data"])
y_df = pd.DataFrame(data = y_true, columns = ["Y"])
my_data = pd.concat([x_df, y_df], axis = 1) # DataFrame created with X and Y values.
my_data.sample(250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
batch_size = 8 # Requirement by Tensorflow for how many records are trained in one go.
rnd = np.random.randn(2)

m = tf.Variable(rnd[0], ) # purely arbitrary values.
b = tf.Variable(rnd[1], )

xph = tf.placeholder(tf.float32, [batch_size]) # placeholder equal to batch size.
yph = tf.placeholder(tf.float32, [batch_size])

y_model = tf.cast(m, tf.float32) * xph + tf.cast(b, tf.float32)

error = tf.reduce_sum(tf.square(y_model - yph)) # Error function, notice that tensorflow functions are used tf.square.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) # Gradient descent optimizer is used.
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000 # How many batches to train.
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size = batch_size) # Data Selection.
        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}
        sess.run(train, feed_dict = feed)
    model_m, model_b = sess.run([m,b])

print(model_m, model_b) # Final params learned
y_predicted = model_m * x_data + model_b
my_data.sample(250).plot(kind = 'scatter', x = 'X Data', y = 'Y')
plt.plot( x_data, y_predicted, 'r')