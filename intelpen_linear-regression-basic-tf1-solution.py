#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
#training data
x_train = np.array([50, 100.,150., 200])
y_train = np.array([100., 150.,180.,200.])
W = tf.Variable([0.3], tf.float32)
b= tf.Variable([80.0], tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W*x +b
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.00000003);
train = optimizer.minimize(loss)
#To initialize the variables we  call a special operation in TF :
init = tf.global_variables_initializer()

my_session = tf.Session()
my_session.run(init)

for i in range (10000):
    my_session.run(train, {x:x_train, y : y_train})
#evaluate training error
curr_W, curr_b, curr_loss = my_session.run([W, b, loss], {x:x_train, y:y_train})
print("W: "+str(curr_W)+" , b: "+str(curr_b)+ ", loss: " + str(curr_loss))