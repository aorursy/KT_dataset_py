# This is sample of linear model apply in tensorflow for novice.
import tensorflow as tf

# create variables w and b
W = tf.Variable([1.], dtype=tf.float32)
b = tf.Variable([-1.1], dtype=tf.float32)

x = tf.placeholder(tf.float32)
# create Line Model
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# create loss model
loss = tf.reduce_sum(tf.square(linear_model - y))
init = tf.global_variables_initializer()

# choose the optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# train data
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]

with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))
    print("W init:" + str(sess.run(W)))
  
    # repeat 4000 times
    for i in range(4000):
        sess.run(train, {x: x_train, y: y_train})
        print('epoch:%s W: %s b: %s loss: %s' % (i+1,sess.run(W),
         sess.run(b), sess.run(loss, {x: x_train , y: y_train})))

    print('W: %s b: %s loss: %s' % (sess.run(W), sess.run(b),
    sess.run(loss, {x: x_train , y: y_train})))