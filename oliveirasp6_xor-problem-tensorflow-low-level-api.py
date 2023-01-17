import numpy as np
import tensorflow as tf
binary_i = np.array([[0,0], [0,1], [1,0], [1,1]])
binary_o = np.array([[0], [1], [1], [0]])
X = tf.placeholder(tf.float32, shape=(None, 2), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
n_inputs = 2
n_hidden1 = 10
n_outputs = 1
hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1-new",
                          activation=tf.nn.sigmoid)
logits = tf.layers.dense(hidden1, n_outputs, name="outputs-new")
eta = 0.1
n_epochs = 500

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
#train = tf.train.AdamOptimizer(eta).minimize(cost)
optimizer = tf.train.AdamOptimizer(eta)
training_op = optimizer.minimize(cost)
pred = tf.nn.sigmoid(logits)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_epochs):
        sess.run(training_op, feed_dict={X: binary_i, y: binary_o})
    print("Treino completo!")
    
    prediction = sess.run(pred, feed_dict={X: binary_i})
    print()
    print(prediction)
    print("\nValores arredondados\n")
    print(np.round(prediction))
