import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from numpy.random import seed, shuffle
def fetch_batches(X, y, n_batches, set_seed=None):
    """
    Retrieve the i-th batch from a random shuffled
    training dataset. Each epoch the training
    dataset gets reshuffled.
    """
    seed(set_seed)
    batches = np.c_[y, X]
    shuffle(batches)
    batches = np.array_split(batches, n_batches)
    for batch in batches:
        yield batch[:, 1:], batch[:, 0].ravel().tolist()
mnist = pd.read_csv("../input/train.csv")
X_train = mnist.iloc[:, 1:].values / 255
y_train = mnist.iloc[:, 0].values
tf.reset_default_graph()
n_input = 28 * 28
n_hidden1 = 300
n_hidden2 = 200
n_hidden3 = 100
n_output = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_input), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("DNN"):
    hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.elu, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, activation_fn=tf.nn.elu, scope="hidden2")
    hidden3 = fully_connected(hidden2, n_hidden3, activation_fn=tf.nn.elu, scope="hidden3")
    outputs = fully_connected(hidden3, n_output, activation_fn=None, scope="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=outputs)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

with tf.name_scope("acccuracy"):
    # Asses whether the highest index in outputs corresponds
    # to the index given by y
    correct = tf.nn.in_top_k(outputs, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 600
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1, n_epochs + 1):
        for X_batch, y_batch in fetch_batches(X_train, y_train, batch_size):
            sess.run(train_step, feed_dict={X:X_batch, y:y_batch})
        epoch_acc = sess.run(accuracy, feed_dict={X:X_train, y:y_train.ravel().tolist()})
        end = "\n" if epoch % 50 == 0 else "\r"
        print(f"@Epoch {epoch}. Train accuracy: {epoch_acc:0.3%}", end=end)
    save_path = saver.save(sess, "./tmp/mnist_model.ckpt")
X_test = pd.read_csv("../input/test.csv").values / 255
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./tmp/mnist_model.ckpt")
    y_hat_logits = sess.run(outputs, feed_dict={X: X_test})
y_pred = np.argmax(y_hat_logits, axis=1)
y_pred_df = pd.DataFrame(dict(Label=y_pred))
y_pred_df.index = range(1, len(y_pred) + 1)
y_pred_df.index.name = "ImageId"
y_pred_df.to_csv("MNIST_test_pred.csv", header=True)