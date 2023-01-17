import kagglegym

# Create environment

env = kagglegym.make()

# Get first observation

observation = env.reset()
observation.train.dropna().shape
for col in observation.train.columns:

    print(col)
observation.train["y"].hist(bins=100)
observation.train.dropna()["technical_5"].hist(bins=100)
import tensorflow as tf

print(tf.__version__)
import tensorflow.contrib.layers as layers

import tensorflow.contrib.losses as losses



N_FEATURES=108

LEARNING_RATE = 0.001



x = tf.placeholder(tf.float32, shape=(None, N_FEATURES))

y = tf.placeholder(tf.float32, shape=(None,1))

p = tf.placeholder(tf.float32)

logits = layers.fully_connected(x, 56, activation_fn=tf.nn.relu)

logits = layers.dropout(logits, keep_prob=p)

logits = layers.fully_connected(x, 56, activation_fn=tf.nn.relu)

logits = layers.dropout(logits, keep_prob=p)

y_ = layers.fully_connected(logits, 1)



loss = losses.mean_squared_error(y, y_)



# loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-y_)) / tf.square(y_ - tf.reduce_mean(y_))) # Equivalent to minimize R2



train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
from sklearn.cross_validation import train_test_split

traindf, testdf = train_test_split(observation.train.drop(axis=1, labels=["id", "timestamp"]).dropna(),

                                  train_size=0.8,

                                  test_size=0.2)



Y_train = traindf["y"]

X_train = traindf.drop(axis=1, labels=["y"])



Y_test = testdf["y"]

X_test = testdf.drop(axis=1, labels=["y"])
num_examples = X_train.shape[0]

batch_size = 32

n_epoch = 2

n_batch = int(num_examples / batch_size)

print("Feeding {} batches per epoch".format(n_batch))

start = 0



sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())



for _ in range(n_epoch):

    start = 0

    for batch_idx in range(n_batch-1):

    #for batch_idx in range(15):

        feeding_dict = { x: X_train.iloc[start:(start+batch_size)].values,

                        y: Y_train.iloc[start:(start+batch_size)].values.reshape(-1, 1),

                       p:0.5}

        start+=batch_size



        _, l  = sess.run([train_op, loss], feed_dict=feeding_dict)



        if not(batch_idx%1000):

            print("Loss on batch {}: {}".format(batch_idx, l))

    
import tensorflow.contrib.metrics as metrics



smse, smse_update_op = metrics.streaming_mean_squared_error(y, y_)



num_examples = X_test.shape[0]

batch_size = 32

n_batch = int(num_examples / batch_size)

print("Feeding {} batches per epoch".format(n_batch))

start = 0



sess.run(tf.initialize_local_variables())



for batch_idx in range(n_batch-1):

    feeding_dict = { x: X_test.iloc[start:(start+batch_size)].values,

    y: Y_test.iloc[start:(start+batch_size)].values.reshape(-1, 1),

                   p:1.}

    start+=batch_size

    sess.run(smse_update_op, feed_dict=feeding_dict)

print("Total loss: {}".format(sess.run(smse)))