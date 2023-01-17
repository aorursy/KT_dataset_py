import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



import pandas as pd

import numpy as np



# to make this notebook's output stable across runs

def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)
df = pd.read_csv("../input/Iris.csv")

print(df.head())

encoder = LabelEncoder()

labels = encoder.fit_transform(df['Species'])

# Split the dataset into 2/3 training data and 1/3 test data

train, test, train_y, test_y = train_test_split(df.iloc[:, 1:5].values, labels, test_size = 0.33)

#train, val, train_y, val_y = train_test_split(tv, tv_y, test_size = 0.2)



print("training set:{} ".format(train.shape))

#print("val set:{}".format(val.shape))

print("test set:{} ".format(test.shape))


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)

for train_index, test_index in split.split(df, df["Species"]):

    strat_train_set = df.loc[train_index]

    strat_test_set = df.loc[test_index]



print(strat_train_set.shape)

unique, counts = np.unique(strat_train_set['Species'], return_counts=True)

print(unique)

print(counts)

print(dict(zip(unique, counts)))
reset_graph()



def leaky_relu(z, name=None):

    return tf.maximum(0.01 * z, z, name=name)

    

n_examples,n_inputs = train.shape 

n_hidden1 = 20

n_hidden2 = 15

n_outputs = 3



X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

y = tf.placeholder(tf.int64, shape=(None), name="y")



with tf.name_scope("dnn"):

    hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")

    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=leaky_relu, name="hidden2")

    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    

with tf.name_scope("loss"):

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(xentropy, name="loss")    

    

learning_rate = 0.01



with tf.name_scope("train"):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    training_op = optimizer.minimize(loss)    

    

with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1)

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    

init = tf.global_variables_initializer()

saver = tf.train.Saver()   



n_epochs = 50

batch_size = 10



means = np.mean(train, axis=0)

stds = np.std(train, axis=0) + 1e-10



with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        idxs = np.random.permutation(n_examples) #shuffled ordering

        X_random = train[idxs]

        Y_random = train_y[idxs]

        for i in range(n_examples // batch_size):

            X_batch = X_random[i * batch_size:(i+1) * batch_size]

            y_batch = Y_random[i * batch_size:(i+1) * batch_size]

            X_batch_scaled = (X_batch - means) / stds

            sess.run(training_op, feed_dict={X: X_batch_scaled, y: y_batch})

            if epoch % 5 == 0:

                acc_train = accuracy.eval(feed_dict={X: X_batch_scaled, y: y_batch})

                val_scaled = (test - means) / stds

                acc_test = accuracy.eval(feed_dict={X: val_scaled, y: test_y})

                print(epoch, "Batch accuracy:", acc_train, "Validation accuracy:", acc_test)



    save_path = saver.save(sess, "./my_model_final_selu.ckpt")