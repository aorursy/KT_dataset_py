# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#importing data

import os

CSV_PATH = "../input"

train = pd.read_csv(os.path.join(CSV_PATH, "train.csv"))

test = pd.read_csv(os.path.join(CSV_PATH, "test.csv"))



from sklearn import preprocessing

minMax = preprocessing.MinMaxScaler()

train[train.columns.difference(['label'])] = minMax.fit_transform(train[train.columns.difference(['label'])])

test = pd.DataFrame(minMax.fit_transform(test))



train[train.columns.difference(['label'])].head(20)
#network params

n_inputs = 28*28  # MNIST

n_hidden1 = 300

n_hidden2 = 100

n_outputs = 10



learning_rate = 0.01



n_epochs = 40

batch_size = 50
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

y = tf.placeholder(tf.int64, shape=(None), name="y")
def neuron_layer(X, n_neurons, name, activation=None):

    with tf.name_scope(name):

        n_inputs = int(X.get_shape()[1])

        stddev = 2 / np.sqrt(n_inputs)

        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)

        W = tf.Variable(init, name="kernel")

        b = tf.Variable(tf.zeros([n_neurons]), name="bias")

        Z = tf.matmul(X, W) + b

        if activation is not None:

            return activation(Z)

        else:

            return Z
with tf.name_scope("dnn"):

    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",

                           activation=tf.nn.elu)

    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",

                           activation=tf.nn.elu)

    logits = neuron_layer(hidden2, n_outputs, name="outputs")
with tf.name_scope("loss"):

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,

                                                              logits=logits)

    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("train"):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1)

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        for iteration in range(len(train.index) // batch_size):

            batch = train.sample(n=batch_size)

            X_batch = batch.drop("label", axis=1)

            Y_batch = batch["label"]

            sess.run(training_op, feed_dict={X: X_batch, y: Y_batch})

        acc_train = accuracy.eval(feed_dict={X: X_batch, y: Y_batch})

        acc_test = accuracy.eval(feed_dict={X: X_train,

                                            y: Y_train})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        

    save_path = saver.save(sess, "./my_model_final.ckpt")
with tf.Session() as sess:

    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path

    Z = logits.eval(feed_dict={X: test})

    y_pred = np.argmax(Z, axis=1)
from IPython.display import clear_output, Image, display, HTML



def strip_consts(graph_def, max_const_size=32):

    """Strip large constant values from graph_def."""

    strip_def = tf.GraphDef()

    for n0 in graph_def.node:

        n = strip_def.node.add() 

        n.MergeFrom(n0)

        if n.op == 'Const':

            tensor = n.attr['value'].tensor

            size = len(tensor.tensor_content)

            if size > max_const_size:

                tensor.tensor_content = b"<stripped %d bytes>"%size

    return strip_def



def show_graph(graph_def, max_const_size=32):

    """Visualize TensorFlow graph."""

    if hasattr(graph_def, 'as_graph_def'):

        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)

    code = """

        <script>

          function load() {{

            document.getElementById("{id}").pbtxt = {data};

          }}

        </script>

        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>

        <div style="height:600px">

          <tf-graph-basic id="{id}"></tf-graph-basic>

        </div>

    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))



    iframe = """

        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>

    """.format(code.replace('"', '&quot;'))

    display(HTML(iframe))
submission = pd.DataFrame({

        "ImageId": list(range(1, len(y_pred)+1)),

        "Label": y_pred

    })

submission.to_csv('mnist.csv', index=False)