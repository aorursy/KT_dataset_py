# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv("../input/train.csv", index_col=None, header=0)

test  = pd.read_csv("../input/test.csv", index_col=None, header=0)



# Any results you write to the current directory are saved as output.
trainArr = train.as_matrix()
print(trainArr)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

         trainArr[:,1:], trainArr[:,0], test_size=0.3, random_state=50)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=150, min_samples_split=2, 

                             min_samples_leaf=2,  random_state=1)

clf.fit(X_train, y_train)
clf.score(X_test, y_test)
#sklearn neural network seems not easy to use, try tensorflow
import tensorflow as tf
def getBits(x):

    a = [0,0,0,0,0,0,0,0,0,0]

    a[x] = 1

    return a
y_NN_train = np.array(list(map(getBits, y_train)))
print(y_NN_train.shape)

print(X_train.shape)
def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

          / predictions.shape[0])
graph = tf.Graph()

with graph.as_default():

    weights = tf.Variable(np.zeros((784,10)).astype(np.float32))

    biases = tf.Variable(np.zeros([10]).astype(np.float32))



    X_train = X_train.astype(np.float32)

    y_NN_train = y_NN_train.astype(np.float32)

    tf_train_dataset = tf.constant(X_train)

    tf_train_labels = tf.constant(y_NN_train)

    logits = tf.matmul(tf_train_dataset, weights, name=None) + biases



    # TODO: convert labels from number to bits

    loss = tf.reduce_mean(

          tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

       

    train_prediction = tf.nn.softmax(logits)

    

y_train_labels = np.array(list(map(getBits, y_train)))
with tf.Session(graph=graph) as session:

  # This is a one-time operation which ensures the parameters get initialized as

  # we described in the graph: random weights for the matrix, zeros for the

  # biases. 

  init = tf.initialize_all_variables()

  session.run(init)

  print('Initialized')

  for step in range(1000):

    # Run the computations. We tell .run() that we want to run the optimizer,

    # and get the loss value and the training predictions returned as numpy

    # arrays.

    _, l, predictions = session.run([optimizer, loss, train_prediction])

    

  print('Training accuracy: %.1f%%' % accuracy(

        predictions, y_train_labels))