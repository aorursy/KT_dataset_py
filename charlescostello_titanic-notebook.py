# Imports

import numpy as np

import pandas as pd

import tensorflow as tf

from math import isnan

from sklearn import preprocessing

from IPython.display import display



# Datasets

train = pd.read_csv("../input/train.csv")

train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})

test = pd.read_csv("../input/test.csv")

test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
display(train.head())

display(test.head())
# Create train set

train_x = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()

train_y = train[['Survived']].as_matrix()

train_mean_age = np.mean([x for x in train_x[:,2] if not isnan(x)])

train_x[:,2] = [(train_mean_age if isnan(x) else x) for x in train_x[:,2]]

train_y = np.reshape(train_y, [891])

train_x = preprocessing.scale(train_x)



print('train x: ' + str(train_x.shape))

print(train_x[:5])

print('\ntrain y: ' + str(train_y.shape))

print(train_y[:5])



# Create test set

test_x = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].as_matrix()

test_y = test[['PassengerId']].as_matrix()

test_mean_age = np.mean([x for x in test_x[:,2] if not isnan(x)])

test_x[:,2] = [(test_mean_age if isnan(x) else x) for x in test_x[:,2]]

test_x[:,5] = [(0 if isnan(x) else x) for x in test_x[:,2]]

test_x = preprocessing.scale(test_x)



print('\ntest x: ' + str(test_x.shape))

print(test_x[:5])

print('\ntest y: ' + str(test_y.shape))

print(test_y[:5])
from sklearn import svm

clf = svm.SVC()

clf.fit(train_x, train_y)

preds = clf.predict(test_x)

print(clf.score(train_x, train_y))
train_y_one_hot = np.ndarray([train_y.shape[0], 2])

train_y_one_hot[:,0] = [(1 if x == 0 else 0) for x in train_y]

train_y_one_hot[:,1] = [x for x in train_y]
# Variables

hidden_nodes = 4

output_nodes = 2

reg_constant = 4e-5



# Graph

graph = tf.Graph()

with graph.as_default():

    # Data

    train_x_tf = tf.constant(train_x, tf.float32)

    train_y_tf = tf.constant(train_y_one_hot, tf.float32)

    test_x_tf = tf.constant(test_x, tf.float32)



    # Weights and biases

    hidden_weights = tf.Variable(tf.random_normal([train_x.shape[1], hidden_nodes], stddev=0.1))

    hidden_bias = tf.Variable(tf.zeros(hidden_nodes))

    output_weights = tf.Variable(tf.random_normal([hidden_nodes, output_nodes], stddev=0.1))

    output_bias = tf.Variable(tf.zeros(output_nodes))

    

    # Output

    def model(data, dropout):

        hidden = tf.nn.relu(tf.matmul(data, hidden_weights) + hidden_bias)

        return tf.matmul(hidden, output_weights) + output_bias

                

    train_output = model(train_x_tf, True)

    test_output = model(test_x_tf, False)

    

    # Regularization

    reg = reg_constant * (tf.nn.l2_loss(hidden_weights) + 

                          tf.nn.l2_loss(hidden_bias) +

                          tf.nn.l2_loss(output_weights) +

                          tf.nn.l2_loss(output_bias))

    

    # Loss

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

        train_output, train_y_tf) + reg)



    # Optimizer

    optimizer = tf.train.AdamOptimizer().minimize(loss)



    # Predictions

    train_preds = tf.nn.softmax(train_output)

    test_preds = tf.nn.softmax(test_output)



# Session

with tf.Session(graph=graph) as session:

    tf.initialize_all_variables().run()

    

    for i in range(10000):

        _, l, preds = session.run([optimizer, loss, train_preds])

             

    print(sum(np.argmax(preds, 1) == train_y) / len(preds))

    test_preds_eval = test_preds.eval()
predictions = np.zeros([test_x.shape[0], 2], np.int16)

predictions[:,0] = test_y[:,0]

#predictions[:,1] = preds

predictions[:,1] = np.argmax(test_preds_eval, 1)

np.savetxt('preds.csv', 

           predictions, 

           fmt='%i', 

           delimiter=',', 

           header='PassengerId,Survived', 

           comments='')