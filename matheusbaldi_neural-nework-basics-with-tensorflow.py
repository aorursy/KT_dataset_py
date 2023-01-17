# For preparing and analyzing the dataset
import pandas as pd
import numpy as np
%matplotlib inline

from time import time

# For creating the model
import tensorflow as tf
# Open file

data = pd.read_csv("../input/mushrooms.csv")
samples = data.shape[0]
features = data.shape[1]
print(f"{samples} samples and {features} features")
data.head(5)
data.info()
# Casting the column types to categorical.
columns = list(data.columns)

# For that, we will need to create a dictionary where the keys are
# the name of the column and the values are strings with the type we want.
columns_dtype = {column:"category" for column in columns}

data = data.astype(columns_dtype)

# Getting category codes
data = data[columns].apply(lambda x: x.cat.codes)
# Shuffle data
data = data.sample(frac=1, random_state=3)
data.reset_index(inplace=True, drop=True)
data.head(5)
# One-hot labels
# Create dataframe with the labels
Y_labels = pd.DataFrame(data['class'].copy())
# Create dataframe with the data
X_data = pd.DataFrame(data.iloc[:,1:])

# Create 'class2' column that is the opposite of 'class' column
Y_labels['class2'] = Y_labels['class'] == 0
# Cast 'class' values to bool
Y_labels['class'] = Y_labels['class'].astype('bool', copy=True)

Y_labels.reset_index(inplace=True, drop=True)
Y_labels.head()
# Normalize columns
for column in X_data.columns:
    mean = X_data[column].mean()
    standard_deviation = X_data[column].std()
    X_data[column] = (X_data[column] - mean) / (standard_deviation) 
X_data = X_data.fillna(0)
#X_data.drop(columns="veil-type", inplace=True)
# Calculating splits
train_size = int(8124*.60)
dev_size = (8124 - train_size) // 2
test_size = (8124 - (train_size+dev_size))
print(train_size, dev_size, test_size)
X_train = X_data.values[:train_size,:]
Y_train = Y_labels.values[:train_size,:]
X_dev = X_data.values[train_size:train_size+dev_size,:]
Y_dev = Y_labels.values[train_size:train_size+dev_size,:]
X_test = X_data.values[train_size+dev_size:,:]
Y_test = Y_labels.values[train_size+dev_size:,:]
def parameters(shape, layer_name):
    """ Initialize the parameters W and b of a layer with name ´layer_name´ and a shape ´shape´. """
    W = tf.get_variable(shape=shape, initializer=tf.glorot_normal_initializer(3),
                        regularizer=tf.contrib.layers.l2_regularizer(0.2), name=layer_name+"_weight")
    b = tf.Variable(tf.constant(0.1, shape=[shape[1]]), name=layer_name+"bias")
    return W,b
        
def linear_op(W,X,b):
    """ Run a linear function ´X´ * ´W´ + ´b´ """
    linear = tf.add(tf.matmul(X,W), b)
    return linear

def activation(linear, activate):
    """ Apply an activation function ´activate´ in the value passed to ´linear´ """
    activation = activate(linear)
    return activation

def layer(X, shape, layer_name, activate=tf.nn.relu):
    """ Create a neural network layer """
    W, b = parameters(shape, layer_name)
    linear = linear_op(W,X,b)
    activation = activate(linear)
    return activation

def basic_model(X):
    """ Create a neural network model in the current Graph """
    graph = tf.get_default_graph()
    with graph.as_default():
        l = layer(X, [22, 32], 'hidden1')
        l = layer(l, [32, 16], 'hidden3')
        l = layer(l, [16, 8], 'hidden4')
        output = layer(l, [8, 2], 'output', activate=tf.identity)
    
    return output
def other_model(X):
    """ Create a neural network model in the current Graph with tensorflow layers """
    graph = tf.get_default_graph()
    with graph.as_default():
        l = tf.keras.layers.Input(shape=(22,), tensor=X)
        l = tf.layers.dense(l, 32, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer(3),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.2))
        l = tf.layers.dense(l, 16, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer(3),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.2))
        l = tf.layers.dense(l, 8, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer(3),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.2))
        output = tf.layers.dense(l, 2, activation=tf.identity, kernel_initializer=tf.glorot_normal_initializer(3),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.2))

    return output
def simpler_model(X):
    """ Create a simple neural network model in the current Graph with tensorflow layers """
    graph = tf.get_default_graph()
    with graph.as_default():
        l = tf.keras.layers.Input(shape=(22,), tensor=X)
        output = tf.layers.dense(l, 2, activation=tf.identity, kernel_initializer=tf.glorot_normal_initializer(3))

    return output
batch = 128

batches = []
for j in range(0, X_train.shape[0], batch):
    X_train_batch = X_train[j:j+batch,:]
    Y_train_batch = Y_train[j:j+batch,:]
    batches.append((X_train_batch, Y_train_batch))

batches = tuple(batches)
def build_graph(model_):
    """ Create a new graph for training and evaluating models """
    graph = tf.Graph()
    with graph.as_default():
        # Input and Labels
        X = tf.placeholder(tf.float32, [None, 22], name="Input")
        Y = tf.placeholder(tf.float32, [None, 2], name="Labels")

        # Neural Network
        output = model_(X)

        # Operation for calculating accuracy
        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_predictions"):
                correct = tf.equal(tf.argmax(output,1),tf.argmax(Y,1))
            with tf.name_scope("accuracy"):
                accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

        # Cost OP
        with tf.name_scope("cost"):
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y)
            cost = tf.reduce_mean(cost,axis=0)
        # Train OP
        with tf.name_scope("optimization"):
            global_step = tf.Variable(0, trainable=False)
            initial_learning_rate = 0.03
            learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 2048, decay_rate=.97, staircase=True)
            train = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        
        # Initialize variables
        init = tf.global_variables_initializer()        
    
        return graph, init, X, Y, learning_rate, cost, accuracy, train
epochs = 64
mdls = [basic_model, other_model, simpler_model]
for mdl in mdls:
    print(f"\nModel used: {mdl.__name__}\t\t#Epochs: {epochs}")
    graph, init, X, Y, learning_rate, cost, accuracy, train = build_graph(mdl)
    # Session
    with tf.Session(graph=graph) as sess:
        sess.run(init)

        # Train iterations
        for i in range(epochs):
            if i%32 == 0:
                lr, cos_dev, acc_dev = sess.run([learning_rate, cost, accuracy], feed_dict={X: X_dev, Y: Y_dev})
                cos_train, acc_train = sess.run([cost, accuracy], feed_dict={X: X_train, Y: Y_train})
                print(f"epoch {i:4d}: train_cost= {cos_train:0.10f} | train_accuracy= {acc_train:0.2f} || dev_cost= {cos_dev:0.10f} | dev_accuracy= {acc_dev:0.2f} | lr: {lr:0.10f}")

            # Running training steps through our mini-batches
            for X_batch, Y_batch in batches:
                # Training Step
                sess.run([train], feed_dict={X: X_batch, Y: Y_batch})
        
        # Evaluate the last learing rate and our model cost and accuracy with train, dev and test sets
        lr, cos_train, acc_train = sess.run([learning_rate, cost, accuracy], feed_dict={X: X_train, Y: Y_train})
        cos_dev, acc_dev = sess.run([cost, accuracy], feed_dict={X: X_dev, Y: Y_dev})
        cos_test, acc_test = sess.run([cost, accuracy], feed_dict={X: X_test, Y: Y_test})
        
        # Print the last training step values
        print(f"epoch {epochs-1:4d}: train_cost= {cos_train:0.10f} | train_accuracy= {acc_train:0.2f} || dev_cost= {cos_dev:0.10f} | dev_accuracy= {acc_dev:0.2f} | lr: {lr:0.10f}")
        
        # Print the performance in all sets
        print(f"\nPERFORMANCE:\ntrain_cost\t= {cos_train:0.10f} | train_accuracy\t= {acc_train:0.2f} |\ndev_cost\t= {cos_dev:0.10f} | dev_accuracy\t= {acc_dev:0.2f} |\ntest_cost\t= {cos_test:0.10f} | test_accuracy\t= {acc_test:0.2f} |")
        print("="*96)