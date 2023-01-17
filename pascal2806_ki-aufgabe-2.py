import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

"""Daten werden in 28x28 PixelMatritzen aufgeteilt"""

traindev = pd.read_csv('../input/train.csv')
X_traindev = traindev.loc[:,'pixel0':'pixel783']
Y_traindev = traindev.loc[:,'label']

for n in range(1,10):
    plt.subplot(1,10,n)
    plt.imshow(X_traindev.iloc[n].values.reshape((28,28)),cmap='gray')
    plt.title(Y_traindev.iloc[n])
    
    """Daten werden aufgeteilt in Trainings und Validierungsdaten"""
    
    # Erstelle das Trainings Dataset für die Trainingsdaten
X_train = X_traindev[:40000].T.values
Y_train = Y_traindev[:40000]
Y_train = pd.get_dummies(Y_train).T.values

# Erstelle das Dataset für die Validierungsdaten
X_dev = X_traindev[40000:42000].T.values
Y_dev = Y_traindev[40000:42000]
Y_dev = pd.get_dummies(Y_dev).T.values

# Lese das Test Set ein
X_test = pd.read_csv('../input/test.csv').T.values

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of cross-validation examples = " + str(X_dev.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_dev shape: " + str(X_dev.shape))
print ("Y_dev shape: " + str(Y_dev.shape))
print ("X_test shape: " + str(X_test.shape))

"""Graph wird erstellt"""

def create_graph(X_train,Y_train):
    #setup
    ops.reset_default_graph()                         # reset computation graph

    # initialisiere Variablen
    (n_x, training_examples) = X_train.shape                          
    n_y = Y_train.shape[0]                            
    costs = []

    # Definiere Placeholder
    X = tf.placeholder(tf.float32, shape=(n_x, None),name = "X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None),name = "Y")
    
    # Definiere die Gewichte
    W1 = tf.get_variable("W1", [32,784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W2 = tf.get_variable("W2", [16,32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W3 = tf.get_variable("W3", [10,16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    
    # initialize biases
    b1 = tf.get_variable("b1", [32,1], initializer = tf.zeros_initializer())
    b2 = tf.get_variable("b2", [16,1], initializer = tf.zeros_initializer())
    b3 = tf.get_variable("b3", [10,1], initializer = tf.zeros_initializer())

    # Erstelle den Graph für die Forward Propagation
    Z1 = tf.add(tf.matmul(W1,X),b1)                                             
    A1 = tf.nn.relu(Z1)                                                         
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                            
    A2 = tf.nn.relu(Z2)                                                         
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    return X, Y, Z3, training_examples

"""Daten werden optimiert"""
def define_optimization(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) )
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    
    return optimizer, cost


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

"""Netzwerk wird trainiert"""

def train_network(X_train,Y_train,X_dev, Y_dev, X_test, num_epochs,minibatch_size=64,print_n_epochs=1):
    
    tf.set_random_seed(1)                             
    X,Y,Z_final,training_examples = create_graph(X_train,Y_train)
    optimizer, cost = define_optimization(Z_final,Y)
    init = tf.global_variables_initializer() # set up variable initialization
    
    with tf.Session() as sess:
        sess.run(init) # initializes all the variables we've created
        for epoch in range(num_epochs):

            epoch_cost = 0.                       
            num_minibatches = int(training_examples / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                              feed_dict={X: minibatch_X, Y: minibatch_Y})                
                epoch_cost += minibatch_cost / num_minibatches
            
            print ("Cost after epoch %i: %.3f" % (epoch+1, epoch_cost), end = "") 
            correct_prediction = tf.equal(tf.argmax(Z_final), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("     Train Accuracy: %.3f" % (accuracy.eval({X: X_train, Y: Y_train})), end = "")
            print ("     Dev Accuracy: %.3f" % (accuracy.eval({X: X_dev, Y: Y_dev})))
        
        print ("Network has been trained")
        predict = tf.argmax(Z_final).eval({X: X_test})
        probs = tf.nn.softmax(Z_final).eval({X: X_test})
        
        return predict, probs
    
Y_predict, Y_probs = train_network(X_train, Y_train, X_dev, Y_dev, X_test, num_epochs = 20)

Y_predict = Y_predict.reshape(-1,1)
predictions_df = pd.DataFrame (Y_predict,columns = ['Label'])
predictions_df['ImageID'] = predictions_df.index + 1
submission_df = predictions_df[predictions_df.columns[::-1]]
submission_df.to_csv("submission.csv", index=False, header=True)
submission_df.head()




