import numpy as np
import pandas as pd
from pandas import set_option
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
print(os.listdir("../input"))
# Reading the dataset
data = pd.read_csv('../input/sonar.all-data.csv')
data.info()
data.head(10)
# Describe data
data.shape
set_option('precision', 3)
data.describe()
print("Total Columns : ", len(data.columns))
# Total Labels
data[data.columns[60]].value_counts()
# histograms
data.hist(sharex=False, sharey=False, xlabelsize=13, ylabelsize=13, figsize=(20,20))
plt.show()
# density
data.plot(kind='density', subplots=True, layout=(8,8), sharex=False, legend=True, fontsize=13, figsize=(40,40))
plt.show()
# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
fig.set_size_inches(10,10)
plt.show()
X = data[data.columns[0:60]].values
y = data[data.columns[60]].values
print(X.shape)
print(y.shape)
def one_hot_encode(labels):
    n_labels = len(labels)
    # np.unique - Find the unique elements of an array (pour éviter les doublons)
    n_unique_labels = len(np.unique(labels))
    # np.zeros([n,p]) - Représente la matrice nulle de taille n*p
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    # np.arange(a,b,i) - Construit le tableau générique [a; a+i; a+2i;...; b]
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
Y = one_hot_encode(y)
Y[0]
#We shuffle the data
X,Y = shuffle (X, Y, random_state = 0)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.20, random_state = 0)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
learning_rate = 0.2
training_epochs = 1000
#Annonce en sortie la dimension de la matrice X, soit 60 colonnes
n_dim = X.shape[1]
print("n_dim = ", n_dim)
n_class = 2
cost_history = np.empty(shape=[1],dtype=float)
n_hidden_1 = 60
x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])
# Define the model
def multilayer_perceptron(x, weights, biases):
 
    # Hidden layer with sigmoid activations
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
 
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer
# Define the weights for each layers
 
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_class]))
}
# Define the bias for each layers
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_class]))
}
init = tf.global_variables_initializer()
# Calling model
y = multilayer_perceptron(x, weights, biases)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
sess = tf.Session()
sess.run(init) 
mse_history = []
accuracy_history = []
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict = {x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict = {x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)
    
    if epoch % 50 == 0:
        print('epoch : ', epoch, ' ; ', 'cost: ', cost, " ; MSE: ", mse_, "- Train Accuracy: ", accuracy )

