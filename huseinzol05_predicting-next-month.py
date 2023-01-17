import tensorflow as tf

import numpy as np

import time

import pandas as pd

from sklearn.cross_validation import train_test_split
def get_data(data_location, split_dataset):

    dataset = pd.read_csv(data_location)



    # 0 shape to get total of rows, 1 to get total of columns

    rows = dataset.shape[0]

    print ("there are ", rows, " rows before cleaning\n")



    # removing unimportant columns

    columns = ['ID']

    for text in columns:

        del dataset[text]



    # get all data except last column

    x = dataset.ix[: , :-1].values



    # get all data on last column only

    y = dataset.ix[: , -1:].values



    # split our dataset to reduce overfitting

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_dataset)

    

    return x_train, x_test, y_train, y_test
def return_embedded(x):



    data = np.zeros((x.shape[0], np.unique(x).shape[0]), dtype = np.float32)

    

    for i in range(x.shape[0]):

        data[i][x[i][0]] = 1.0

    

    return data
data_location = '../input/UCI_Credit_Card.csv'



# not included input and output layer

num_layers = 5

# all hidden layers are same wide size

size_layer = 64

learning_rate = 0.01

# batch mini size for training

batch_size = 100



# beta for regularizer, learn from penalty value

beta = 0.05



# probability to disconnect connection between nodes

prob_dropout = 1.0



biased_node = True



split_dataset = 0.7



# iteration for training

epoch = 100
# got sigmoid, softmax, tanh

activation = 'relu'



if activation == 'sigmoid':

    activation = tf.nn.sigmoid

elif activation == 'tanh':

    activation = tf.nn.tanh

elif activation == 'relu':

    activation = tf.nn.relu

else:

    raise Exception("model type not supported")

    

x_train, x_test, y_train, y_test = get_data(data_location, split_dataset)



y_train = return_embedded(y_train)

y_test = return_embedded(y_test)
# Neural Network pipelining ===========================================================================



X = tf.placeholder("float", [None, x_train.shape[1]])

Y = tf.placeholder("float", [None, y_train.shape[1]])

        

input_layer = tf.Variable(tf.random_normal([x_train.shape[1], size_layer]))



if biased_node:

    biased_input_layer = tf.Variable(tf.random_normal([size_layer]))

    biased = []

    for i in range(num_layers):

        biased.append(tf.Variable(tf.random_normal([size_layer])))



layers = []

for i in range(num_layers):

    layers.append(tf.Variable(tf.random_normal([size_layer, size_layer])))



output_layer = tf.Variable(tf.random_normal([size_layer, y_train.shape[1]]))



if biased_node:

    first_l = activation(tf.add(tf.matmul(X, input_layer), biased_input_layer))

    

    # reduce nodes connection

    first_l = tf.nn.dropout(first_l, prob_dropout)

    

    next_l = activation(tf.add(tf.matmul(first_l, layers[0]), biased[0]))

    # reduce nodes connection

    next_l = tf.nn.dropout(next_l, prob_dropout)

    

    for i in range(1, num_layers - 1):

        next_l = activation(tf.add(tf.matmul(next_l, layers[i]), biased[i]))

        

        # reduce nodes connection

        next_l = tf.nn.dropout(next_l, prob_dropout)

else:

    first_l = activation(tf.matmul(X, input_layer))

    

    # reduce nodes connection

    first_l = tf.nn.dropout(first_l, prob_dropout)

    

    next_l = activation(tf.matmul(first_l, layers[0]))

    

    # reduce nodes connection

    next_l = tf.nn.dropout(next_l, prob_dropout)

    

    for i in range(1, num_layers - 1):

        next_l = activation(tf.matmul(next_l, layers[i]))

        

        # reduce nodes connection

        next_l = tf.nn.dropout(next_l, prob_dropout)

    

last_l = tf.matmul(next_l, output_layer)



# adding up all penalties values

regularizers = tf.nn.l2_loss(input_layer) + sum(map(lambda x: tf.nn.l2_loss(x), layers)) + tf.nn.l2_loss(output_layer)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = last_l, labels = Y))



# included penalty values

cost = tf.reduce_mean(cost + beta * regularizers)



optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



correct_prediction = tf.equal(tf.argmax(last_l, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# start the session graph

sess = tf.InteractiveSession()

    

# initialize global variables

sess.run(tf.global_variables_initializer())



print ("Train for ", epoch, " iteration")

print ("There are ", x_train.shape[0], " of rows for training")

for i in range(epoch):

    last_time = time.time()

    total_lost = 0

    total_accuracy = 0

    

    for n in range(0, x_train.shape[0], batch_size):

        out, _, loss = sess.run([accuracy, optimizer, cost], feed_dict={X: x_train[n : n + batch_size, :], Y: y_train[n : n + batch_size, :]})

        total_accuracy += out

        total_lost += loss

    

    print ("total accuracy: ", total_accuracy / (x_train.shape[0] / batch_size * 1.0))

    diff = time.time() - last_time

    print ("batch: ", i + 1, ", loss: ", total_lost/x_train.shape[0], ", speed: ", diff, " s / epoch")

    total_lost = 0

    total_accuracy = 0
total_correct = 0

total_positive = 0

total_correct_positive = 0

for n in range(x_test.shape[0]):

    

    correct = sess.run(accuracy, feed_dict={X: x_test[n : n + 1, :], Y: y_test[n : n + 1 , :]})

    total_correct += correct

    if y_test[n][1] == 1:

        total_positive += 1

        if correct == 1:

            total_correct_positive += 1

    

print ("total correct positive: ", total_correct_positive, " / ", total_positive)

print ("total correct: ", int(total_correct), " / ", x_test.shape[0]) 

print ("total accuracy: ", total_correct / (x_test.shape[0] * 1.0))