import tensorflow as tf

import numpy as np

import time

from sklearn.preprocessing import LabelEncoder

import pandas as pd

from sklearn.cross_validation import train_test_split
def get_data(data_location, split_dataset):

    dataset = pd.read_csv(data_location)



    # 0 shape to get total of rows, 1 to get total of columns

    rows = dataset.shape[0]

    print ("there are ", rows, " rows before cleaning\n")



    # removing unimportant columns

    columns = ['#', 'Name', 'Type 2', 'Generation', 'Legendary', 'Total']

    for text in columns:

        del dataset[text]



    # get all data except first column

    x = dataset.ix[: , 1:].values



    # get all data on first column only

    y = dataset.ix[: , :1].values



    # split our dataset to reduce overfitting

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_dataset)

    

    return x_train, x_test, y_train, y_test
def return_embedded(x):



    data = np.zeros((x.shape[0], np.unique(x).shape[0]), dtype = np.float32)

    

    for i in range(x.shape[0]):

        data[i][x[i][0]] = 1.0

    

    return data
data_location = '../input/Pokemon.csv'



# not included input and output layer

# atleast 1

num_layers = 1

size_layer = 128



learning_rate = 0.5



split_dataset = 0.5



biased_node = True



batch_size = 50



epoch = 1



# got sigmoid, tanh, relu

activation = 'sigmoid'
x_train, x_test, y_train, y_test = get_data(data_location, split_dataset)



label = sorted(list(set(y_train[:, 0])))



x_train = x_train.astype(float)

x_test = x_test.astype(float)



y_train = np.array([LabelEncoder().fit_transform(y_train)]).T

y_test = np.array([LabelEncoder().fit_transform(y_test)]).T



y_train_ = return_embedded(y_train)



print ("Train for ", epoch, " iteration")

print ("There are ", x_train.shape[0], " of rows for training")
class Model:

    

    def __init__(self, activation, biased_node, learning_rate, num_layers, size_layer, size_x, size_y):

        

        if activation == 'sigmoid':

            self.activation = tf.nn.sigmoid

        elif activation == 'tanh':

            self.activation = tf.nn.tanh

        elif activation == 'relu':

            self.activation = tf.nn.relu

        else:

            raise Exception("model type not supported")

        

        self.X = tf.placeholder("float", [None, size_x])

        self.Y = tf.placeholder("float", [None, size_y])

        

        if biased_node:

            self.biased = tf.Variable(tf.random_normal([size_layer * num_layers], mean = 0.0, stddev = 0.1))

        

        self.inner_layer = tf.Variable(tf.random_normal([size_x, size_layer * num_layers], mean = 0.0, stddev = 0.1))

        

        output_layer = tf.Variable(tf.random_normal([size_layer * num_layers, size_y], mean = 0.0, stddev = 0.1))

        

        if biased_node:

            batched_layer = self.activation(tf.add(tf.matmul(self.X, self.inner_layer), self.biased))

            

        else:

            batched_layer = self.activation(tf.matmul(self.X, self.inner_layer))

        

        self.W = tf.matmul(batched_layer, output_layer)

        

        self.b = tf.Variable(tf.random_normal([size_y]))

                                 

        self.y = self.W + self.b 

        

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y, labels = self.Y))

        

        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        

        self.final_outputs = self.y

        
# start the session graph

sess = tf.InteractiveSession()



model = Model(activation, biased_node, learning_rate, num_layers, size_layer, x_train.shape[1], y_train_.shape[1])



# initialize global variables

sess.run(tf.global_variables_initializer())



for i in range(epoch):

    last_time = time.time()

    total_lost = 0

    total_accuracy = 0

    for n in range(0, x_train.shape[0], batch_size):

        output, _, loss = sess.run([model.final_outputs, model.optimizer, model.cost], 

                                   feed_dict = {model.X: x_train[n : n + batch_size, :], 

                                    model.Y: y_train_[n : n + batch_size, :]})

        out = output[0].argmax()

        if out == y_train[n, :][0]:

            total_accuracy += 1

        

        total_lost += loss

        

    diff = time.time() - last_time

    print ("total accuracy: ", total_accuracy / (x_train.shape[0] * 1.0))

    print ("batch: ", i + 1, ", loss: ", total_lost / x_train.shape[0], ", speed: ", diff, " s / epoch")

    total_lost = 0

    total_accuracy = 0
print ("\nDone training, Benchmarking ===========================================\nThere are ", x_test.shape[0],  " of rows for testing")



# 0 = bug .. 18 = water

totalelement = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



# 0 = bug .. 18 = water

elementfound = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]



total_correct = 0



for n in range(x_test.shape[0]):

    

    output = sess.run([model.final_outputs], feed_dict={model.X: x_test[n : n + 1, :]})

    

    # to get index value

    i,j = np.unravel_index(output[0].argmax(), output[0].shape)

    

    if np.where(output[0] == output[0][i][j])[1][0] == y_test[n, :][0]:

        total_correct += 1

        elementfound[y_test[n, :][0]] += 1

    

    totalelement[y_test[n, :][0]] += 1

    

    

accuracy = (total_correct / (x_test.shape[0] * 1.0))



print ("total correct: ", total_correct, " over ", x_test.shape[0], " test sets")

print ("overall accuracy: ", accuracy)



for i in range(len(totalelement)):

    print (label[i], " accuracy: ", elementfound[i] / (totalelement[i] * 1.0))