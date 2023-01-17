# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import packages

import warnings

warnings.filterwarnings("ignore")

import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import math

from tensorflow.python.framework import ops

from sklearn.model_selection import train_test_split



CKPT_PATH = './cnn/cnn.ckpt'



def load_data():

    labeled_images = pd.read_csv('../input/train.csv')

    images = labeled_images.iloc[:, 1:]

    labels = labeled_images.iloc[:, :1]

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)



    # test_images[test_images>0] = 1

    # train_images[train_images>0] = 1



    test_images = test_images / 50

    train_images = train_images / 50



    train_images = shape_image(train_images)

    test_images = shape_image(test_images)



    train_images = np.array(train_images)

    test_images = np.array(test_images)



    train_labels = np.array(train_labels)

    test_labels = np.array(test_labels)



    train_labels = convert_to_one_hot(train_labels, 10).T

    test_labels = convert_to_one_hot(test_labels, 10).T



    X_train = train_images.astype("float")

    Y_train = train_labels.astype("float")

    X_test = test_images.astype("float")

    Y_test = test_labels.astype("float")



    return X_train, Y_train, X_test, Y_test





def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y





def shape_image(img):

    return np.array(img).reshape(img.shape[0], 28, 28, 1)





def create_placeholders(n_w, n_h, n_c, n_classes):

    X = tf.placeholder(dtype=tf.float32, shape=(None, n_w, n_h, n_c), name='X')

    Y = tf.placeholder(dtype=tf.float32, shape=(None, n_classes), name='Y')

    return X, Y





def initialize_parameters():

    tf.set_random_seed(1)                           

    initializer = tf.contrib.layers.xavier_initializer(seed=0)

    W1 = tf.get_variable("W1", [4,4,1,16], initializer=initializer)

    W2 = tf.get_variable("W2", [2,2,16,32], initializer=initializer)

    W3 = tf.get_variable("W3", [4,4,32,64], initializer=initializer)

    return W1, W2, W3





def forward_propagation(X, W1, W2, W3):

    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')

    A1 = tf.nn.relu(Z1)

    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')



    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')

    A2 = tf.nn.relu(Z2)

    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')



    Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding='VALID')

    A3 = tf.nn.relu(Z3)

    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')



    P3 = tf.contrib.layers.flatten(P3)

    Z4 = tf.contrib.layers.fully_connected(P3, 10, activation_fn=None)

    return Z4





def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):



    m = X.shape[0]               

    mini_batches = []

    np.random.seed(seed)

    

    # Step 1: Shuffle (X, Y)

    permutation = list(np.random.permutation(m))

    shuffled_X = X[permutation,:,:,:]

    shuffled_Y = Y[permutation,:]



    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning

    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]

        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    # Handling the end case (last mini-batch < mini_batch_size)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]

        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]

        mini_batch = (mini_batch_X, mini_batch_Y)

        mini_batches.append(mini_batch)

    

    return mini_batches





def compute_cost(Z3, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y)) 

    return cost

X_train, Y_train, X_test, Y_test = load_data()
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,

          num_epochs = 10, minibatch_size = 64, print_cost = True):

    """

    Implements a three-layer ConvNet in Tensorflow:

    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    """



    tf.set_random_seed(1)

    seed = 3 

    (m, n_H0, n_W0, n_C0) = X_train.shape             

    n_y = Y_train.shape[1]                            

    costs = [] 



    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    W1, W2, W3 = initialize_parameters()

    

    Z4 = forward_propagation(X, W1, W2, W3)

    cost = compute_cost(Z4, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    

    mincost = float('inf')

    ep = 0

    

    with tf.Session() as sess:

    

        sess.run(init)

        for epoch in range(num_epochs):

            

            ep += 1

            

            minibatch_cost = 0.

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)



            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches     

                

            if print_cost == True and epoch % 1 == 0: print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))

            if print_cost == True and epoch % 1 == 0: costs.append(minibatch_cost)

                    

            if minibatch_cost < mincost:

                ep = 0

                mincost = minibatch_cost

                save_path = saver.save(sess, CKPT_PATH)

                print("Model saved in file: %s" % save_path)

            

            if ep > 10:

                break

                

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        predict_op = tf.argmax(Z4, 1)

        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})

        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})



        print("Train Accuracy:", train_accuracy)

        print("Test Accuracy:", test_accuracy)

        
model(X_train, Y_train, X_test, Y_test, num_epochs = 5, minibatch_size = 64)
