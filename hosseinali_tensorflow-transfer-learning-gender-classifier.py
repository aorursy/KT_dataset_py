import os

import matplotlib.pyplot as pyp

import numpy
def dataloader( train_or_val ):    

    X_cnn = []

    X_flat = []

    Y = []

    if train_or_val == 'T':

        path = "../input/fliker-face-gender/aligned/"

    elif train_or_val == 'V':

        path = "../input/gender-face-validation/valid/"

    folder_list = sorted(os.listdir(path))

    for folder in folder_list:

        img_name_list = sorted(os.listdir(path+folder))

        for img_name in img_name_list:

            ax = pyp.imread(path+folder+'/'+img_name)

            X_cnn.append(ax)

            X_flat.append(ax.flatten())

            if folder[3] == 'F':

                Y.append([1])

            elif folder[3] == 'M':

                Y.append([0])

    X_cnn = numpy.array(X_cnn)

    X_flat = numpy.array(X_flat)

    Y = numpy.array(Y)    

    m = Y.shape[0]    

    permutation = list(numpy.random.permutation(m))

    X_cnn = X_cnn[permutation,:]

    X_flat = X_flat[permutation,:]

    Y = Y[permutation,:]    

    return X_flat, X_cnn, Y 
X_train_flattened, X_train_cnn, Y_train = dataloader('T')

X_val_flattened, X_val_cnn, Y_val = dataloader('V')
print("number of training examples = " + str(X_train_cnn.shape[0]))

print("number of test examples = " + str(X_val_cnn.shape[0]))

print('\nTrain datasets shape')

print(X_train_cnn.shape,'<-- Dataset format for fitting a CNN')

print(X_train_flattened.shape,'<-- Dataset format for fitting a fully connected')

print(Y_train.shape,'<-- Target variable')

print('\nValidation datasets shape:\n')

print(X_val_cnn.shape,'<-- Dataset format for fitting a CNN')

print(X_val_flattened.shape,'<-- Dataset format for fitting a fully connected')

print(Y_val.shape,'<-- Target variable')
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
def initialize_transfer_parameters(Transfer_parameters):

    tf.set_random_seed(1)  

    W1 = tf.get_variable("W1", [3, 3, 3, 64], initializer = Transfer_parameters[W1] )

    W2 = tf.get_variable("W2", [3, 3, 64, 64], initializer = Transfer_parameters[W1])

    W3 = tf.get_variable("W3", [3, 3, 64, 128], initializer = Transfer_parameters[W1])

    W4 = tf.get_variable("W1", [3, 3, 128, 128], initializer = Transfer_parameters[W1] )

    W5 = tf.get_variable("W2", [3, 3, 128, 256], initializer = Transfer_parameters[W1])

    W5 = tf.get_variable("W2", [3, 3, 256, 256], initializer = Transfer_parameters[W1])

    W6 = tf.get_variable("W3", [3, 3, 256, 512], initializer = Transfer_parameters[W1])

    W7 = tf.get_variable("W1", [3, 3, 512, 512], initializer = Transfer_parameters[W1] )

    W8 = tf.get_variable("W2", [3, 3, 512, 512], initializer = Transfer_parameters[W1])

    W9 = tf.get_variable("W3", [3, 3, 512, 512], initializer = Transfer_parameters[W1])

    W10 = tf.get_variable("W1", [3, 3, 512, 512], initializer = Transfer_parameters[W1] )

    W11 = tf.get_variable("W2", [3, 3, 512, 512], initializer = Transfer_parameters[W1])

    W12 = tf.get_variable("W3", [3, 3, 512, 512], initializer = Transfer_parameters[W1])

    W13 = tf.get_variable("W1", [3, 3, 512, 512], initializer = Transfer_parameters[W1] )

    W14 = tf.get_variable("W2", [7, 7, 512, 4096], initializer = Transfer_parameters[W1])

    W15 = tf.get_variable("W3", [1, 1, 4096, 4096], initializer = Transfer_parameters[W1])

    W16 = tf.get_variable("W1", [1, 1, 4096, 2622], initializer = Transfer_parameters[W1] )    

    Transfer_parameters_obj = {"W1":W1,"W2":W2,"W3": W3,"W4":W4,"W5":W5,"W6":W6,"W7":W7,"W8":W8,

                       "W9":W9,"W10":W10,"W11":W11,"W12":W12,"W13":W13,"W14":W14,"W15":W15,"W16":W16}

    return Transfer_parameters_obj

def Transfer_forward_propagation(X, Transfer_parameters_obj):

    

    W1 = Transfer_parameters['W1']

    W2 = Transfer_parameters['W2']

    W3 = Transfer_parameters['W3']

    W4 = Transfer_parameters['W4']

    W5 = Transfer_parameters['W5']

    W6 = Transfer_parameters['W6']

    W7 = Transfer_parameters['W7']

    W8 = Transfer_parameters['W8']

    W9 = Transfer_parameters['W9']

    W10 = Transfer_parameters['W10']

    W11 = Transfer_parameters['W11']

    W12 = Transfer_parameters['W12']

    W13 = Transfer_parameters['W13']

    W14 = Transfer_parameters['W14']

    W15 = Transfer_parameters['W15']

    W16 = Transfer_parameters['W16']

    

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

    Z2 = tf.nn.relu(Z1)

    Z3 = tf.nn.conv2d(Z2, W2, strides=[1, 1, 1, 1], padding='SAME')

    Z4 = tf.nn.relu(Z3)

    Z5 = tf.nn.max_pool(Z4, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding='SAME')

    Z6 = tf.nn.conv2d(Z5, W3, strides=[1, 1, 1, 1], padding='SAME')

    Z7 = tf.nn.relu(Z6)

    Z8 = tf.nn.conv2d(Z7, W4, strides=[1, 1, 1, 1], padding='SAME')

    Z9 = tf.nn.relu(Z8)

    Z10 = tf.nn.max_pool(Z9, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')

    Z11 = tf.nn.conv2d(Z10, W5, strides=[1, 1, 1, 1], padding='SAME')

    Z12 = tf.nn.relu(Z11)

    Z13 = tf.nn.conv2d(Z12, W6, strides=[1, 1, 1, 1], padding='SAME')

    Z14 = tf.nn.relu(Z13)

    Z15 = tf.nn.conv2d(Z14, W7, strides=[1, 1, 1, 1], padding='SAME')

    Z16 = tf.nn.relu(Z15)

    Z17 = tf.nn.max_pool(Z16, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')

    Z18 = tf.nn.conv2d(Z17, W8, strides=[1, 1, 1, 1], padding='SAME')

    Z19 = tf.nn.relu(Z18)

    Z20 = tf.nn.conv2d(Z19, W9, strides=[1, 1, 1, 1], padding='SAME')

    Z21 = tf.nn.relu(Z20)

    Z22 = tf.nn.conv2d(Z21, W10, strides=[1, 1, 1, 1], padding='SAME')

    Z23 = tf.nn.relu(Z22)

    Z24 = tf.nn.max_pool(Z23, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')

    Z25 = tf.nn.conv2d(Z24, W11, strides=[1, 1, 1, 1], padding='SAME')

    Z26 = tf.nn.relu(Z25)

    Z27 = tf.nn.conv2d(Z26, W12, strides=[1, 1, 1, 1], padding='SAME')

    Z28 = tf.nn.relu(Z27)

    Z29 = tf.nn.conv2d(Z28, W13, strides=[1, 1, 1, 1], padding='SAME')

    Z30 = tf.nn.relu(Z29)

    Z31 = tf.nn.max_pool(Z30, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding='SAME')

    Z32 = tf.nn.conv2d(Z31, W14, strides=[1, 1, 1, 1], padding='SAME')

    Z33 = tf.nn.relu(Z32)

    Z34 = tf.nn.conv2d(Z33, W15, strides=[1, 1, 1, 1], padding='SAME')

    Z35 = tf.nn.relu(Z34)

    Z36 = tf.nn.conv2d(Z35, W16, strides=[1, 1, 1, 1], padding='SAME')

    Z37 = tf.contrib.layers.flatten(Z36)    

    return Z37
def initialize_parameters():

    tf.set_random_seed(1) 

    W1 = tf.get_variable("W1", [49152,25], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    W2 = tf.get_variable("W2", [25,12], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    W3 = tf.get_variable("W3", [12,1], initializer = tf.contrib.layers.xavier_initializer(seed=1))

    parameters = { "W1": W1 , "W2": W2 , "W3": W3 }

    return parameters
def create_placeholders(n_x,n_y):

    X = tf.placeholder(tf.float32, [None,n_x], name="X")

    Y = tf.placeholder(tf.float32, [None,n_y], name="Y")    

    return X, Y
def forward_propagation(X, parameters):

    W1 = parameters['W1']

    W2 = parameters['W2']

    W3 = parameters['W3'] 

    

    Z1 = tf.matmul(X, W1)

    A1 = tf.nn.relu(Z1)

    Z2 = tf.matmul(A1, W2)

    A2 = tf.nn.relu(Z2)

    Z3 = tf.matmul(A2, W3)    

    return Z3
def compute_cost(logits,labels):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))    

    return cost
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,

          num_epochs = 2, minibatch_size = 32, print_cost = True):

    

    tf.reset_default_graph()                         

    tf.set_random_seed(1)                             

    seed = 3                                          

    (m,n_x) = X_train.shape                          

    n_y = Y_train.shape[1]                            

    costs = []                                  

    

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)    

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.

            num_minibatches = m//minibatch_size

            seed = seed + 1

            minibatches = numpy.array_split(X_train,num_minibatches,axis=0)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch

                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

                

        plt.plot(np.squeeze(costs))

        plt.ylabel('cost')

        plt.xlabel('iterations (per tens)')

        plt.title("Learning rate =" + str(learning_rate))

        plt.show()



        parameters = sess.run(parameters)

        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))



        # print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))

        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        

        return parameters
parameters = model(X_train, Y_train, X_val, Y_val)