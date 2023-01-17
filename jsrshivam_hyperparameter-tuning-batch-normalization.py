import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors
data = pd.read_csv('../input/data (1).csv')

data.head()


X = data.loc[:, data.columns != 'target'].values

y = data['target'].values

colors=['green','blue']

cmap = matplotlib.colors.ListedColormap(colors)

#Plot the figure

plt.figure()

plt.title('Non-linearly separable classes')

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap=cmap,

            s=25, edgecolor='k')

plt.show()
from pandas.plotting import scatter_matrix

%matplotlib inline

color_wheel = {0: "#0392cf", 

               1: "#7bc043", 

            }



colors_mapped = data["target"].map(lambda x: color_wheel.get(x))



axes_matrix = scatter_matrix(data.loc[:, data.columns != 'target'], alpha = 0.2, figsize = (10, 10), color=colors_mapped )
X_data = X.T

y_data = y.reshape(1, -1)



assert X_data.shape == (2, 1000)

assert y_data.shape == (1, 1000)


layer_dims = [2, 20, 20, 20, 20, 1]

import tensorflow as tf
def placeholders(num_features):

    

    A_0 = tf.placeholder( shape=([num_features, None]), dtype=tf.float32)

    Y = tf.placeholder(shape=([1,None]), dtype=tf.float32)

    

    return A_0,Y
def initialize_parameters_deep(layer_dims):

    tf.set_random_seed(1)

    L = len(layer_dims)

    parameters = {}

    for l in range(1,L):

        

        parameters['W' + str(l)] = tf.get_variable('W'+ str(l), shape=([layer_dims[l], layer_dims[l-1]]), dtype=tf.float32,

                                                  initializer=tf.contrib.layers.xavier_initializer())

        parameters['b' + str(l)] = tf.get_variable('b'+ str(l), shape=([layer_dims[l], 1]), dtype=tf.float32, initializer=tf.zeros_initializer())

        

    return parameters 
def linear_forward_prop(A_prev,W,b, activation):

    

    Z = tf.add(tf.matmul( W, A_prev), b)                          

    #call linear_fowrward prop

    Z = tf.layers.batch_normalization(inputs=Z, axis=0, training=True, gamma_initializer=tf.ones_initializer(), 

                                      beta_initializer=tf.zeros_initializer())                             

    #implement batch normalization on Z

    

    if activation == "sigmoid":

        A = Z

    elif activation == "relu":

        A = tf.nn.relu(Z)

    return A
def l_layer_forwardProp(A_0, parameters):

    A = A_0

    L = len(parameters)//2

    for l in range(1,L):

        A_prev = A

    

        A = linear_forward_prop(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu' )                 

        #call linear forward prop with relu activation

    A = linear_forward_prop(A, parameters['W' +str(L)], parameters['b' + str(L)], activation='sigmoid')                      

    #call linear forward prop with sigmoid activation

    

    return A
def final_cost(Z_final, Y , parameters, regularization = False, lambd = 0):

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final,labels=Y)

    if regularization:

        reg_term = 0

        L = len(parameters)//2

        for l in range(1,L+1):

            

            reg_term += tf.nn.l2_loss(parameters['W'+str(l)])             #add L2 loss term

            

        cost = cost + (lambd/2) * reg_term

    return tf.reduce_mean(cost)
import numpy as np

def random_samples_minibatch(X, Y, batch_size, seed = 1):

    np.random.seed(seed)

    

    m = X.shape[1]                                         #number of samples

    num_batches =int( m / batch_size)                                 #number of batches derived from batch_size

    

    indices = np.random.permutation(m)                                  # generate ramdom indicies

    shuffle_X = X[:,indices]

    shuffle_Y = Y[:,indices]

    mini_batches = []

    

    #generate minibatch

    for i in range(num_batches):

        X_batch = shuffle_X[ :, i * batch_size:(i+1) * batch_size]

        Y_batch = shuffle_Y[ :, i * batch_size:(i+1) * batch_size]

        

        assert X_batch.shape == (X.shape[0], batch_size)

        assert Y_batch.shape == (Y.shape[0], batch_size)

        

        mini_batches.append((X_batch, Y_batch))

    

    #generate batch with remaining number of samples

    if m % batch_size != 0:

        X_batch = shuffle_X[ :, (num_batches * batch_size): ]

        Y_batch = shuffle_Y[:, (num_batches * batch_size): ]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
def model_with_minibatch(X_train,Y_train, layer_dims, learning_rate,num_iter, mini_batch_size):

    tf.reset_default_graph()

    num_features, num_samples = X_train.shape

    

    A_0, Y =   placeholders(num_features)

    #call placeholder function to initialize placeholders A_0 and Y

    parameters = initialize_parameters_deep(layer_dims)

    #Initialse Weights and bias using initialize_parameters

    Z_final =  l_layer_forwardProp(A_0, parameters) 

    #call the function l_layer_forwardProp() to define the final output

    

    cost = final_cost(Z_final, Y , parameters, regularization = True)

    #call the final_cost function with regularization set TRUE

    

    

    #use adam optimization to train the network

    train_net = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(cost)

    

    seed = 1

    num_minibatches = int(num_samples / mini_batch_size)

    init = tf.global_variables_initializer()

    costs = []

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_iter):

            epoch_cost = 0

            

            mini_batches =  random_samples_minibatch(X_train, Y_train, mini_batch_size, seed)

            #call random_sample_minibatch to return minibatches

            

            seed = seed + 1

            

            #perform gradient descent for each mini-batch

            for mini_batch in mini_batches:

                

                X_batch, Y_batch = mini_batch 

                #assign minibatch

                

                _,mini_batch_cost = sess.run([train_net, cost], feed_dict={A_0: X_batch, Y: Y_batch})

                

                epoch_cost += mini_batch_cost/num_minibatches

            if epoch % 2 == 0:

                costs.append(epoch_cost)

            if epoch % 100 == 0:

                print(epoch_cost)

        with open("output.txt", "w+") as file:

            file.write("%f" % epoch_cost)

        plt.ylim(0 ,2, 0.0001)

        plt.xlabel("epoches per 2")

        plt.ylabel("cost")

        plt.plot(costs)

        plt.show()

        params = sess.run(parameters)

    return params


parameters =  model_with_minibatch(X_data,y_data, layer_dims, learning_rate=0.001,num_iter=1000, mini_batch_size=256)