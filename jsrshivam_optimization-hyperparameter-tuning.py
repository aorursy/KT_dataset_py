import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors


data = pd.read_csv('../input/data.csv')



print(data.head())

print(data['class'].unique())


X = data.loc[:, data.columns != 'class'].values

y = data['class'].values

###End code



assert X.shape == (10000, 10)

assert y.shape == (10000, )
colors=['green','blue']

cmap = matplotlib.colors.ListedColormap(colors)

#Plot the figure

plt.figure()

plt.title('Non-linearly separable classes')

plt.scatter(X[:,0], X[:,3], c=y,

           marker= 'o', s=50,cmap=cmap,alpha = 0.5 )

plt.show()
from pandas.plotting import scatter_matrix

%matplotlib inline

color_wheel = {0: "#0392cf", 

               1: "#7bc043", 

            }



colors_mapped = data["class"].map(lambda x: color_wheel.get(x))



axes_matrix = scatter_matrix(data.loc[:, data.columns != 'class'], alpha = 0.2, figsize = (10, 10), color=colors_mapped )
X_data = X.T

y_data = y.reshape(1,len(y))



assert X_data.shape == (10, 10000)

assert y_data.shape == (1, 10000)
layer_dims = [10,9,9,1]
import tensorflow as tf
def placeholders(num_features):

    A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]))

    Y = tf.placeholder(dtype = tf.float64, shape = ([1,None]))

    return A_0,Y
def initialize_parameters_deep(layer_dims):

    tf.set_random_seed(1)

    L = len(layer_dims)

    parameters = {}

    for l in range(1,L):

        parameters['W' + str(l)] = tf.get_variable("W" + str(l), shape=[layer_dims[l], layer_dims[l-1]], dtype = tf.float64,

                                   initializer=tf.random_normal_initializer())

                                   

        parameters['b' + str(l)] = tf.get_variable("b"+ str(l), shape = [layer_dims[l], 1], dtype= tf.float64, initializer= tf.zeros_initializer() )

        

    return parameters 
def linear_forward_prop(A_prev,W,b, activation):

    Z = tf.add(tf.matmul(W, A_prev), b)

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

        A = linear_forward_prop(A_prev,parameters['W' + str(l)],parameters['b' + str(l)], "relu")     

        #call linear forward prop with relu activation

    A = linear_forward_prop(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid" )                  

    #call linear forward prop with sigmoid activation

    

    return A
def final_cost(Z_final, Y , parameters, regularization = False, lambd = 0):

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_final,labels=Y)

    if regularization:

        reg_term = 0

        L = len(parameters)//2

        for l in range(1,L+1):

            

            reg_term +=  tf.nn.l2_loss(parameters['W'+str(l)])              #add L2 loss term

            

        cost = cost + (lambd/2) * reg_term

    return tf.reduce_mean(cost)
import numpy as np

def random_samples_minibatch(X, Y, batch_size, seed = 1):

    np.random.seed(seed)

    

    m =  X.shape[1]                                          #number of samples

    num_batches = int(m / batch_size )                               #number of batches derived from batch_size

    

    indices =  np.random.permutation(m)                                 # generate ramdom indicies

    shuffle_X = X[:,indices]

    shuffle_Y = Y[:,indices]

    mini_batches = []

    

    #generate minibatch

    for i in range(num_batches):

        X_batch = shuffle_X[:,i * batch_size:(i+1) * batch_size]

        Y_batch = shuffle_Y[:,i * batch_size:(i+1) * batch_size]

        

        assert X_batch.shape == (X.shape[0], batch_size)

        assert Y_batch.shape == (Y.shape[0], batch_size)

        

        mini_batches.append((X_batch, Y_batch))

    

    #generate batch with remaining number of samples

    if m % batch_size != 0:

        X_batch = shuffle_X[:, (num_batches * batch_size):]

        Y_batch = shuffle_Y[:, (num_batches * batch_size):]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
def model(X_train,Y_train, layer_dims, learning_rate, optimizer ,num_iter, mini_batch_size):

    tf.reset_default_graph()

    num_features, num_samples = X_train.shape

    

    A_0, Y = placeholders(num_features)

    #call placeholder function to initialize placeholders A_0 and Y

    parameters =  initialize_parameters_deep(layer_dims)                   

    #Initialse Weights and bias using initialize_parameters

    Z_final = l_layer_forwardProp(A_0, parameters)                      

    #call the function l_layer_forwardProp() to define the final output

    

    cost =  final_cost(Z_final, Y , parameters, regularization = True)

    #call the final_cost function with regularization set TRUE

    

    

    

    if optimizer == "momentum":

        train_net = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost)                 

        #call tensorflow's momentum optimizer with momentum = 0.9

    elif optimizer == "rmsProp":

        train_net = tf.train.RMSPropOptimizer(learning_rate, decay=0.999).minimize(cost)

                   

        #call tensorflow's RMS optimiser with decay = 0.999

    elif optimizer == "adam":

        train_net = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999).minimize(cost)                 

        ##call tensorflow's adam optimizer with beta1 = 0.9, beta2 = 0.999

    

    seed = 1

    num_minibatches = int(num_samples / mini_batch_size)

    init = tf.global_variables_initializer()

    costs = []

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_iter):

            epoch_cost = 0

            

            mini_batches = random_samples_minibatch(X_train, Y_train, mini_batch_size, seed)

            #call random_sample_minibatch to return minibatches

            

            seed = seed + 1

            

            #perform gradient descent for each mini-batch

            for mini_batch in mini_batches:

                

                X_batch, Y_batch = mini_batch            #assign minibatch

                

                _,mini_batch_cost = sess.run([train_net, cost], feed_dict={A_0: X_batch, Y: Y_batch})

                epoch_cost += mini_batch_cost/num_minibatches

            

            if epoch % 2 == 0:

                costs.append(epoch_cost)

            if epoch % 10 == 0:

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
params_momentum = model(X_data,y_data, layer_dims, learning_rate=0.001, optimizer='momentum' ,num_iter=100, mini_batch_size=256)
params_momentum = model(X_data,y_data, layer_dims, learning_rate=0.001, optimizer='rmsProp' ,num_iter=100, mini_batch_size=256)
params_momentum = model(X_data,y_data, layer_dims, learning_rate=0.001, optimizer='adam' ,num_iter=100, mini_batch_size=256)