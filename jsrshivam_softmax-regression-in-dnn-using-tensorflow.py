import pandas as pd

data = pd.read_csv('../input/data.csv')

print(data.head())


X =  data[['feature1', 'feature2']].values                                        #Extract feature1 and feature2 values

Y =  data['class'].values                                        #Extract target values



print("target labels: ", list(set(Y)))



assert X.shape == (2000, 2)

assert Y.shape == (2000, )
import matplotlib.pyplot as plt

import matplotlib.colors

%matplotlib inline

colors=['blue','green','red', 'yellow']

cmap = matplotlib.colors.ListedColormap(colors)

#Plot the figure

plt.figure()

plt.title('Non-linearly separable classes')

plt.scatter(X[:,0], X[:,1], c=Y,

           marker= 'o', s=50,cmap=cmap,alpha = 0.5 )

plt.show()
import tensorflow as tf

import numpy as np

#import tensorflow
labels = list(set(Y))

depth =  len(np.unique(Y))                  #number of unique labels



print("labels: ", labels)

with tf.Session() as sess:

    YtrainOneHot = tf.one_hot(Y, depth, axis = 0) # columnwise: axis = 0, axis=-1 row-wise

    Y_onehot = sess.run(YtrainOneHot)

assert Y_onehot.shape == (4, 2000)



print("\nfirst five rows of target Y:\n", Y[:5])

print("\nfirst five rows of target Y_onehot:\n", Y_onehot[:,:5])

print("X dimension:{} ,Y_onehot dimension:{}".format(X.shape, Y_onehot.shape))
X_data = X.T

Y_data = Y_onehot

print(Y_data.shape, Y.shape)


layer_dims = [2, 25, 25, 4]
def placeholders(num_features, num_classes):

    A_0 = tf.placeholder(dtype = tf.float64, shape = ([num_features,None]))

    Y = tf.placeholder(dtype = tf.float64, shape = ([num_classes,None]))

    return A_0,Y
def initialize_parameters_deep(layer_dims):

    tf.set_random_seed(1)

    L = len(layer_dims)

    parameters = {}

    for l in range(1,L):

        parameters['W' + str(l)] = tf.get_variable("W" + str(l), shape=[layer_dims[l], layer_dims[l-1]], dtype = tf.float64,

                                   initializer=tf.contrib.layers.xavier_initializer())

        parameters['b' + str(l)] = tf.get_variable("b"+ str(l), shape = [layer_dims[l], 1], dtype= tf.float64, initializer= tf.zeros_initializer() )

    return parameters 
def linear_forward_prop(A_prev,W,b, activation):

    Z =   tf.add(tf.matmul(W, A_prev), b)    

    if activation == "softmax":

        A = Z

    elif activation == "relu":

        A =  tf.nn.relu(Z) 

    return A
def l_layer_forwardProp(A_0, parameters):

    A = A_0

    L = len(parameters)//2

    for l in range(1,L):

        A_prev = A

        A = linear_forward_prop(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")

    A = linear_forward_prop(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax" )

    return A
def final_cost(Z_final, Y ):

    logits = tf.transpose(Z_final)

    labels = tf.transpose(Y)

    ###Start code

    cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)                  #use tensorflow's softmax_cross_entropy to calculate cost

    ###End code

    return tf.reduce_mean(cost)
import numpy as np

def random_samples_minibatch(X, Y, batch_size, seed = 1):

    np.random.seed(seed)

    m = X.shape[1]                                 #number of samples

    num_batches = int(m/batch_size)                #number of batches derived from batch_size

    indices = np.random.permutation(m)             # generate ramdom indicies

    shuffle_X = X[:,indices]

    shuffle_Y = Y[:,indices]

    mini_batches = []

    

    #generate minibatch

    for i in range(num_batches):

        ##Start code here

        X_batch = shuffle_X[:,i * batch_size:(i+1) * batch_size]

        Y_batch = shuffle_Y[:,i * batch_size:(i+1) * batch_size]

        ##End code

        

        assert X_batch.shape == (X.shape[0], batch_size)

        assert Y_batch.shape == (Y.shape[0], batch_size)

        

        mini_batches.append((X_batch, Y_batch))

    

    #generate batch with remaining number of samples

    if m % batch_size != 0:

        ##Srart code here

        X_batch = shuffle_X[:, (num_batches * batch_size):]

        Y_batch = shuffle_Y[:, (num_batches * batch_size):]

        ##Srart code here

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
def model_with_minibatch(X_train,Y_train, layer_dims, learning_rate, num_iter, mini_batch_size):

    

    tf.reset_default_graph()

    num_features, num_samples = X_train.shape

    num_classes = Y_train.shape[0]

    A_0, Y = placeholders(num_features, num_classes)

    parameters = initialize_parameters_deep(layer_dims)

    Z_final = l_layer_forwardProp(A_0, parameters)

    cost = final_cost(Z_final, Y)

    train_net = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    seed = 1

    num_minibatches = int(num_samples / mini_batch_size)

    init = tf.global_variables_initializer()

    costs = []

    

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_iter):

            epoch_cost = 0

            mini_batches = random_samples_minibatch(X_train, Y_train, mini_batch_size, seed) #generate array of minibatch using random_samples_minibatch()

            seed = seed + 1

            

            #perform gradient descent for each mini-batch

            for mini_batch in mini_batches: 

                ##Start code

                X_batch, Y_batch = mini_batch

                _,mini_batch_cost = sess.run([train_net, cost], feed_dict={A_0: X_batch, Y: Y_batch})

                ##End code

                

                epoch_cost += mini_batch_cost/num_minibatches

            if epoch % 1 == 0:

                costs.append(epoch_cost)

            if epoch % 10 == 0:

                print(epoch_cost)

        with open('output.txt', 'w') as file:

            file.write("cost = %f" % costs[-1])

        plt.ylim(0, max(costs), 0.0001)

        plt.xlabel("epoches per 100")

        plt.ylabel("cost")

        plt.plot(costs)

        plt.show()

        params = sess.run(parameters)

    return params
params = model_with_minibatch(X.T, Y_onehot, layer_dims, 0.01, 100, 32)
def predict(X_train, params):

    with tf.Session() as sess:

        Y = tf.arg_max(l_layer_forwardProp(X_train,params), dimension=0)

        return sess.run(Y)
def plot_decision_boundary1( X, y, model):

    plt.clf()

    # Set min and max values and give it some padding

    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1

    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   

    colors=['blue','green','red', 'yellow']

    cmap = matplotlib.colors.ListedColormap(colors)   

    h = 0.01

    # Generate a grid of points with distance h between them

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid

    A = model(np.c_[xx.ravel(), yy.ravel()])

    A = A.reshape(xx.shape)

    # Plot the contour and training examples

    plt.contourf(xx, yy, A, cmap="spring")

    plt.ylabel('x2')

    plt.xlabel('x1')

    plt.scatter(X[0, :], X[1, :], c=y, s=8,cmap=cmap)

    plt.title("Decision Boundary")

    plt.show()
plot_decision_boundary1(X_data,Y,lambda x: predict(x.T,params))