import numpy as np 



# sigmoid activation function

def sigmoid(x):

    return 1/(1+np.exp(-x))



# derivative of sigmoid 

def deriv(x):

     return x*(1-x)



def network_0(X,y):

    """network with zero hidden layers

       can't solve the XOR problem

    """



    # initialize weights to nums btn 0-1 (connects directly from input to output)

    weights = 2 * np.random.random((3, 1)) - 1



    for i in range(10000): #ten thousand training iterations

        # activation

        act = sigmoid(np.dot(X, weights))



        # calculate error

        error = y - act



        # gradient = error * derivative of sigmoid( activation )

        delta = error * deriv(act)



        # weight update

        weights += np.dot(X.T, delta)



    print("Prediction:")

    print(act)

    



#input dataset

X = np.array([[0,0,1],

              [1,1,1],

              [1,0,1],

              [0,1,1],])



# output dataset

y = np.array([[0,1,1,0]]).T



# seed random numbers for reproducibility

np.random.seed(1)



network_0(X,y)
X = np.array([[0,0,1], #XOR prob

              [0,1,1],

              [1,0,1],

              [1,1,1],])



network_0(X,y)
def network_1(X,y):

    """network with one hidden layer, no bias units.

       can solve the XOR problem

    """

    # weights between input and hidden

    weight1 = 2*np.random.random((3,4)) - 1



    # weights between hidden and output

    weight2 = 2*np.random.random((4,1)) - 1



    for i in range(60000):

        #feed forward

        act1 = sigmoid(np.dot(X,weight1))

        act2 = sigmoid(np.dot(act1,weight2))

        

        # a loss function, technically

        error = y - act2



        # report error intermittently

        if (i%10000) == 0:

            print("Error at :",i,"is "+str(np.mean(np.abs(error))))



        # calc rate of change of error  wrt to act2

        act2_delta = error*sigmoid(act2)



        # calc each act1's contribution to error

        act1_error = act2_delta.dot(weight2.T)



        # calc rate of change of error  w.r.t to act1

        act1_delta = act1_error * deriv(act1)



        # Update weights, aka backprop

        weight2 += act1.T.dot(act2_delta)

        weight1 += X.T.dot(act1_delta)



    print("Prediction:")

    print(act2)

    



X = np.array([[0,0,1], #XOR prob

              [0,1,1],

              [1,0,1],

              [1,1,1],])



# output dataset, same as before

y = np.array([[0,1,1,0]]).T 



# seed random numbers for reproducibility

np.random.seed(1)



# run 1-hidden layer network

network_1(X, y)

# Network0 implemented in tenserFlow

import tensorflow as tf

    

# ----------------design network architecture



# define variables

X = tf.convert_to_tensor(X, dtype=tf.float32) # convert np X to a tensor

y = tf.convert_to_tensor(y, dtype=tf.float32) # convert np y to a tensor

W1 = tf.Variable(tf.random_uniform([3, 4],minval=-1,maxval=1))

W2 = tf.Variable(tf.random_uniform([4, 1],minval=-1,maxval=1))



# define operations

a1 = tf.matmul(X, W1)

a2 = tf.matmul(a1, W2)





# ---------------define loss and select training algorithm

loss = tf.losses.absolute_difference(y,a2)

# loss = y - a2

optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)



# ----------------run graph to train and get result

with tf.Session() as sess:



    # initialize variables

    sess.run(tf.initialize_all_variables())



    for i in range(10000):

        sess.run(train)

        if i % 1000 == 0:

            print("Loss: ", sess.run(loss))



    print("Output: ", sess.run(a2))

    print("Loss: ", sess.run(loss))
