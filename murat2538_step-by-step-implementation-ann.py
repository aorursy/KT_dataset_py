import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow.python.framework import ops
import math

import warnings
warnings.filterwarnings("ignore")
tf.__version__
#dataset
#test data,test label and trainin data and trainin label
(X_train_orig,y_train_orig),((X_test_orig,y_test_orig))=mnist.load_data()
index=100
fig,axs=plt.subplots(1,5,figsize=(20,10))

for i in range(5):
    digit=X_train_orig[index]
    #tahmin için kullanılacak resmin yeniden boyutlandırmesi
    digit=digit.reshape(28,28)
    axs[i].imshow(digit,plt.cm.binary)
    axs[i].set_title("number {}".format(y_train_orig[index]))
    index+=16
                     
def convert_to_one_hot(Y, C):
    """
    Y=test or train data label
    c=number of  classes
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
#flatten test and train images
X_train_flatten=X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten=X_test_orig.reshape(X_test_orig.shape[0],-1).T

#normalize images
X_train = X_train_flatten/255
X_test = X_test_flatten/255

y_train=convert_to_one_hot(y_train_orig,10)
y_test=convert_to_one_hot(y_test_orig,10)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))
def create_placeholders(n_x,n_y):
    """
    n_x=size of image vector[28*28]
    n_y=number of classes[0 to 9:so 10]
    
    X: placeholder for data input, shape[n_x,None]
    Y: placeholder for input classes, shape[n_y,None]
    """
    X=tf.placeholder(shape=(n_x,None),dtype=tf.float32)
    Y=tf.placeholder(shape=(n_y,None),dtype=tf.float32)
    
    return X,Y
X,Y=create_placeholders(784,10)
print("X:",str(X))
print("Y:",str(Y))
def initialize_parameters():
    """
        parameters shape ara
        W1=[128,X.shape[0]]
        b1=[128,1]
        W2=[64,128]
        b2=[64,1]
        W3=[20,64]
        b3=[20,1]
        W4=[10,20]
        b4=[10,1]
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
    
    #using tensorflow define parameter
    W1 = tf.get_variable("W1", [128,784], initializer =  tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [128,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [64,128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [64,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [20,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [20,1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [10,20], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [10,1], initializer = tf.zeros_initializer())
   
    
    parameters = { "W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,}
    
    return parameters
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    print("W3 = " + str(parameters["W3"]))
    print("b3 = " + str(parameters["b3"]))
    print("W4 = " + str(parameters["W4"]))
    print("b4 = " + str(parameters["b4"]))
def forward_propagation(X,parameters):
    
    """
    forward_propagataion?linear->relu->linear->relu->linear->relu->linear->softmax
    """
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]
    W4=parameters["W4"]
    b4=parameters["b4"]
    
    Z1=tf.add(tf.matmul(W1,X),b1)       #np.dot(W1,x)+b1
    A1=tf.nn.relu(Z1)
    Z2=tf.add(tf.matmul(W2,A1),b2)      #np.dot(W2,A1)+b2
    A2=tf.nn.relu(Z2)
    Z3=tf.add(tf.matmul(W3,A2),b3)      #np.dot(W3,A2)+b3
    A3=tf.nn.relu(Z3)
    Z4=tf.add(tf.matmul(W4,A3),b4)       #np.dot(W4,A3)+b4
    
    return Z4
    
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(784, 10)
    parameters = initialize_parameters()
    Z4 = forward_propagation(X, parameters)
    print("Z4 = " + str(Z4))
def compute_cost(Z4,Y):
    """
     Z3=output of forward propagation for  last linear layer
    """
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    
    cost=  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits, labels = labels))
    
    return cost
  
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(784, 10)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost=compute_cost(Z3,Y)
    print("Z3 = " + str(Z3))
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0): 
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    
    permutation = list(np.random.permutation(m))#Randomly permute a sequence, or return a permuted range.
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]#x[:,0:64],x[:,64:128]...
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(X_train,y_train,X_test,y_test,learning_rate=0.0001,
          num_epochs=50,minibatch_size=32):
    
    
    ops.reset_default_graph()
    tf.set_random_seed(1)    
    (n_x,m)=X_train.shape
    n_y=y_train.shape[0]
    
    seed=1
    
    costs=[] #for plotting cost fuction decleraed costs list
    
    #call placeholder fonction
    X,Y=create_placeholders(n_x,n_y)
    
    #initialize parameter
    parameters=initialize_parameters()
    
    #forward propagation
    Z3=forward_propagation(X,parameters)
    
    #compute cost
    cost=compute_cost(Z3,Y)
    
    #Backward propagation and define optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    #initialize all parameters
    init=tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        
        
        for epoch in range(num_epochs):
            
            epoch_cost=0
            
            number_of_minibatches=m/minibatch_size
            
            #chosing mini-batch
            seed=seed+1#for np.andom.seed()
            
            minibatches=random_mini_batches(X_train,y_train,minibatch_size,seed)
            
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y)=minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / minibatch_size
            
            if (epoch % 5 == 0):
                print ("Cost   after epoch %i: %f" % (epoch, epoch_cost))
            if (epoch % 2 == 0):
                costs.append(epoch_cost)
                
                    
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: y_test}))
           
        
        
        return parameters
                  
### Run the Model and ploting Cost
parameters = model(X_train, y_train, X_test, y_test)
def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    W4 = tf.convert_to_tensor(parameters["W4"])
    b4 = tf.convert_to_tensor(parameters["b4"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
              "W4": W4,
              "b4": b4}
    
    x = tf.placeholder("float", [784, 1])
    
    #for predict forward propagation
    z4 = forward_propagation(x, params)
    p = tf.argmax(z4)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction
index=100
fig,axs=plt.subplots(1,5,figsize=(20,10))

for i in range(5):
    digit=X_test_orig[index]
    digit_pred=digit.reshape(784,-1)
    #tahmin için kullanılacak resmin yeniden boyutlandırmesi
    digit=digit.reshape(28,28)
    axs[i].imshow(digit,plt.cm.binary)
    axs[i].set_title("predict {}".format(predict(digit_pred,parameters)))
    index+=16
                