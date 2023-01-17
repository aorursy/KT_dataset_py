import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
%matplotlib inline
import os
print(os.listdir("../input"))

#Load in our image dataset into a Numpy matrix
Image_path='../input/Sign-language-digits-dataset/X.npy' 
X = np.load(Image_path)
#Load in our classification into a Numpy matrix
label_path='../input/Sign-language-digits-dataset/Y.npy'
Y = np.load(label_path)
#Let's see the dimensions of our pixel matrix and classification matrix
print("Our feature vector is of size: " + str(np.shape(X)))
print("Our classification vector is of size: " + str(np.shape(Y)))
X[0] #Let's see how each image is stored
#Let's plot a few sample images, so we have a good sense of the type of images we are feeding into our training algorithm.
print('Sample images from dataset (this is 9 in sign language):')
n = 5
plt.figure(figsize=(15,15))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X[i])
    plt.gray()
    plt.axis('off')
#Let's split our data into test/training sets
#We'll use ~2/3 for training and the remaining 1/3 for testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

print("Training set feature matrix shape: " + str(X_train.shape))
print("Training set classification matrix shape: " + str(Y_train.shape))
print("Testing set feature matrix shape: " + str(X_test.shape))
print("Testing set classification matrix shape: " + str(Y_test.shape))
#Flatten our data
X_train_flat = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]).T #Flatten our data and transpose ~ (64*64,1381)
X_test_flat = X_test.reshape(X_test.shape[0],X_train.shape[1]*X_train.shape[2]).T #Flatten our data and transpose ~ (64*64,681)
Y_train = Y_train.T
Y_test = Y_test
print(str(X_train_flat.shape))
print(str(Y_train.shape))
print(str(X_test_flat.shape))
print(str(Y_test.shape))

#Let's create some tensorflow place holder values to be used in our model later.
def create_placeholders(n_x, n_y):
    
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    return X, Y
#Let's initialize the parameters to be used in our neural network
def initialize_params():
    
    W1 = tf.get_variable("W1",[25,4096],initializer = tf.contrib.layers.xavier_initializer()) #We will be using Xavier initialization for our weight parameters
    b1 = tf.get_variable("b1",[25,1],initializer=tf.zeros_initializer()) #We will be using a zero vector for our intercept parameter initialization
    W2 = tf.get_variable("W2",[15,25],initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2",[15,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3",[10,15],initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3",[10,1],initializer=tf.zeros_initializer())
    
    #Create a dictionary of our parameters to be used in forward propogation
    parameters = {"W1": W1,
                 "W2": W2,
                 "W3": W3,
                 "b1": b1,
                 "b2": b2,
                 "b3": b3}
    
    return parameters
#Define our forward propogation algorithm
def forward_prop(X,parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.tanh(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.tanh(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    
    return Z3
#Define our cost function
def cost_calc(Z3,Y,parameters):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels) + 0.001*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)))
    
    return cost
#Define our neural network model
def neural_net(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001):

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train_flat.shape                     # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create our placeholder variables of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize our parameters
    parameters = initialize_params()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_prop(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = cost_calc(Z3, Y,parameters)
    
    # Backpropagation: We will be using an Adam optimizer for our backward propogation algorithm
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        #Loop through 2000 iterations of our Adam optimizer to determine optimal parameters
        for i in range (1,2000):   
            a,b = sess.run([optimizer,cost],feed_dict={X: X_train, Y: Y_train})
            costs.append(b)
            
        parameters = sess.run(parameters)
        print ("Parameters have been optimized.")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters
    
    
parameters = neural_net(X_train_flat, Y_train, X_test_flat, Y_test.T, learning_rate = 0.0001,
         )
X_single_test = X_test[7] #Let's use the 7th image in our test set for our individual example.
Y_single_test = Y_test[7]
print("Here's the 7th example in our test set.  This is sign-language for 7.")
plt.imshow(X_single_test)
print("The classification vector associated with this image is: " + str(Y_single_test))
print("This is what our model should predict when we input this image.")
X_single_test_flat = X_test_flat[:,7:8]
Z3 = forward_prop(X_single_test_flat, parameters)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    a = sess.run(tf.sigmoid(Z3)) #We'll take the max of the sigmoid of the output vector to determine the prediction
z = 1*(a == np.max(a))
z.T
if int(z[2]) == 1:
    print("The image represents a 7 in sign-language")
    