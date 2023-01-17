import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import re
import os
import cv2
import math
image_width = 150
image_height = 150
number_of_color_channels = 3 #RGB
data = "../input/train/"
data_set = [data+idx for idx in os.listdir(path=data)]
def atoi(text):
    if text.isdigit() == True: #text contains digits only
        return int(text)
    else:
        return text
    
def natural_keys(text):
    return [atoi(idx) for idx in re.split(pattern='(\d+)', string=text)]
data_set.sort(key=natural_keys)
#data_set = data_set[0:5000] + data_set[12500:17500]
data_set = data_set[0:2500] + data_set[12500:15000]
#data_set = data_set[0:3000] + data_set[12500:15500]
def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = np.zeros(shape=(len(list_of_images), image_width, image_height, number_of_color_channels))
    y = np.zeros(shape=(1, len(list_of_images)))
    
    for (index, image) in enumerate(list_of_images):
        #read_image = np.array(ndimage.imread(image, flatten=False)) #deprecated in SciPy 1.0.0
        #my_image = scipy.misc.imresize(read_image, size=(150,150)) #deprecated in Scipy 1.0.0, will be removed in SciPy 1.2.0
        read_image = np.array(cv2.imread(image))
        my_image = cv2.resize(src=read_image, dsize=(image_width, image_height))
        x[index] = my_image
    
    for (index, image) in enumerate(list_of_images):
        if 'dog' in image:
            y[:, index] = 1
        elif 'cat' in image:
            y[:, index] = 0
            
    return x, y
x_all, y_all = prepare_data(data_set)
def numpy_array_properties(array):
    print("type:{}, shape:{}, dimensions:{}, size:{}, datatype:{}".format(type(array), array.shape, array.ndim, array.size, array.dtype))
numpy_array_properties(x_all)
numpy_array_properties(y_all)
fig = plt.figure()
img1 = fig.add_subplot(1,2,1) #1 row, 2 coulmns, image fills the 1st column
img1.imshow(x_all[50])
img2 = fig.add_subplot(1,2,2) #1 row, 2 columns, image fills the 2nd column
img2.imshow(np.array(cv2.imread(data_set[50])))
fig = plt.figure()
img1 = fig.add_subplot(1,2,1) #1 row, 2 coulmns, image fills the 1st column
img1.imshow(x_all[4305]) 
img2 = fig.add_subplot(1,2,2)
img2.imshow(np.array(cv2.imread(data_set[4305]))) #1 row, 2 columns, image fills the 2nd column
x_all_flatten = x_all.reshape(x_all.shape[0], -1).T
x_all_flatten_normalized = x_all_flatten/255.
numpy_array_properties(x_all_flatten_normalized)
def synch_shuffle_data(x, y):
    np.random.seed(1)
    permutation = list(np.random.permutation(x.shape[1]))
    x_shuffled = x[:, permutation]
    y_shuffled = y[:, permutation]
    return x_shuffled, y_shuffled
x_all_flatten_normalized_shuffled, y_all_shuffled = synch_shuffle_data(x_all_flatten_normalized, y_all)
numpy_array_properties(x_all_flatten_normalized_shuffled)
numpy_array_properties(y_all_shuffled)
x_train = x_all_flatten_normalized_shuffled[:, 0:3000]
x_dev = x_all_flatten_normalized_shuffled[:, 3000:4000]
x_test = x_all_flatten_normalized_shuffled[:, 4000:5000]
y_train = y_all_shuffled[:, 0:3000]
y_dev = y_all_shuffled[:, 3000:4000]
y_test = y_all_shuffled[:, 4000:5000]
numpy_array_properties(x_train)
numpy_array_properties(y_train)
numpy_array_properties(x_dev)
numpy_array_properties(y_dev)
numpy_array_properties(x_test)
numpy_array_properties(y_test)
print(x_train)
print("*"*100)
print(y_train)
print("*"*100)
print(x_dev)
print("*"*100)
print(y_dev)
print("*"*100)
print(x_test)
print("*"*100)
print(y_test)
def recreate_image_from_numpy_array(x, y, idx):
    x = x*255. #undo normalize
    x = x.T.reshape(x.shape[1], 150, 150, 3) #undo flatten ex: x_dev.shape = (400, 150, 150, 3)
    print("Label for the below image is {}".format(int(np.squeeze(y[:, idx]))))
    plt.imshow(x[idx])
recreate_image_from_numpy_array(x_dev, y_dev, 397)
recreate_image_from_numpy_array(x_dev, y_dev, 394)
def create_placeholders(n_x, n_y, m):
    X = tf.placeholder(name="X", shape=(n_x, None), dtype=tf.float32)
    Y = tf.placeholder(name="Y", shape=(n_y, None), dtype=tf.float32)
    #the below two lines of code will cause problems. 
    #Firstly, there is no freedom in varying number of examples(m).
    #When using minibatches, the last mininbatch size may differ from the rest of the mininbatches
    #Secondly, it will need training and dev sets to consist of same number of examples.
    #X = tf.placeholder(name="X", shape=(n_x, m), dtype=tf.float32)
    #Y = tf.placeholder(name="Y", shape=(n_y, m), dtype=tf.float32)
    
    return X, Y
def initialize_parameters():
    #just so that initialization remains consistent. can be removed.
    #In fact, its better to not use seeding becaus maybe it will lead to a better performance(reach global minimum instead of local minimum).
    tf.set_random_seed(1) 
    W1 = tf.get_variable(name="W1", shape=(125, 67500), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable(name="b1", shape=(125, 1), 
                         initializer=tf.zeros_initializer())
    W2 = tf.get_variable(name="W2", shape=(50, 125), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable(name="b2", shape=(50, 1), 
                         initializer=tf.zeros_initializer())
    W3 = tf.get_variable(name="W3", shape=(50, 50), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable(name="b3", shape=(50, 1), 
                         initializer=tf.zeros_initializer())
    W4 = tf.get_variable(name="W4", shape=(1, 50), 
                         initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable(name="b4", shape=(1, 1), 
                         initializer=tf.zeros_initializer())
    parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3, "W4":W4, "b4":b4}
    return parameters
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    W4 = parameters["W4"]
    b4 = parameters["b4"]
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    return Z4
def compute_cost(Z4, Y, regularize, parameters):
    logits = tf.transpose(Z4)
    labels = tf.transpose(Y)
    lambd = 0.01
    
    if regularize == 'no':
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(name="cost", logits=logits, labels=labels))
    elif regularize == 'yes':
        cost = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(name="cost", logits=logits, labels=labels)) +
                lambd*tf.nn.l2_loss(parameters["W1"]) + 
                lambd*tf.nn.l2_loss(parameters["W2"]) + 
                lambd*tf.nn.l2_loss(parameters["W3"]) + 
                lambd*tf.nn.l2_loss(parameters["W4"]))
    return cost
def random_mini_batches(X, Y, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for dog / 0 for cat), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)           
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*k:mini_batch_size*(k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size*num_complete_minibatches:m]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*num_complete_minibatches:m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, learning_rate = 0.01, 
          minibatch_size = 64, num_epochs = 250, print_cost = True):
    
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    seed = 0
    
    X, Y = create_placeholders(n_x=n_x, n_y=n_y, m=m)
    parameters = initialize_parameters()
    Z4 = forward_propagation(parameters=parameters, X=X)
    cost = compute_cost(Y=Y, Z4=Z4, regularize='no', parameters=parameters)
        
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                           # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch               
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run(fetches=[optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            costs.append(epoch_cost)
            if print_cost == True and epoch%10 == 0:
                print("Cost after epoch {} is : {}".format(epoch, epoch_cost))
                
        plt.plot(np.squeeze(costs))
        plt.ylabel("Cost")
        plt.xlabel("Iterations")
        plt.title("Learning rate: {}".format(learning_rate))
        plt.show()
        
        #lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
        
        correct_prediction = tf.equal(x=tf.round(x=tf.sigmoid(Z4)), y=Y)
        accuracy = tf.reduce_mean(tf.cast(x=correct_prediction, dtype="float"))
        
        print("Train accuracy:", accuracy.eval({X:X_train, Y:Y_train}))
        print("Dev set accuracy:", accuracy.eval({X:X_dev, Y:Y_dev}))
        print("Test set accuracy:", accuracy.eval({X:X_test, Y:Y_test}))
        
        return parameters
parameters = model(X_train=x_train, Y_train=y_train, X_dev=x_dev, Y_dev=y_dev, X_test=x_test, Y_test=y_test)
