import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.framework import ops
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape, test.shape
num_images = 6
m = train.shape[0]
idx = np.random.choice(m, size=num_images)
len(idx)
num_rows = 2
num_cols = 3

fig, ax = plt.subplots(nrows = num_rows, ncols = num_cols)
fig.set_size_inches(12,10)

for i, j in enumerate(idx):
    # Find the right place to put the images, a is the row in the figure and b is the column
    
    a = i//num_cols
    b = i%num_cols

    # Remove ticks
    
    ax[a][b].tick_params(
    which='both',
    left=False,
    right=False,
    bottom=False,
    top=False,
    labelleft = False,
    labelbottom=False)
    
    # Draw image and set x label as the actual label of the image i.e. the value of the digit in the image
    
    ax[a][b].imshow(np.array(train.loc[j][1:]).reshape(28,28), cmap=plt.get_cmap('gray'))
    ax[a][b].set_xlabel(str(train.loc[j][0]), fontsize = 50)

plt.show()
train.describe()
X = np.array(train.iloc[:,1:])
X.shape
X = X.reshape((m,28,28,1))
y = np.array(train.label)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y]
    return Y

y = convert_to_one_hot(y,10)
y[0:5]
train.label.head()
# Set random seed

seed = 5
np.random.seed(seed)

# Get random training index

train_index = np.random.choice(m, round(m*0.95), replace=False)
dev_index = np.array(list(set(range(m)) - set(train_index)))

# Make training and dev
#X_train = X
X_train = X[train_index]
X_dev = X[dev_index]
#y_train = y
y_train = y[train_index]
y_dev = y[dev_index]
X_train.shape
m_test = test.shape[0]
X_test = np.array(test).reshape((m_test,28,28,1))

X_test.shape
X_train = X_train/255.
X_dev = X_dev/255.
X_test = X_test/255.
X_test = np.float32(X_test)
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of validation (dev) examples = " + str(X_dev.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("X_dev shape: " + str(X_dev.shape))
print ("y_dev shape: " + str(y_dev.shape))
print ("X_test shape: " + str(X_test.shape))
# Create Placeholders

def create_placeholders(n_H0,n_W0,n_C0,n_y):
    X = tf.placeholder(dtype = tf.float32,shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(dtype = tf.float32,shape = [None, n_y])
    drop_rate = tf.placeholder(dtype=tf.float32, name = 'drop_rate')
    return X, Y, drop_rate
# Initialize parameters

def initialize_parameters():
    tf.set_random_seed(1)
    initializer = tf.contrib.layers.xavier_initializer(seed = 0)
    W1a = tf.get_variable(name = 'W1a', shape = [7, 7, 1, 20], initializer = initializer)
    W1b = tf.get_variable(name = 'W1b', shape = [5, 5, 1, 20], initializer = initializer)
    W1c = tf.get_variable(name = 'W1c', shape = [3, 3, 1, 20], initializer = initializer)
    W2a = tf.get_variable(name = 'W2a', shape = [5, 5, 60, 50], initializer = initializer)
    W2b = tf.get_variable(name = 'W2b', shape = [3, 3, 60, 50], initializer = initializer)
    W3 = tf.get_variable(name = 'W3', shape = [3, 3, 100, 120], initializer = initializer)
    parameters = {"W1a": W1a,
                  "W1b": W1b,
                  "W1c": W1c,
                  "W2a": W2a,
                  "W2b": W2b,
                  "W3": W3,}
    
    return parameters
# Check parameters

tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1a = " + str(parameters["W1a"].eval()[1,1,0]))
    print("W1b = " + str(parameters["W1b"].eval()[1,1,0]))
    print("W1c = " + str(parameters["W1c"].eval()[1,1,0]))
    print("W2a = " + str(parameters["W2a"].eval()[1,1,1]))
    print("W2b = " + str(parameters["W2b"].eval()[1,1,1]))
    print("W3 = " + str(parameters["W3"].eval()[1,1,0]))
# Build forward propagation computation graph

def forward_propagation(X, parameters, drop_rate):
    W1a = parameters['W1a']
    W1b = parameters['W1b']
    W1c = parameters['W1c']
    W2a = parameters['W2a']
    W2b = parameters['W2b']
    W3 = parameters['W3']
    #rate_1, rate_2, rate_3 = drop_rate
 
    # CONV2D: stride of 1, padding 'SAME'
    Z1a = tf.nn.conv2d(X,W1a, strides = [1,1,1,1], padding = 'SAME')
    Z1b = tf.nn.conv2d(X,W1b, strides = [1,1,1,1], padding = 'SAME')
    Z1c = tf.nn.conv2d(X,W1c, strides = [1,1,1,1], padding = 'SAME')
    Z1 = tf.concat([Z1a, Z1b, Z1c], axis = -1)
    # RELU
    A1 = tf.nn.relu(tf.layers.batch_normalization(Z1))
    # MAXPOOL: window 2x2, stride 2, padding 'VALID'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2a = tf.nn.conv2d(P1,W2a, strides = [1,1,1,1], padding = 'SAME')
    Z2b = tf.nn.conv2d(P1,W2b, strides = [1,1,1,1], padding = 'SAME')
    Z2 = tf.concat([Z2a, Z2b], axis = -1)
    # RELU
    A2 = tf.nn.relu(tf.layers.batch_normalization(Z2))
    # MAXPOOL: window 2x2, stride 2, padding 'VALID'
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # CONV2D: filters W3, stride 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A3 = tf.nn.relu(tf.layers.batch_normalization(Z3))
    # MAXPOOL: window 2x2, stride 2, padding 'VALID'
    P3 = tf.nn.max_pool(A3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # FLATTEN
    P3 = tf.contrib.layers.flatten(P3)
    # FULLY-CONNECTED 
    Z4 = tf.layers.dropout(tf.contrib.layers.fully_connected(P3, 200, normalizer_fn=tf.layers.batch_normalization), rate = drop_rate)
    # FULLY-CONNECTED 
    Z5 = tf.layers.dropout(tf.contrib.layers.fully_connected(Z4, 120, normalizer_fn=tf.layers.batch_normalization), rate = drop_rate)
    # FULLY-CONNECTED 
    Z6 = tf.layers.dropout(tf.contrib.layers.fully_connected(Z5, 84, normalizer_fn=tf.layers.batch_normalization), rate = drop_rate)
    # FULLY-CONNECTED 
    Z7 = tf.contrib.layers.fully_connected(Z6, 10, normalizer_fn=tf.layers.batch_normalization, activation_fn = tf.nn.softmax)
    
    return Z7
# Check forward propagation

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y, drop_rate = create_placeholders(28, 28, 1, 10)
    parameters = initialize_parameters()
    Z7 = forward_propagation(X, parameters, drop_rate)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z7, {X: np.random.randn(2,28,28,1), Y: np.random.randn(2,10), drop_rate: 0})
    print("Z7 = " + str(a))
    
# Compute Cost

def compute_cost(Z7, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z7, labels = Y))
    return cost
# Check cost function

tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y, drop_rate = create_placeholders(28, 28, 1, 10)
    parameters = initialize_parameters()
    Z7 = forward_propagation(X, parameters, drop_rate)
    cost = compute_cost(Z7, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,28,28,1), Y: np.random.randn(4,10), drop_rate: 0})
    print("cost = " + str(a))
# Set hyperparameters and optimization function

learning_rate = 7e-5
num_epochs = 20
batch_size = 16

def model(X_train, X_dev, X_test, y_train, y_dev, learning_rate = learning_rate, num_epochs = num_epochs, batch_size = batch_size, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (m_train, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    if m_train%batch_size !=0:
        num_batches = (m_train//batch_size) + 1
    else:
        num_batches = m_train//batch_size
    
    # Create Placeholders of the correct shape
    
    X, Y, drop_rate = create_placeholders(n_H0, n_W0, n_C0, n_y)
    

    # Initialize parameters
    
    parameters = initialize_parameters()
    
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    
    Z7 = forward_propagation(X, parameters, drop_rate)
    
    
    # Cost function: Add cost function to tensorflow graph
    
    cost = compute_cost(Z7, Y)
    
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
  

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        for epoch in range(num_epochs):
            # Generate random batch index
            minibatch_cost = 0
            full_batch = range(m_train)

            for batch in range(num_batches):        
                try:
                    batch_index = np.random.choice(full_batch, size=batch_size, replace = False)
                    full_batch = np.array(list(set(full_batch) - set(batch_index)))
                except ValueError:
                    batch_index = full_batch
                batch_train_X = X_train[batch_index]
                batch_train_y = y_train[batch_index]

                # Run session to reach goal 

                sess.run(optimizer, feed_dict={X: batch_train_X, Y: batch_train_y, drop_rate: 0.4})
                temp_cost = sess.run(cost, feed_dict={X: batch_train_X, Y: batch_train_y, drop_rate: 0})
                minibatch_cost += temp_cost / num_batches

            # Print the cost every epoch
            
            print ("Cost after epoch %i: %f" % (epoch+1, minibatch_cost))
            costs.append(minibatch_cost)


        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        #Z6 = forward_propagation(X, parameters, rate = [1, 1])
        predict_op = tf.argmax(Z7, 1)
        #correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
      
        # Calculate accuracy on the train and dev sets
        
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        #print(accuracy)
        #train_accuracy = accuracy.eval({X: X_train, Y: y_train})
        #dev_accuracy = accuracy.eval({X: X_dev, Y: y_dev})
        
        #print("Dev Accuracy:", dev_accuracy)
        
        # Make predictions
        train_preds = np.empty(shape = m_train)
        for batch in range(num_batches):
            if batch != num_batches - 1:
                batch_index = range(batch*batch_size, (batch+1)*batch_size)
            else:
                batch_index = range(batch*batch_size,m_train)
            X_train_batch = X_train[batch_index]
            Y_train_batch_preds = sess.run(predict_op, feed_dict ={X: X_train_batch, drop_rate: 0})
            train_preds[batch_index] = Y_train_batch_preds
            #print('Train batch {} completed'.format(batch+1))
        
        m_dev = X_dev.shape[0]
        dev_preds = np.empty(shape = m_dev)
        if m_dev%batch_size !=0:
            num_batches = (m_dev//batch_size) + 1
        else:
            num_batches = m_dev//batch_size
        
        for batch in range(num_batches):
            if batch != num_batches - 1:
                batch_index = range(batch*batch_size, (batch+1)*batch_size)
            else:
                batch_index = range(batch*batch_size,m_dev)
            X_dev_batch = X_dev[batch_index]
            Y_dev_batch_preds = sess.run(predict_op, feed_dict ={X: X_dev_batch, drop_rate: 0})
            dev_preds[batch_index] = Y_dev_batch_preds
        
        m_test = X_test.shape[0]
        test_preds = np.empty(shape = m_test)
        if m_test%batch_size !=0:
            num_batches = (m_test//batch_size) + 1
        else:
            num_batches = m_test//batch_size
        
        for batch in range(num_batches):
            if batch != num_batches - 1:
                batch_index = range(batch*batch_size, (batch+1)*batch_size)
            else:
                batch_index = range(batch*batch_size,m_test)
            X_test_batch = X_test[batch_index]
            Y_test_batch_preds = sess.run(predict_op, feed_dict ={X: X_test_batch, drop_rate: 0})
            test_preds[batch_index] = Y_test_batch_preds
            #print('Train batch {} completed'.format(batch+1))
        #train_preds = sess.run(predict_op, feed_dict ={X:X_train})
        #test_preds = sess.run(predict_op, feed_dict ={X:X_test})
        train_accuracy = np.mean(train_preds.astype(int)==np.argmax(y_train,1))
        print("Train Accuracy:", train_accuracy)
        dev_accuracy = np.mean(dev_preds.astype(int)==np.argmax(y_dev,1))
        print("Dev Accuracy:", dev_accuracy)
        
        return parameters, train_preds, dev_preds, test_preds
parameters, train_preds, dev_preds, test_preds = model(X_train, X_dev, X_test, y_train, y_dev)
i = np.random.choice(m_test)
print("Test sample no.: {}".format(i))

print('Prediction: {}'.format(test_preds[i]))
plt.imshow(X_test[i,:,:,0],cmap = plt.get_cmap('gray'))
plt.show()

train_labels = np.argmax(y_train, axis = 1)
train_accuracy = np.mean(train_labels==train_preds.astype(int))
train_accuracy
dev_labels = np.argmax(y_dev, axis = 1)
dev_accuracy = np.mean(dev_labels==dev_preds.astype(int))
dev_accuracy
dev_new = pd.DataFrame()
dev_new['Label'] = dev_labels
dev_new['Preds'] = dev_preds.astype(int)
m_dev = X_dev.shape[0]
dev_new['ImageId'] = list(range(m_dev))
dev_new.head()
dev_mismatch = dev_new[dev_new['Label']!=dev_new['Preds']]
dev_mismatch.Label.value_counts()
i = np.random.choice(dev_mismatch['ImageId'])
print("Image ID.: {}".format(i))
print('Prediction: {}'.format(int(dev_preds[i])))
print('Correct Label: {}'.format(dev_labels[i]))
plt.imshow(X_dev[i,:,:,0],cmap = plt.get_cmap('gray'))
plt.show()
test['Label'] = test_preds.astype(int)
test['ImageId'] = list(range(1,m_test+1))
test.head()
test[['ImageId', 'Label']].to_csv('submission_lenet5.csv', index = False, header = ['ImageId','Label'])