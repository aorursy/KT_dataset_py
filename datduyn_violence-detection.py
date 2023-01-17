import os
print(os.listdir("../input"))

# As usual, a bit of setup
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
def train_part34(model_init_fn, optimizer_init_fn, num_epochs=1):
    """
    Simple training loop for use with models defined using tf.keras. It trains
    a model for one epoch on the CIFAR-10 training set and periodically checks
    accuracy on the CIFAR-10 validation set.
    
    Inputs:
    - model_init_fn: A function that takes no parameters; when called it
      constructs the model we want to train: model = model_init_fn()
    - optimizer_init_fn: A function which takes no parameters; when called it
      constructs the Optimizer object we will use to optimize the model:
      optimizer = optimizer_init_fn()
    - num_epochs: The number of epochs to train for
    
    Returns: Nothing, but prints progress during trainingn
    """
    tf.reset_default_graph()    
    with tf.device(device):
        # Construct the computational graph we will use to train the model. We
        # use the model_init_fn to construct the model, declare placeholders for
        # the data and labels
        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.int32, [None])
        
        # We need a place holder to explicitly specify if the model is in the training
        # phase or not. This is because a number of layers behaves differently in
        # training and in testing, e.g., dropout and batch normalization.
        # We pass this variable to the computation graph through feed_dict as shown below.
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        # Use the model function to build the forward pass.
        scores = model_init_fn(x, is_training)
        
        # Compute the loss like we did in Part II
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        loss = tf.reduce_mean(loss)

        # Use the optimizer_fn to construct an Optimizer, then use the optimizer
        # to set up the training step. Asking TensorFlow to evaluate the
        # train_op returned by optimizer.minimize(loss) will cause us to make a
        # single update step using the current minibatch of data.
        
        # Note that we use tf.control_dependencies to force the model to run
        # the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
        # holds the operators that update the states of the network.
        # For example, the tf.layers.batch_normalization function adds the running mean
        # and variance update operators to tf.GraphKeys.UPDATE_OPS.
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

    # Now we can run the computational graph many times to train the model.
    # When we call sess.run we ask it to evaluate train_op, which causes the
    # model to update.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t = 0
        for epoch in range(num_epochs):
            print('Starting epoch %d' % epoch)
            for x_np, y_np in train_dset:
                feed_dict = {x: x_np, y: y_np, is_training:1}
                loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if t % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss_np))
                    check_accuracy(sess, val_dset, x, scores, is_training=is_training)
                    print()
                t += 1

def model_init_fn(inputs, is_training):
    model = None
    ############################################################################
    # TODO: Construct a model that performs well on CIFAR-10                   #
    #Pool dim equation: 1 + (H - pool_h) // stride
    ############################################################################
    ################################VGG Architecture############################
#     hidden_dim = 100
#     input_shape = (32, 32, 3)
#     num_classes = 10
#     intializer = tf.variance_scaling_initializer(scale=2.0)
#     layers = [
#         tf.keras.layers.InputLayer(input_shape=input_shape),
        
#         tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="SAME", 
#                activation=tf.nn.relu, kernel_initializer=intializer),
#         tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding="SAME", 
#                activation=tf.nn.relu, kernel_initializer=intializer),
#         tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #Shape of one images (16x16)
        
        
#         tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="SAME", 
#                activation=tf.nn.relu, kernel_initializer=intializer),
#         tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="SAME", 
#                activation=tf.nn.relu, kernel_initializer=intializer),
#         tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)), #Shape of one images (8x8)
        
#         #Fully connected layer
#         tf.keras.layers.Flatten(input_shape=(8,8,16)),
#         tf.keras.layers.Dense(hidden_dim, activation=tf.nn.relu, kernel_initializer=intializer),
#         tf.keras.layers.Dense(num_classes, kernel_initializer=intializer)
#     ]
#     model = tf.keras.Sequential(layers)
#     net = model(inputs)
    ###########################################################################
    
    ###########################################################################
    ##################THIS TAKE EXTREMELY long to train.:( #######################
    ## [conv <-> bn <-> relu <-> drop]x2 <-> max_pool
    initializer = tf.variance_scaling_initializer(scale=2.0)
    cv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(3,3),
                               strides=(1,1), padding="same",kernel_initializer=initializer)
    bn1 = tf.layers.batch_normalization(inputs=cv1, training=is_training)
    relu1 = tf.nn.relu(bn1)
    d1 = tf.layers.dropout(inputs=relu1, rate=0.5)
    
    cv2 = tf.layers.conv2d(inputs=d1, filters=32, kernel_size=(3,3),
                               strides=(1,1), padding="same",kernel_initializer=initializer)
    bn2 = tf.layers.batch_normalization(inputs=cv2, training=is_training)
    relu2 = tf.nn.relu(bn2)
    d2 = tf.layers.dropout(inputs=relu2, rate=0.5)
    pool1 = tf.layers.max_pooling2d(inputs=d2, pool_size=(2,2), strides=(2,2)) #shape (16 x 16) 
    ##########
    
    ## [conv <-> bn <-> relu <-> drop]x2 <-> max_pool
    cv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(3,3),
                               strides=(1,1), padding="same",kernel_initializer=initializer)
    bn3 = tf.layers.batch_normalization(inputs=cv3, training=is_training)
    relu3 = tf.nn.relu(bn3)
    d3 = tf.layers.dropout(inputs=relu3, rate=0.5)
    
    cv4 = tf.layers.conv2d(inputs=d3, filters=64, kernel_size=(3,3),
                               strides=(1,1), padding="same",kernel_initializer=initializer)
    bn4 = tf.layers.batch_normalization(inputs=cv4, training=is_training)
    relu4 = tf.nn.relu(bn4)
    d4 = tf.layers.dropout(inputs=relu4, rate=0.5)
    pool2 = tf.layers.max_pooling2d(inputs=d4, pool_size=(2,2), strides=(2,2)) #shape (16 x 16) 
    #########
    
    flat = tf.layers.flatten(inputs=pool2)
    scores = tf.layers.dense(inputs=flat, units=100)# hidden unit
    scores = tf.layers.dense(inputs=scores, units=10) #num class
    net = scores 
    ###########################################################################
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################
    return net

'''
def two_layer_fc_functional(inputs, hidden_size, num_classes):     
    initializer = tf.variance_scaling_initializer(scale=2.0)
    flattened_inputs = tf.layers.flatten(inputs)
    fc1_output = tf.layers.dense(flattened_inputs, hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=initializer)
    scores = tf.layers.dense(fc1_output, num_classes,
                             kernel_initializer=initializer)
    return scores
'''
learning_rate = 1e-3
def optimizer_init_fn():
    optimizer = None
    ############################################################################
    # TODO: Construct an optimizer that performs well on CIFAR-10              #
    ############################################################################
    optimizer = tf.train.AdamOptimizer(learning_rate)
    ############################################################################
    #                            END OF YOUR CODE                              #
    ############################################################################
    return optimizer


device = '/cpu:0'
def test_net_functional():
    """ A small unit test to exercise the TwoLayerFC model above. """
    tf.reset_default_graph()

    # As usual in TensorFlow, we first need to define our computational graph.
    # To this end we first construct a two layer network graph by calling the
    # two_layer_network() function. This function constructs the computation
    # graph and outputs the score tensor.
    with tf.device(device):
        x = tf.zeros((23, 32, 32, 3))
        scores = model_init_fn(x, True)

    # Now that our computational graph has been defined we can run the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores_np = sess.run(scores)
        print(scores_np.shape)
print("Test Model Forward pass")
test_net_functional()

device = '/cpu:0'
print_every = 300
num_epochs = 10
train_part34(model_init_fn, optimizer_init_fn, num_epochs)