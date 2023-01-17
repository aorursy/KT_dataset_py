
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
X = train.drop('label', axis=1).values 
width = int(np.sqrt(X.shape[1])) # this os only true if we know the images are square.
height = width # useful to be able to think of height and width separately, despite being the same here.
channels = 1 # number of colour channels in this case. 
# helper function for formatting the data for use with tensorflow (which prefers np.float32 inputs).
# the default values are for this data set.
def format_data(X, height=28, width=28, channels=1, normalisation=255.0):
    # Reshape each image from a single array/vector into a 3 dimensional tensor.
    # i.e. each channel of each image is represented as a matrix.
    x = np.reshape(X, (X.shape[0], height, width, channels))
    x = x.astype(np.float32)
    return x/normalisation
X_train = format_data(X)
X_test = format_data(test.values)
# helper to onehot encode the labels
# this requires that the labels are 0, 1, ...
def onehot_encode(y,num_classes=10):
    onehot = np.arange(num_classes)==y[:,None]
    return onehot.astype(np.float32)
y_train = onehot_encode(train['label'].values)
# helper functions for creating weights (W), biases (b), convolutions (conv) and pooling (pool).
def W(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
def b(shape):
    return tf.Variable(tf.constant(0.0, shape = shape))

# we fix padding as SAME here. 
# Can also be VALID, but you need to be careful with how this affects dimensions between layers.
# Strides are only in the width and height dimensions.
# we don't want to convolve between multiple images or channels (there's only one channel here anyway).
def conv(x,W, stride=1):
    return tf.nn.conv2d(x,W,strides = [1,stride,stride,1], padding = "SAME")

# Similarly the window and strides here are 1 in the image-index and channels dimensions as we want these separate.
def pool(x, window = 2, stride = 2):
    return tf.nn.max_pool(x,ksize = [1,window,window,1], strides = [1,stride,stride,1], 
                          padding = 'SAME')
# The model has 2 convolutional layers, followed by a fully connected layer and an output layer.
# Most the parameters can be edited. The strides in the convolutions and pooling, 
# and the pooling window are fixed, but these could be made paramters too.

def model(X, height = 28, width = 28, channels = 1, 
          first_layer_patch_size = 5, first_layer_depth = 32,
          second_layer_patch_size = 5, second_layer_depth = 64, 
          activation = tf.nn.relu, hidden_layer_depth = 1024, dropout_percent = 0.5, output_depth = 10):
    # First convolutional layer
    W1 = W([first_layer_patch_size,first_layer_patch_size,channels,first_layer_depth])
    b1 = b([first_layer_depth])
    h1 = activation(conv(X,W1) + b1)
    hp1 = pool(h1, stride=2)
    
    # Second convolutional layer
    W2 = W([second_layer_patch_size, second_layer_patch_size, first_layer_depth, second_layer_depth])
    b2 = b([second_layer_depth])
    h2 = activation(conv(hp1, W2) + b2)
    hp2 = pool(h2, stride=2)

    # Due to the strides in a pooling step, we shrink the image by 2 in each dimension. This is done twice.
    flat_dim = (height//(2*2)) * (width//(2*2))
    hp2_flat = tf.reshape(hp2, [-1, flat_dim * second_layer_depth])
    
    # fully connected layer
    W_fc = W([flat_dim*second_layer_depth, hidden_layer_depth])
    b_fc = b([hidden_layer_depth])
    h_fc = tf.nn.relu(tf.matmul(hp2_flat, W_fc) + b_fc)
    h_drop = tf.nn.dropout(h_fc, dropout_percent)
    W_o = W([hidden_layer_depth, output_depth])
    b_o = b([output_depth])
    return tf.matmul(h_drop, W_o) + b_o
# set up tensorflow graph
graph = tf.Graph()
with graph.as_default():
    # placeholders for inputting minibatch data, and a dropout percent.
    # Can use placeholders to feed in other parameters also.
    X = tf.placeholder(tf.float32, shape = (None, height, width, channels))
    y = tf.placeholder(tf.float32, shape = (None, 10))
    dropout = tf.placeholder(tf.float32, shape = ())
    
    # Use the model.
    logits = model(X, dropout_percent = dropout)
    # Cross entropy loss.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits))
    
    # Learning rate decay. 
    # The idea is as the model learns, we want the gradient descent to take finer steps.
    # Too larger steps might always overshoot a minimum loss. 
    # However too smaller steps take a long time to converge, 
    # and could also get stuck in a sub-optimal loss.
    # Decaying the step size over time should help the optimizer get near to a minimum in few steps, 
    # then fine tune around this minimum.
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0004
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.98, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    prediction = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = 100*tf.reduce_mean(tf.cast(correct, 'float'))
    predict = tf.argmax(prediction, 1)
# Need to split off a validation set from the train data.
# Reset X_train here as I've reused X_train as the variable name after the split.
X_train = format_data(train.drop('label', axis=1).values)
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42)
# validation cycle
num_steps = 501
batch_size = 50
train_accuracies = []
valid_accuracies = []
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        # Semi-stochastic mini-batch selection, as used in the Udacity deep learning course.
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict = {X : batch_data, y : batch_labels, dropout:0.75}
        
        # Note you seem to have to pass the optimizer out at each step.
        _  = session.run([optimizer], feed_dict=feed_dict)
        if (step % 100 == 0):
            print("Step {}".format(step))
            train_accuracy = accuracy.eval(feed_dict = feed_dict)
            train_accuracies.append(train_accuracy)
            print('Minibatch training accuracy: {}'.format(train_accuracy))
            valid_offset = (step*batch_size)%(y_valid.shape[0] - batch_size)
            valid_batch_data = X_valid[valid_offset:(valid_offset+batch_size), :,:,:]
            valid_batch_labels = y_valid[valid_offset:(valid_offset + batch_size),:]
            valid_accuracy = accuracy.eval(feed_dict= {X:valid_batch_data, y:valid_batch_labels, dropout:1})
            valid_accuracies.append(valid_accuracy)
            print('Minibatch validation accuracy: {}'.format(valid_accuracy))
    # Compute accuracy on the full validation set at the last step.
    full_validation_accuracy = accuracy.eval(feed_dict = {X:X_valid, y:y_valid, dropout:1})
    print("Full Validation Accuracy: ", full_validation_accuracy)
# Plots of the accuracies on minibatches.
import matplotlib.pyplot as plt
steps = np.arange(num_steps/100)*100
plt.plot(steps, train_accuracies, c = 'b', label = "Train")
plt.plot(steps, valid_accuracies, c='r', label = "Valid")
plt.legend(loc=4)
plt.ylabel("Accuracy")
plt.xlabel("Steps")
plt.show()
# full cycle for the test data
num_steps = 5001
batch_size = 100
train_accuracies = []

X_full_train = train.drop('label', axis = 1).values
y_full_train = onehot_encode(train['label'].values)
X_full_train = format_data(X_full_train)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for step in range(num_steps):
        offset = (step * batch_size) % (y_full_train.shape[0] - batch_size)
        batch_data = X_full_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_full_train[offset:(offset + batch_size), :]
        feed_dict = {X : batch_data, y : batch_labels, dropout:0.75}
        _ = session.run(
            [optimizer], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Step {}".format(step))
            train_accuracy = accuracy.eval(feed_dict = feed_dict)
            train_accuracies.append(train_accuracy)
            print('Minibatch accuracy: {}'.format(train_accuracy))
    test_size = X_test.shape[0]
    test_predictions = np.zeros(test_size)
    # Performance is better to predict in batches.
    for i in range(0, test_size//batch_size):
        test_predictions[i*batch_size:(i+1)*batch_size] = predict.eval(feed_dict = {X:X_test[i*batch_size:(i+1)*batch_size], dropout : 1})
        
submission_df = pd.DataFrame(test_predictions.astype(int))
submission_df.columns = ['Label']
submission_df['ImageId'] = submission_df.index + 1
submission_df = submission_df.set_index('ImageId')
submission_df.to_csv("submission.csv")
