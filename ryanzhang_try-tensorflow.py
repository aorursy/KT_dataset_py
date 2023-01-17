# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read in data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(train.shape)

print(test.shape)
# seperate out the training label

y = train["label"]

train.drop(labels = "label", axis = 1, inplace = True)
# check label balance

y.value_counts()
# my math is bad, how many labels are there?

len(y.unique())
# create validation set from training set

train, validation, train_labels, validation_labels = train_test_split(

    train, y, test_size = 0.3, random_state = 1106)



# convert to float32 np.ndarray also rescale to 0-1

train = train.astype(np.float32).values / 255

validation = validation.astype(np.float32).values / 255

test = test.astype(np.float32).values / 255

train_labels = (np.arange(10) == train_labels[:,None]).astype(np.float32)

validation_labels = (np.arange(10) == validation_labels[:,None]).astype(np.float32)



#define a accuracy function

def accuracy(predictions, labels):

    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

          / predictions.shape[0])
batch_size = 128

strength_of_regularizer = 0.01



# configure a simply network with only one hidden layer of ReLus and dropouts

graph = tf.Graph()

with graph.as_default():

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 784))

    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 10))

    tf_validation_dataset = tf.constant(validation)

    tf_test_dataset = tf.constant(test)

    

    # Variables, since we got two layers, we got more weights and biases

    weights_layer_1 = tf.Variable(tf.truncated_normal([784, 1024]))

    biases_layer_1 = tf.Variable(tf.zeros([1024]))

    weights_layer_2 = tf.Variable(tf.truncated_normal([1024, 10]))

    biases_layer_2 = tf.Variable(tf.zeros([10]))



    # Training computation.

    layer_1_nets = tf.matmul(tf_train_dataset, weights_layer_1) + biases_layer_1

    layer_1_activations = tf.nn.dropout(tf.nn.relu(layer_1_nets), 0.5) # apply drop out as well

    logits = tf.matmul(layer_1_activations, weights_layer_2) + biases_layer_2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    

    # Adding Regularization

    L2_regularizer = (tf.nn.l2_loss(weights_layer_1) + tf.nn.l2_loss(biases_layer_1) + 

                    tf.nn.l2_loss(weights_layer_2) + tf.nn.l2_loss(biases_layer_2))



    # Add the regularization term to the loss.

    loss += strength_of_regularizer * L2_regularizer  

    

    # Optimizer.

    optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

    

    # Predictions for the training

    train_prediction = tf.nn.softmax(logits)

    

    valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_validation_dataset, weights_layer_1) + biases_layer_1), weights_layer_2) + biases_layer_2)

    test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_layer_1) + biases_layer_1), weights_layer_2) + biases_layer_2)

    

# run the model

num_steps = 3000



np.random.seed(36)



with tf.Session(graph=graph) as session:

    # initialize weights

    tf.initialize_all_variables().run()

    print("Initialized")

    for step in range(num_steps):

        # Pick an offset within the training data, which has been randomized.

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

        # Generate a minibatch.

        batch_data = train[offset:(offset + batch_size), :]

        batch_labels = train_labels[offset:(offset + batch_size),:]

        # Prepare a dictionary telling the session where to feed the minibatch.

        # The key of the dictionary is the placeholder node of the graph to be fed,

        # and the value is the numpy array to feed to it.

        feed_dict = {tf_train_dataset  : batch_data, tf_train_labels : batch_labels}

        

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 1000 == 0):

            print("Minibatch loss at step %d: %f" % (step, l))

            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))

            print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validation_labels))

    test_predictions = np.argmax(test_prediction.eval(), 1)
submission = pd.DataFrame({

    "ImageId": range(1, test.shape[0] + 1),

    "label": test_predictions})

submission.to_csv("sub.csv", index = False)