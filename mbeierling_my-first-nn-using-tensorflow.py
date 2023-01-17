# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd



import tensorflow as tf



import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv("../input/train.csv")
train_data.shape
train_data.head() # we see that the first column is our target label.
train_data.loc[0:2]
# define a "next batch" function that will return X image-label batches

def next_batch(c, last_index, df):

    '''

    c: amount of batches

    last_index: the index where to start getting the c datapoints

    df: dataframe to get the data from

    

    returns: tuple, where 

        index 0 holds c arrays with 784 pixel values

        index 1 holds c arrays with 10 label values in one-hot format

        

    This is built to work like the 'tensorflow.examples.tutorials.mnist.train.next_batch' function

    '''

    temp_data = df.loc[last_index : last_index + c - 1]

    last_index += c

    

    # Output has to be

    # tuple-> ( 

    #  np.array([<arrays of pixel values per image>]), 

    #  np.array([<arrays of one-hot target values>]) 

    # )

    

    temp_pixel_data = temp_data[temp_data.columns[1:]]

    temp_label_data = temp_data[temp_data.columns[:1]]

    

    one_hot_label_prior = temp_label_data.as_matrix().flatten() # will give us a flat array of labels

    

    count = len(one_hot_label_prior)

    one_hot_label_final = np.zeros((count, 10))

    one_hot_label_final[np.arange(count), one_hot_label_prior] = 1

    

    final_batch_out = (temp_pixel_data.as_matrix(), one_hot_label_final)

    

    return (last_index, final_batch_out)
# Let's do 2 test outputs:



curr_i = 0

curr_i, curr_batch = next_batch(4, curr_i, train_data)





curr_batch
curr_i, curr_batch = next_batch(4, curr_i, train_data)

curr_batch
curr_i
# Nice! Our custom next_batch function works exactly as we want it to!
# Let's get a sample and render it

_, batch = next_batch(1, 0, train_data) # get first image; we don't care about our index-increment here
img_sample = batch[0][0] # get first tuple entry and first and only img pixel data array

img_sample
img_sample = img_sample.reshape(28,28) # because our images have 784 total pixels = 28*28
plt.imshow(img_sample, cmap='Greys') # looks like a '1' to me!
# split data into training & testing sets using a random boolean mask

msk = np.random.rand(len(train_data)) < 0.8



train_data_split = train_data[msk]

test_data_split = train_data[~msk]



print('train_data_split: {}'.format( train_data_split.shape ))

print('test_data_split: {}'.format( test_data_split.shape ))

print('train to test: {:1.4f}%'.format( train_data_split.shape[0] * 100. / train_data.shape[0] ))
# let's lay out our parameters

learning_rate = 0.001 # the lower the higher possibility for accurate training result

training_epochs = 50 # how many times we run learning. higher takes longer but yields better accuracy

batch_size = 100 # batches of training data



n_classes = 10 # there are 10 numbers: 0-9

n_samples = train_data_split.shape[0]

n_input = len(train_data_split.columns[1:]) # label column is no input



# Count of possible values for our hidden layers

n_hidden_1 = 256 # possible pixel values (8-bit colors)

n_hidden_2 = 256
# Now create our variables with randomization

weights = {

    'w1': tf.Variable( tf.random_normal( [n_input, n_hidden_1] )),

    'w2': tf.Variable( tf.random_normal( [n_hidden_1, n_hidden_2] )),

    'out': tf.Variable( tf.random_normal( [n_hidden_2, n_classes] ))

}



biases = {

    'b1': tf.Variable( tf.random_normal( [n_hidden_1] )),

    'b2': tf.Variable( tf.random_normal( [n_hidden_2] )),

    'out': tf.Variable( tf.random_normal( [n_classes] ))

}



X = tf.placeholder('float', shape=[None, n_input])

y = tf.placeholder('float', shape=[None, n_classes])
def multi_layer_perceptron(data_input, weights, biases):

    

    # LAYER 1 with some more explanations:

    

    # INPUTS * WEIGHTS + BIASES

    layer_1 = tf.add(tf.matmul(data_input, weights['w1']), biases['b1'])

    

    # Activation using RELU, which is just f(x) = max(0, x)

    layer_1 = tf.nn.relu(layer_1)

    

    # Layer 2

    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])

    layer_2 = tf.nn.relu(layer_2)

    

    # Output Layer

    layer_out = tf.matmul(layer_2, weights['out']) + biases['out']

    

    return layer_out
# Construct the optimizer, which will essentially tune weights in our perceptron



# Instance of our model with variables and placeholders

pred = multi_layer_perceptron(X, weights, biases)



# cost function, we use a mean over the cross entropy

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( pred, y ))



# our final optimizer. AdamOptimizer is quite new. More here: https://arxiv.org/pdf/1412.6980v8.pdf

optimizer = tf.train.AdamOptimizer( learning_rate=learning_rate ).minimize(cost)
# Create an interactive session, which works best for this notebook

sess = tf.InteractiveSession()
# initialize our variables inside the session

sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):

    

    # optional: we'll print out the cost at each epoch

    avg_cost = 0.

    

    # number of batches we need to run to iterate over all samples

    batch_total = int( n_samples / batch_size )

    

    batch_idx = 0

    

    for idx in range(batch_total):

        

        # get next batch, assigning batch index to new incremented value

        batch_idx, batch = next_batch(batch_size, batch_idx, train_data_split)

        

        # use tuple unpacking to get the data and labels seperately

        batch_X, batch_y = batch

        

        # run our session with our defined optimizer & cost tensors, as well as data & labels

        _, local_cost = sess.run( [optimizer, cost], feed_dict={ X: batch_X, y: batch_y })

        

        avg_cost += (local_cost / batch_total)

        

    print("{}. epoch with cost of {}".format(epoch + 1, avg_cost))

    

print("Training Complete!")
# transform dataframe to needed array representation again

# TODO should be combined with the same process done in the next_batch function



test_data = test_data_split[test_data_split.columns[1:]]

test_label = test_data_split[test_data_split.columns[:1]]



one_hot_test_label_prior = test_label.as_matrix().flatten() # will give us a flat array of labels



test_count = len(one_hot_test_label_prior)

one_hot_test_label_final = np.zeros((test_count, 10))

one_hot_test_label_final[np.arange(test_count), one_hot_test_label_prior] = 1
correct_preds = tf.equal( tf.argmax(pred, 1), tf.argmax(y, 1) )



# will give us boolean type, we need numerical to calculate accuracy using mean

correct_preds = tf.cast(correct_preds, 'float')



# create our accuracy tensor

accuracy = tf.reduce_mean(correct_preds)



# evaluate accuracy using our test split data

accuracy.eval( feed_dict={ X: test_data, y: one_hot_test_label_final } )
# First we need our test data

test_data = pd.read_csv("../input/test.csv")

test_data.head()
# get the label-index with highest probability (in our case index = predicted number)

maximum_probability = tf.argmax(y,1)



# actually predict our labels using the test dataset

predicted_lables = pred.eval( feed_dict={ X: test_data.as_matrix() } )



# then transform those predictions into our final array of predicted labels

predicted_labels = maximum_probability.eval( feed_dict={ y: predicted_lables })
# our final predicted labels array looks good

predicted_labels
# save results

# we need to have an additional incremental id column for the submission file

result_array = np.c_[ range(1,len(test_data)+1), predicted_labels]



np.savetxt('submission.csv', result_array, delimiter=',', 

           header='ImageId,Label', comments = '', fmt='%d')
result_array