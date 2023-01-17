# I started with the following notebook and made deletions and additions as necessary:
# https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn
from __future__ import print_function

import numpy as np
import pandas as pd
#from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler

#%matplotlib inline
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

import tensorflow as tf

import time

now = time.time()
tag = str(now)

#
DEBUG = True
# settings
LEARNING_RATE = 5e-6
TRAINING_ITERATIONS = 20000

# sizes of the hidden layers
NN_HL_ARCH = [100, 100, 100, 50]  
    
# probability of keeping a neuron = 1-prob(dropout)
DROPOUT = 1.0

#
BATCH_SIZE = 1 # there may be some errors if set to > 1. 

# set to 0 to train on all available data
# (currently validation is not used)
VALIDATION_SIZE = 0 #2000

# read training data from CSV file
if DEBUG:
    print('reading CSV input...')
data = pd.read_csv('../input/train.csv')
headers = list(data)
headertext = ','.join(headers)



def normalize(X,has_outliers=True,has_missing=True,is_sparse=True):
    robust_scaler = RobustScaler(with_centering=True)
    data = robust_scaler.fit_transform(X)
    if has_outliers:
        pass
        #data = preprocessing.scale(X,with_centering=False)
    return data
    #print('data({0[0]},{0[1]})'.format(data.shape))
    #print (data.head())
inputs = data.iloc[:,1:].values
inputs = inputs.astype(np.float)

if DEBUG:
    print('finished reading data...')

# convert from [0:255] => [0.0:1.0]
# inputs = np.multiply(inputs, 1.0 / 255.0)

if DEBUG:
    print('normalizing data...')
inputs = normalize(inputs)


# split data into training & validation
# for an autoencoder labels are the same as inputs
if DEBUG:
    print('performing train/validation split...')
validation_inputs = inputs[:VALIDATION_SIZE]
validation_labels = inputs[:VALIDATION_SIZE]

train_inputs = inputs[VALIDATION_SIZE:]
train_labels = inputs[VALIDATION_SIZE:]

input_size = len(inputs[0])

if DEBUG:
    print('finished performing train/validation split...')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

if DEBUG:
    print('creating placeholders for input and output...')


# inputs
x = tf.placeholder('float', shape=[None,input_size])
# outputs = labels
y_ = tf.placeholder('float', shape=[None,input_size])

# To prevent overfitting, we  apply [dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout) before the readout layer.
# 
# Dropout removes some nodes from the network at each training stage. Each of the nodes is either kept in the network with probability *keep_prob* or dropped with probability *1 - keep_prob*. After the training stage is over the nodes are returned to the NN with their original weights.
keep_prob = tf.placeholder('float')
if DEBUG:
    print('creating W,b,h variables for various layers...')


W = []
b = []
h = []
h_tmp = []

if DEBUG:
    print('\tnow creating input layer...')
    print('\t\t',input_size,',',NN_HL_ARCH[0])

W.append(weight_variable([input_size,NN_HL_ARCH[0]]))
b.append(bias_variable([NN_HL_ARCH[0]]))
h.append(tf.nn.relu(tf.matmul(x,W[0]) + b[0]))

for i in range(1,len(NN_HL_ARCH)):
    u = NN_HL_ARCH[i-1]
    v = NN_HL_ARCH[i]
    if DEBUG:
        print('\tnow creating layer:', i)
        print('\t\t',u,',',v)

    W.append(weight_variable([u,v]))
    b.append(bias_variable([v]))
    if DEBUG:
        pass
        #print('h', tf.shape(h[i-1]), ', W:', tf.shape(W[i]), ', b:', tf.shape(b[i]))

    h_tmp.append(tf.nn.relu(tf.matmul(h[i-1],W[i]) + b[i]))
    # dropout
    h.append(tf.nn.dropout(h_tmp[-1], keep_prob))

if DEBUG:
    print('\tnow creating output layer...')
    print('\t\t',NN_HL_ARCH[-1],',',input_size)

W.append(weight_variable([NN_HL_ARCH[-1],input_size]))
b.append(bias_variable([input_size]))
if DEBUG:
    print('\tsetting up output vector...')
y = tf.nn.relu(tf.matmul(h[-1],W[-1]) + b[-1])

# 
# ADAM optimiser is a gradient based optimization algorithm, based on adaptive estimates, it's more sophisticated than steepest gradient descent and is well suited for problems with large data or many parameters.
# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
if DEBUG:
    print('defining objective function...')
# but we shall use rmse
rmse = tf.sqrt(tf.reduce_mean(tf.pow(y-y_, 2)))

if DEBUG:
    print('defining optimization step...')
# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(rmse)

# evaluation
if DEBUG:
    print('defining optimization step...')

# CHECK
error_sq_vector = tf.pow(y - y_,2)

# CHECK
accuracy = tf.sqrt(tf.reduce_mean(error_sq_vector))
predict = tf.identity(y)


# *Finally neural network structure is defined and TensorFlow graph is ready for training.*
# ## Train, validate and predict
# #### Helper functions
# 
# Ideally, we should use all data for every step of the training, but that's expensive. So, instead, we use small "batches" of random data. 
# 
# This method is called [stochastic training](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). It is cheaper, faster and gives much of the same result.
epochs_completed = 0
index_in_epoch = 0
num_examples = train_inputs.shape[0]

# serve data by batches
def next_batch(batch_size):
    
    global train_inputs
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_inputs = train_inputs[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_inputs[start:end], train_labels[start:end]
# Now when all operations for every variable are defined in TensorFlow graph all computations will be performed outside Python environment.
# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)
# Each step of the loop, we get a "batch" of data points from the training set and feed it to the graph to replace the placeholders.  In this case, it's:  *x, y* and *dropout.*
# 
# Also, once in a while, we check training accuracy on an upcoming "batch".
# 
# On the local environment, we recommend [saving training progress](https://www.tensorflow.org/versions/master/api_docs/python/state_ops.html#Saver), so it can be recovered for further training, debugging or evaluation.
# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=1

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_inputs[0:BATCH_SIZE], 
                                                            y_: validation_labels[0:BATCH_SIZE], 
                                                            keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        if i%(display_step*10) == 0 and i:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})
# After training is done, it's good to check accuracy on data that wasn't used in training.
# check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_inputs, 
                                                   y_: validation_labels, 
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.1, ymin = 0.7)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    plt.show()
# When, we're happy with the outcome, we read test data from *test.csv* and predict labels for provided inputs.
# 
# Test data contains only inputs and labels are missing. Otherwise, the structure is similar to training data.
# 
# Predicted labels are stored into CSV file for future submission.
# read test data from CSV file 
test_inputs = pd.read_csv('../input/test.csv').values
test_inputs = test_inputs.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_inputs = normalize(test_inputs) #np.multiply(test_inputs, 1.0 / 255.0)

#print('test_inputs({0[0]},{0[1]})'.format(test_inputs.shape))


# predict test set
output_rows = predict.eval(feed_dict={x: train_inputs, keep_prob: 1.0})
test_rows = predict.eval(feed_dict={x: test_inputs, keep_prob: 1.0})
'''
# using batches is more resource efficient
predicted_lables = np.zeros(test_inputs.shape[0])
for i in range(0,test_inputs.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_inputs[i*BATCH_SIZE : (i+1)*BATCH_SIZE], keep_prob: 1.0})
'''

np.savetxt('normalized_input_'+tag+'.csv', 
           train_inputs,
           delimiter=',', 
           header = headertext,
           comments = '', 
           fmt='%f')

np.savetxt('normalized_testset_'+tag+'.csv', 
           test_inputs,
           delimiter=',', 
           header = headertext,
           comments = '', 
           fmt='%f')

# save results
np.savetxt('autoencoded_input_'+tag+'.csv', 
           output_rows, 
           delimiter=',', 
           header = headertext,
           comments = '', 
           fmt='%f')

np.savetxt('testset_autoencoded_'+tag+'.csv', 
           test_rows, 
           delimiter=',', 
           header = headertext,
           comments = '', 
           fmt='%f')


diff = np.array(train_inputs) - np.array(output_rows)

np.savetxt('diff_'+tag+'.csv', 
           diff, 
           delimiter=',', 
           header = headertext,
           comments = '', 
           fmt='%f')



# ## Appendix
# As it was mentioned before, it is good to output some variables for a better understanding of the process. 
# 
# Here we pull an output of the first convolution layer from TensorFlow graph. 32 features are transformed into an image grid, and it's quite interesting to see how filters picked by NN outline characteristics of different digits.


#layer1_grid = layer1.eval(feed_dict={x: test_inputs[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1], keep_prob: 1.0})
#plt.axis('off')
#plt.imshow(layer1_grid[0], cmap=cm.seismic )

saver = tf.train.Saver()
# Save the variables to disk.
save_path = saver.save(sess, "./model_"+tag+".ckpt")
print("Model saved in file: %s" % save_path)


for w_,b_ in zip(W,b):
    print("W:")
    print(w_.eval())
    print("b:")
    print(b_.eval())

sess.close()