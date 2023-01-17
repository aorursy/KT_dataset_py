# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
# Any results you write to the current directory are saved as output.
#Load the data in
df = pd.read_json('../input/train.json')
df.head()
#Examine the audio embedding data
dist_encodings = {}
dist_samples = {}
for enc_array in df['audio_embedding']:
    if len(enc_array) in dist_samples.keys():
        dist_samples[len(enc_array)]+=1
    else:
        dist_samples[len(enc_array)]=1
    for enc in enc_array:
        for v in enc:
            if v in dist_encodings.keys():
                dist_encodings[v]+=1
            else:
                dist_encodings[v]=1

print("Encodings range: {} to {}".format(min(dist_encodings.keys()), max(dist_encodings.keys())))
print("Encodings: {}".format(dist_encodings))

print("Encoding samples range: {} to {}".format(min(dist_samples.keys()), max(dist_samples.keys())))
print("Encoding samples: {}".format(dist_samples))
#Split the data into training and validation
x_train_data, x_val_data = train_test_split(df,test_size=0.1,train_size=None,random_state=34,shuffle=True)

def normalise_and_pad(sequence, max_val=255.0, max_seq_len=10):
    ret = np.pad(np.array(sequence) / max_val, ((0, max_seq_len-len(sequence)),(0,0)), 'wrap')
    return ret


def create_binary_classifier(binary_array):
    yvals = np.zeros(shape=(len(binary_array), 2), dtype='float32')
    for idx, val in enumerate(binary_array):
        if val == 1:
            yvals[idx][1] = 1
        else:
            yvals[idx][0] = 1
    return yvals
    
    

xtrain = np.asarray([normalise_and_pad(x) for x in x_train_data['audio_embedding']], dtype='float32')
ytrain = create_binary_classifier(x_train_data['is_turkey'].values)


xval = np.asarray([normalise_and_pad(x) for x in x_val_data['audio_embedding']], dtype='float32')
yval = create_binary_classifier(x_val_data['is_turkey'].values)

#Examine types and compare outputs
print("xtrain: {}; ytrain:{}, xval: {}; yval: {}".format(xtrain.shape, ytrain.shape, xval.shape, yval.shape))

print(x_train_data['is_turkey'].values[:10])
print(ytrain[:10])
def get_batches(x_train, y_train, batch_size):
    current_index=0
    while current_index+batch_size < len(x_train):
        batch_x = x_train[current_index:current_index+batch_size]
        batch_y = y_train[current_index:current_index+batch_size]
        yield (batch_x, batch_y)
        current_index += batch_size
import tensorflow as tf
import time

#Set logging and reset the graph
tf.reset_default_graph()

save_file = './model.ckpt'

# Parameters
learning_rate = 0.000005
training_epochs = 800
batch_size = 128  # Decrease batch size if you don't have enough memory
display_step = 5
keep_prob_val = 0.5

n_input = 10*128  #10*128 audio embeddings
n_classes = 2  # is not vs is turkey 

#Size of the network:
n_hidden_layer_1 = 512 # layer number of features
n_hidden_layer_2 = 256 # layer number of features
with tf.name_scope("variables_scope"):
    
    with tf.name_scope("input_variables"):
        # tf Graph input
        x = tf.placeholder("float32", [None, 10, 128], name="input_x")
        y = tf.placeholder("float32", [None, n_classes], name="targets")
        keep_prob = tf.placeholder(tf.float32) # probability to keep units

        x_flat = tf.reshape(x, [-1, n_input], name="input_x_flat")
    
    
    with tf.name_scope("weights_scope"):
        # Store layers weight & bias
        weights = {
            'hidden_layer_1': tf.Variable(tf.random_normal([n_input, n_hidden_layer_1]), name="w_hidden_1"),
            'hidden_layer_2': tf.Variable(tf.random_normal([n_hidden_layer_1, n_hidden_layer_2]), name="w_hidden_2"),
            'out': tf.Variable(tf.random_normal([n_hidden_layer_2, n_classes]), name="w_out")
        }
        
        biases = {
            'hidden_layer_1': tf.Variable(tf.random_normal([n_hidden_layer_1])),
            'hidden_layer_2': tf.Variable(tf.random_normal([n_hidden_layer_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
    
    
    with tf.name_scope("network_scope"):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer_1']),biases['hidden_layer_1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, keep_prob)

        layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer_2']),biases['hidden_layer_2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)

        # Output layer with linear activation
        logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
with tf.name_scope("training_scope"):
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y), name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, name='gradDescent')


# Calculate accuracy
with tf.name_scope("accuracy_scope"):
    argmax_logits = tf.argmax(logits, 1)
    argmax_y = tf.argmax(y, 1)
    correct_prediction = tf.equal(argmax_logits, argmax_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for batch_x, batch_y in get_batches(xtrain, ytrain, batch_size):
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_val})
        
        # Print status for every 10 epochs        
        if epoch % display_step == 0:
            valid_accuracy = sess.run(
                accuracy,
                feed_dict={
                    x: xval,
                    y: yval,
                    keep_prob: 1.0})
            print('Epoch {:<3} - Validation Accuracy: {}'.format(
                epoch,
                valid_accuracy))
            
    saver.save(sess, save_file)
#Load the test data in
df_test = pd.read_json('../input/test.json')
print(len(df_test['vid_id']))
df_test.head()
#Process the data to be ready to feed the model
xsubmission = np.asarray([normalise_and_pad(x) for x in df_test['audio_embedding']], dtype='float32')

#Examine types and compare outputs
print("xsubmission: {}".format(xsubmission.shape))

with tf.Session() as sess:
    saver.restore(sess, save_file)
    argmax_output = sess.run(
                argmax_logits,
                feed_dict={
                    x: xsubmission,
                    y: yval,
                    keep_prob: 1.0})


submit_df = pd.DataFrame(columns=['vid_id', 'is_turkey'])
submit_df['vid_id'] = df_test['vid_id']
submit_df['is_turkey'] = list(argmax_output)
print("Dataframe size: {}".format(len(submit_df['vid_id'])))
submit_df.head()
submit_df.to_csv('../submission_turkey.csv',index=None,columns=['vid_id','is_turkey'])