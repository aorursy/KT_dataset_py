# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Processing to split train.csv to training and validation datasets; and to create npz files for processing 

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_inputs_all = train_data.iloc[:,1:]

train_targets_all = train_data.iloc[:,0:1] 

train_targets_all = pd.get_dummies(train_targets_all.label)



sample_count = train_data.iloc[:,1:].shape[0]



train_sample_count = int(0.9*sample_count)

validation_sample_count = sample_count - train_sample_count



train_inputs = train_inputs_all.iloc[:train_sample_count]

train_targets = train_targets_all.iloc[:train_sample_count]



validation_inputs = train_inputs_all.iloc[train_sample_count:]

validation_targets = train_targets_all.iloc[train_sample_count:]





# Any results you write to the current directory are saved as output.
input_size = 784

output_size = 10

# Use same hidden layer size for all hidden layers

hidden_layer_size = 1000



# Reset any variables left in memory from previous runs.

tf.reset_default_graph()



# Declare placeholders where the data will be fed into.

inputs = tf.placeholder(tf.float32, [None, input_size])

targets = tf.placeholder(tf.float32, [None, output_size])



# Weights and biases for the first linear combination between the inputs and the first hidden layer.

# Use get_variable in order to make use of the default TensorFlow initializer which is Xavier.

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])

biases_1 = tf.get_variable("biases_1", [hidden_layer_size])



# Operation between the inputs and the first hidden layer.

# ReLu is the first activation function. 

outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)



# Weights and biases for the second linear combination.

# This is between the first and second hidden layers.

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])

biases_2 = tf.get_variable("biases_2", [hidden_layer_size])



# Operation between the first and the second hidden layers using sigmoid

outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)



# Weights and biases for the third linear combination.

# This is between second and third hidden layers.

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, hidden_layer_size])

biases_3 = tf.get_variable("biases_3", [hidden_layer_size])



# Operation between the second and third hidden layers

outputs_3 = tf.nn.sigmoid(tf.matmul(outputs_2, weights_3) + biases_3)



# Weights and biases for the fourth linear combination.

# This is between the third and fourth hidden layers.

weights_4 = tf.get_variable("weights_4", [hidden_layer_size, hidden_layer_size])

biases_4 = tf.get_variable("biases_4", [hidden_layer_size])



# Operation between the third and fourth hidden layers.

outputs_4 = tf.nn.sigmoid(tf.matmul(outputs_3, weights_4) + biases_4)



# Weights and biases for the fifth linear combination.

# This is between the fourth and fifth hidden layers.

weights_5 = tf.get_variable("weights_5", [hidden_layer_size, hidden_layer_size])

biases_5 = tf.get_variable("biases_5", [hidden_layer_size])



# Operation between the fourth and fifth hidden layers.

outputs_5 = tf.nn.sigmoid(tf.matmul(outputs_4, weights_5) + biases_5)



# Weights and biases for the final linear combination.

# That's between the fifth hidden layer and the output layer.

weights_6 = tf.get_variable("weights_6", [hidden_layer_size, output_size])

biases_6 = tf.get_variable("biases_6", [output_size])



# Operation between the fifth hidden layer and the final output.

# Haven't used an activation function because it will be included directly in 

# the loss function. This works for softmax and sigmoid with cross entropy.

outputs = tf.matmul(outputs_5, weights_6) + biases_6



# Calculate the loss function for every output/target pair.

# The function used is the same as applying softmax to the last layer and then calculating cross entropy

# Logits here means: unscaled probabilities (so, the outputs, before they are scaled by the softmax)

# Naturally, the labels are the targets.

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)



# Get the average loss

mean_loss = tf.reduce_mean(loss)



# Define the optimization step. Using adaptive optimizers such as Adam in TensorFlow

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)



# Get a 0 or 1 for every input in the batch indicating whether it output the correct answer out of the 10.

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))



# Get the average accuracy of the outputs.

accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))



# Declare the session variable.

sess = tf.InteractiveSession()



# Initialize the variables. Default initializer is Xavier.

initializer = tf.global_variables_initializer()

sess.run(initializer)



# Batching

batch_size = 1000



# Calculate the number of batches per epoch for the training set.

batches_number = train_inputs.shape[0] // batch_size



# Basic early stopping. Set a miximum number of epochs.

max_epochs = 15



# Keep track of the validation loss of the previous epoch.

# If the validation loss becomes increasing, we want to trigger early stopping.

# We initially set it at some arbitrarily high number to make sure we don't trigger it

# at the first epoch

prev_validation_loss = 9999999.



import time

start_time = time.time()



# Create a loop for the epochs. Epoch_counter is a variable which automatically starts from 0.

for epoch_counter in range(max_epochs):

    

    # Keep track of the sum of batch losses in the epoch.

    curr_epoch_loss = 0.

    

    # Iterate over the batches in this epoch.

    for batch_counter in range(batches_number):

        

        # Input batch and target batch are assigned values from the train dataset, given a batch size

        input_batch = train_inputs.iloc[batch_counter*batch_size:batch_counter*batch_size+batch_size,:]

        target_batch = train_targets.iloc[batch_counter*batch_size:batch_counter*batch_size+batch_size,:]

        

        # Run the optimization step and get the mean loss for this batch.

        # Feed it with the inputs and the targets we just got from the train dataset

        _, batch_loss = sess.run([optimize, mean_loss], feed_dict={inputs: input_batch, targets: target_batch})

        

        # Increment the sum of batch losses.

        curr_epoch_loss += batch_loss

    

    # So far curr_epoch_loss contained the sum of all batches inside the epoch

    # We want to find the average batch losses over the whole epoch

    # The average batch loss is a good proxy for the current epoch loss

    curr_epoch_loss /= batches_number

    

    # At the end of each epoch, get the validation loss and accuracy

    # Get the input batch and the target batch from the validation dataset

    input_batch = validation_inputs.iloc[:,:]

    target_batch = validation_targets.iloc[:,:]

    

    

    # Run without the optimization step (simply forward propagate)

    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy], 

        feed_dict={inputs: input_batch, targets: target_batch})

    

    # Print statistics for the current epoch

    # Epoch counter + 1, because epoch_counter automatically starts from 0, instead of 1

    # We format the losses with 3 digits after the dot

    # We format the accuracy in percentages for easier interpretation

    print('Epoch '+str(epoch_counter+1)+

          '. Mean loss: '+'{0:.3f}'.format(curr_epoch_loss)+

          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+

          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')

    

    # Trigger early stopping if validation loss begins increasing.

    if validation_loss > prev_validation_loss:

        break

        

    # Store this epoch's validation loss to be used as previous validation loss in the next iteration.

    prev_validation_loss = validation_loss



# Not essential, but it is nice to know when the algorithm stopped working in the output section, rather than check the kernel

print('End of training.')



#Add the time it took the algorithm to train

print("Training time: %s seconds" % (time.time() - start_time))


out = sess.run([outputs], feed_dict={inputs: test_data})

out_df = np.row_stack(out)



argmax_outputs = np.argmax(out_df,1)

argmax_outputs = np.array([argmax_outputs]).T

#print(argmax_outputs)



image_id = np.arange(1, len(argmax_outputs) + 1, 1)

image_id = image_id.reshape(len(argmax_outputs), 1)



np.savetxt('submission.csv', np.c_[image_id, argmax_outputs], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
