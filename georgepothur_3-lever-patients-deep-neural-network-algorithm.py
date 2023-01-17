# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# To use the Adam optimizer, we should train and validate our data in batches. For batching we create this class.
import numpy as np
# Class for loading the datasets *.npz files and do the batching for the algorithm
# This code is reusable. You should just change Liver_disease_data everywhere in the code
class Liver_Disease_Data_Reader():
    # dataset is a mandatory arugment, while the batch_size is optional. dataset values can be 'train', 'valuatoin' or 'test'
    # If you don't input batch_size, it will automatically take the value: None
    def __init__(self, dataset, batch_size = None):
    
        # The dataset that loads is one of "train", "validation", "test".
        # e.g. if I call this class with x('train',5), it will load 'Liver_disease_data_train.npz' with a batch size of 5.
        npz = np.load('../input/Liver_disease_data_{0}.npz'.format(dataset))
        
        # Two variables that take the values of the inputs and the targets. Inputs are floats, targets are integers
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
        
        # Counts the batch number, given the size you feed it later
        # If the batch size is None, we are either validating or testing, so we want to take the data in a single batch
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size
    
    # A method which loads the next batch
    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()
            
        # You slice the dataset in batches and then the "next" function loads them one after the other
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch += 1
        
        # One-hot encode the targets. In this example it's a bit superfluous since we have a 0/1 column 
        # as a target already. But this will be useful for any classification task with more than one target column
        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1
        
        # The function will return the inputs batch and the one-hot encoded targets
        return inputs_batch, targets_one_hot
    
        
    # A method needed for iterating over the batches, as we will put them in a loop
    # This tells Python that the class we're defining is iterable, i.e. that we can use it like:
    # for input, output in data: 
        # do things
    # An iterator in Python is a class with a method __next__ that defines exactly how to iterate through its objects
    def __iter__(self):
        return self
import tensorflow as tf

input_size = 10        # Input size depends on the number of input variables. We have 10 of them
output_size = 2        # Output size is 2, as we one-hot encoded the targets.
hidden_layer_size = 20 # width of hidden layer

# Reset the default graph, so you can fiddle with the hyperparameters and then rerun the code.
tf.reset_default_graph()

######--- NN Building block 1.DATA (Placeholders for data) ---######

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

######--- NN Building block 2.Layer (Model + Activation function) ---######

# Outline the model. We will create a net with 2 hidden layers

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])

outputs_2 = tf.nn.sigmoid(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, output_size])
biases_3 = tf.get_variable("biases_3", [output_size])

outputs = tf.matmul(outputs_2, weights_3) + biases_3

######--- NN Building block 3.Objective function ---######

# Use of objective function(softmax_cross_entropy_with_logits) since this is a classification problem 
loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)

# Get a 0 or 1 for every input indicating whether it output the correct answer
out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

######--- NN Building block 4.Optimization Algorithm ---######

optimize = tf.train.AdamOptimizer(learning_rate=0.002).minimize(mean_loss)

### Please note, the above NN building blocks will run and learn only when you explicitly call them in a session ###

# Create a session
sess = tf.InteractiveSession()

# Initialize the variables
initializer = tf.global_variables_initializer()
sess.run(initializer)

# Choose the batch size
batch_size = 20

# Set early stopping mechanisms
max_epochs = 100
prev_validation_loss = 9999999.

# Load the first batch of training and validation, using the class we created. 
# Arguments are ending of 'Liver_disease_data_<...>', where for <...> we input 'train', 'validation', or 'test'
# depending on what we want to load
train_data = Liver_Disease_Data_Reader('train', batch_size)
validation_data = Liver_Disease_Data_Reader('validation')

# Create the loop for epochs 
for epoch_counter in range(max_epochs):
    
    # Set the epoch loss to 0, and make it a float
    curr_epoch_loss = 0.
    
    # Iterate over the training data 
    # Since train_data is an instance of the Liver_Disease_Data_Reader class,
    # we can iterate through it by implicitly using the __next__ method we defined above.
    # As a reminder, it batches samples together, one-hot encodes the targets, and returns
    # inputs and targets batch by batch
    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimize, mean_loss], 
            feed_dict={inputs: input_batch, targets: target_batch})
        
        #Record the batch loss into the current epoch loss
        curr_epoch_loss += batch_loss
    
    # Find the mean curr_epoch_loss
    # batch_count is a variable, defined in the Liver_Disease_Data_Reader class
    curr_epoch_loss /= train_data.batch_count
    
    # Set validation loss and accuracy for the epoch to zero
    validation_loss = 0.
    validation_accuracy = 0.
    
    # Use the same logic of the code to forward propagate the validation set
    # There will be a single batch, as the class was created in this way
    for input_batch, target_batch in validation_data:
        validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
            feed_dict={inputs: input_batch, targets: target_batch})
    
    # Print statistics for the current epoch
    print('Epoch '+str(epoch_counter+1)+
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')
    
    # Trigger early stopping if validation loss begins increasing.
    if validation_loss > prev_validation_loss:
        break
        
    # Store this epoch's validation loss to be used as previous in the next iteration.
    prev_validation_loss = validation_loss
    
print('End of training.')
# Load the test data, following the same logic as we did for the train_data and validation data
test_data = Liver_Disease_Data_Reader('test')

# Forward propagate through the training set. This time we only need the accuracy
for inputs_batch, targets_batch in test_data:
    test_accuracy = sess.run([accuracy],
                     feed_dict={inputs: inputs_batch, targets: targets_batch})

# Get the test accuracy in percentages
# When sess.run is has a single output, we get a list (that's how it was coded by Google), rather than a float.
# Therefore, we must take the first value from the list (the value at position 0)
test_accuracy_percent = test_accuracy[0] * 100.

# Print the test accuracy
print('Test accuracy: '+'{0:.2f}'.format(test_accuracy_percent)+'%')
inputs_batch.shape
targets_batch.shape
