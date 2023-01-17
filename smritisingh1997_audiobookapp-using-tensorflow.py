import numpy as np

import pandas as pd

from sklearn import preprocessing

import tensorflow as tf



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_csv_data = pd.read_csv('/kaggle/input/audiobook-app-data/audiobook_data_2.csv')

raw_csv_data.head()
raw_csv_data.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
raw_csv_data = raw_csv_data.values
type(raw_csv_data)
raw_csv_data
# The inputs are all columns in the csv, except for the first one [:,0]

# (which is just the arbitrary customer IDs that bear no useful information),

# and the last one [:,-1] (which is our targets)

unscaled_inputs_all = raw_csv_data[:, 1:-1]



# The targets are in the last column. That's how datasets are conventionally organized.

targets_all = raw_csv_data[:, -1]
# Count how many targets are 1 (meaning that the customer did convert)

num_one_targets = int(np.sum(targets_all))



# Set a counter for targets that are 0 (meaning that the customer did not convert)# Set a counter for targets that are 0 (meaning that the customer did not convert)

zero_targets_counter = 0



# We want to create a "balanced" dataset, so we will have to remove some input/target pairs.

# Declare a variable that will do that:

indices_to_remove = []



# Count the number of targets that are 0. 

# Once there are as many 0s as 1s, mark entries where the target is 0.

for i in range(targets_all.shape[0]):

    if(targets_all[i] == 0):

        zero_targets_counter += 1

        if(zero_targets_counter > num_one_targets):

            indices_to_remove.append(i)



# Create two new variables, one that will contain the inputs, and one that will contain the targets.

# We delete all indices that we marked "to remove" in the loop above.

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)

targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
# When the data was collected it was actually arranged by date

# Shuffle the indices of the data, so the data is not arranged in any way when we feed it.

# Since we will be batching, we want the data to be as randomly spread out as possible

shuffled_indices = np.arange(scaled_inputs.shape[0])

np.random.shuffle(shuffled_indices)



# Use the shuffled indices to shuffle the inputs and targets.

shuffled_inputs = scaled_inputs[shuffled_indices]

shuffled_targets = targets_equal_priors[shuffled_indices]
#count the total number of samples

samples_count = shuffled_inputs.shape[0]



# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.

# Naturally, the numbers are integers.

train_samples_count = int(0.8*samples_count)

validation_samples_count = int(0.1*samples_count)



# The 'test' dataset contains all remaining data.

test_samples_count = samples_count - (train_samples_count + validation_samples_count)



# Create variables that record the inputs and targets for training

# In our shuffled dataset, they are the first "train_samples_count" observations

train_inputs = shuffled_inputs[:train_samples_count]

train_targets = shuffled_targets[:train_samples_count]



# Create variables that record the inputs and targets for validation.

# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned

validation_inputs = shuffled_inputs[train_samples_count:(train_samples_count+validation_samples_count)]

validation_targets = shuffled_targets[train_samples_count:(train_samples_count+validation_samples_count)]



# Create variables that record the inputs and targets for test.

# They are everything that is remaining.

test_inputs = shuffled_inputs[(train_samples_count+validation_samples_count):]

test_targets = shuffled_targets[(train_samples_count+validation_samples_count):]
train_samples_count, validation_samples_count, test_samples_count
# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)

print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)

print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
# Save the three datasets in *.npz.

# In the next lesson, you will see that it is extremely valuable to name them in such a coherent way!

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)

np.savez('Audiobooks_data_validation', inputs=validation_inputs, targets=validation_targets)

np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)
# let's create a temporary variable npz, where we will store each of the three Audiobooks datasets

npz = np.load('Audiobooks_data_train.npz')



# we extract the inputs using the keyword under which we saved them

# to ensure that they are all floats, let's also take care of that

train_inputs = npz['inputs'].astype(np.float)

# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)

train_tragets = npz['targets'].astype(np.int)



# we load the validation data in the temporary variable

npz = np.load('Audiobooks_data_validation.npz')

# we can load the inputs and the targets in the same line

validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)



# we load the test data in the temporary variable

npz = np.load('Audiobooks_data_test.npz')

# we create 2 variables that will contain the test inputs and the test targets

test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
# # Following is the code, in case one wants to see what is the content of npz

# lst = npz.files

# for item in lst:

#     print(item)

#     print(npz[item])
# Set the input and output sizes

input_size = 10

output_size = 2

# Use same hidden layer size for both hidden layers. Not a necessity.

hidden_layer_size = 50



# define how the model will look like

model = tf.keras.Sequential([

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),

    tf.keras.layers.Dense(output_size, activation='softmax')

])



# Choose the optimizer and the loss function

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Training : That's where we train the model we have built.



# set the batch size

batch_size = 100



# set a maximum number of training epochs

max_epochs = 100



# set an early stopping mechanism

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)





# fit the model

model.fit(

         train_inputs, 

         train_targets,

         batch_size=batch_size,

         epochs=max_epochs,

         callbacks=[early_stopping],

         validation_data=(validation_inputs, validation_targets),

         verbose=2

         )
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)