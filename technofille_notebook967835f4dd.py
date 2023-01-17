# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#önce udemy datası üstünde train edip sonra kaggle da test edicez.

import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import matplotlib.pyplot as plt
#import mpl_toolkits
import tensorflow as tf
import tensorflow_datasets as tfds
import os
os.chdir(r"C:\\Users\user\Desktop\Kannada")
from sklearn import preprocessing

# Load the data
raw_train_csv_data = np.loadtxt('train.csv',delimiter=',',skiprows=1)
raw_test_csv_data = np.loadtxt('test.csv',delimiter=',',skiprows=1)
raw_Dig_csv_data = np.loadtxt('Dig-MNIST.csv',delimiter=',',skiprows=1)
# The inputs are all columns in the csv, except for the first one [:,0]
# (which is just the arbitrary customer IDs that bear no useful information),
# and the last one [:,-1] (which is our targets)

unscaled_inputs_train = raw_train_csv_data[:,1:]
targets_train=raw_train_csv_data[:,0]

unscaled_inputs_Dig=raw_Dig_csv_data[:,1:]
targets_Dig=raw_Dig_csv_data[:,0]

raw_test_csv_data=raw_test_csv_data[:,1:]
unscaled_inputs_submission = raw_test_csv_data

# The targets are in the last column. That's how datasets are conventionally organized.

scaled_inputs_train = preprocessing.scale(unscaled_inputs_train)
scaled_inputs_submission=preprocessing.scale(unscaled_inputs_submission)

scaled_inputs_Dig=preprocessing.scale(unscaled_inputs_Dig)

samples_count=unscaled_inputs_train.shape[0]
submission_samples_count=unscaled_inputs_submission.shape[0]



validation_samples_count = int(0.1 * samples_count)
train_samples_count = int(0.8 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count


train_inputs = scaled_inputs_train[:train_samples_count]
train_targets = targets_train[:train_samples_count]


validation_inputs = scaled_inputs_train[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = targets_train[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = scaled_inputs_train[train_samples_count+validation_samples_count:]
test_targets = targets_train[train_samples_count+validation_samples_count:]

#print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
#print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
#print(np.sum(targets_test), test_samples_count, np.sum(targets_test) / test_samples_count)

np.savez('mnist_data_train', inputs=train_inputs, targets=train_targets)
np.savez('mnist_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('mnist_data_test', inputs=test_inputs, targets=test_targets)

npz = np.load('mnist_data_train.npz')

# we extract the inputs using the keyword under which we saved them
# to ensure that they are all floats, let's also take care of that
train_inputs = npz['inputs'].astype(np.float)
# targets must be int because of sparse_categorical_crossentropy (we want to be able to smoothly one-hot encode them)
train_targets = npz['targets'].astype(np.int)

# we load the validation data in the temporary variable
npz = np.load('mnist_data_validation.npz')
# we can load the inputs and the targets in the same line
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

# we load the test data in the temporary variable
npz = np.load('mnist_data_test.npz')
# we create 2 variables that will contain the test inputs and the test targets
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)





input_size = 784 
output_size = 10

hidden_layer_size = 50
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model = tf.keras.Sequential([
    
    # the first layer (the input layer)
    # each observation is 28x28x1 pixels, therefore it is a tensor of rank 3
    # since we don't know CNNs yet, we don't know how to feed such input into our net, so we must flatten the images
    # there is a convenient method 'Flatten' that simply takes our 28x28x1 tensor and orders it into a (None,) 
    # or (28x28x1,) = (784,) vector
    # this allows us to actually create a feed forward neural network
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    #tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.BatchNormalization(),
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NUM_EPOCHS = 10


model.fit(scaled_inputs_Dig,targets_Dig, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets),callbacks=[callback], verbose =2)

test_loss, test_accuracy = model.evaluate(scaled_inputs_Dig,targets_Dig)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

dataframe=pd.DataFrame(columns=['ImageId','Label'])
dataframe['ImageId']=range(1,5001)
#submission=model.predict(unscaled_inputs_submission,batch_size=100,verbose=1)
submission=model.predict(unscaled_inputs_submission,batch_size=100,verbose=1)
error = np.mean(test_targets!=np.argmax(submission, axis = 1))
print (error)

print(np.argmax(submission, axis = 1)[:100]) #[2 5 1 9 3 7 0 3 1 3 5 7 1 0 4 5 3 1 9 0 9 1 1 5 7 4 1 7 1 7 7 5 4 1 6 2 5
 #5 1 6 7 7 4 9 5 7 1 3 6 7 6 8 1 3 8 2 1 2 2 5 4 1 7 0 0 7 1 1 0 1 6 5 1 8
 #2 5 9 9 2 3 5 1 1 0 9 1 4 3 6 7 2 0 6 6 1 4 3 9 7 1]
print(submission.shape[0])

"""
dataframe['Label'] = np.argmax(submission, axis = 1)

print (dataframe)
df=pd.DataFrame(dataframe)
df.to_csv(index=False)
compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df.to_csv('out.zip', index=False,
          compression=compression_opts)  
"""
"""
print(submission[0][:])


dataframe=pd.DataFrame(columns=['ImageId','Label'])
dataframe['ImageId']=range(1,28001)
print (dataframe)
df=np.array(dataframe)

for j in range(28000):
    max=0
    for i in range (10):
        if submission[j][i]>max:
            max=submission[j][i]
            df[j][1]=i

print (submission[10000:11100])
"""