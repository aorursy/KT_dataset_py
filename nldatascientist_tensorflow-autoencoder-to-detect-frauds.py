# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/creditcard.csv")

df.describe()
# reshuffle the data

df=df.sample(frac=1).reset_index(drop=True)
fraud_indices = np.array(df[df.Class == 1].index)

number_records_fraud = len(fraud_indices)



# Picking the indices of the normal classes

normal_indices = np.array(df[df.Class == 0].index)

number_records_normal = len(normal_indices)



trainingratio = 0.7

training_n_normal = round(number_records_normal*trainingratio)

training_n_fraude = round(number_records_fraud*trainingratio)



# Select the fraud cases trainingset

random_fraud_indices = np.random.choice(fraud_indices, training_n_fraude, replace = False)

random_fraud_indices = np.array(random_fraud_indices)



# Out of the fraud indices pick training_n_normal cases with replacement to oversample

duplicated_fraud_indices = np.random.choice(random_fraud_indices, training_n_normal, replace = True)

duplicated_fraud_indices = np.array(duplicated_fraud_indices)



# Select random the training normal cases without replacement

random_normal_indices = np.random.choice(normal_indices, training_n_normal, replace = False)

random_normal_indices = np.array(random_normal_indices)



# Appending the 2 indices

sample_indices = np.concatenate([random_normal_indices,duplicated_fraud_indices])



# Sample dataset

sample_data = df.iloc[sample_indices,:]

test_data = df.drop(sample_indices,axis=0)



# sort on Class for the scatter plots at the end, to make sure that Frauds are drawn last

test_data=test_data.sort_values(['Class'], ascending=[True])



#shuffle the data, because the frauds where added to the tail

sample_data=sample_data.sample(frac=1).reset_index(drop=True)



print("Normal transactions:                     ", number_records_normal)

print("Fraud  transactions:                     ", number_records_fraud)

print("Fraud  transactions for training:        ", len(random_fraud_indices))



print("Selected normal transactions:            ", len(random_normal_indices))

print("Selected oversampled fraud transactions: ",  len(duplicated_fraud_indices))



print("Fraud  transactions selected for test:   ", len(test_data[test_data.Class == 1]))



print("Normal transactions selected for test:   ", len(test_data[test_data.Class == 0]))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(df.drop(['Class','Time'],axis=1))



scaled_data = scaler.transform(sample_data.drop(['Class','Time'],axis=1))

scaled_test_data = scaler.transform(test_data.drop(['Class','Time'],axis=1))

print("Size training data: ", len(scaled_data))

print("Size test data:     ", len(scaled_test_data))
import tensorflow as tf



num_inputs = len(scaled_data[1])

num_hidden = 2  

num_outputs = num_inputs 



learning_rate = 0.001

keep_prob = 0.5

tf.reset_default_graph() 
# placeholder X

X = tf.placeholder(tf.float32, shape=[None, num_inputs])



# weights

initializer = tf.variance_scaling_initializer()

w = tf.Variable(initializer([num_inputs, num_hidden]), dtype=tf.float32)

w_out = tf.Variable(initializer([num_hidden, num_outputs]), dtype=tf.float32)



# bias

b = tf.Variable(tf.zeros(num_hidden))

b_out = tf.Variable(tf.zeros(num_outputs))



#activation

act_func = tf.nn.tanh



# layers

hidden_layer = act_func(tf.matmul(X, w) + b)

dropout_layer= tf.nn.dropout(hidden_layer,keep_prob=keep_prob)

output_layer = tf.matmul(dropout_layer, w_out) + b_out
loss = tf.reduce_mean(tf.abs(output_layer - X))

optimizer = tf.train.AdamOptimizer(learning_rate)

train  = optimizer.minimize( loss)

init = tf.global_variables_initializer()



def next_batch(x_data,batch_size):

    

    rindx = np.random.choice(x_data.shape[0], batch_size, replace=False)

    x_batch = x_data[rindx,:]

    return x_batch
num_steps = 10

batch_size = 150

num_batches = len(scaled_data) // batch_size



with tf.Session() as sess:

    sess.run(init)

    for step in range(num_steps):        

        for iteration in range(num_batches):

            X_batch = next_batch(scaled_data,batch_size)

            sess.run(train,feed_dict={X: X_batch})

        

        if step % 1 == 0:

            err = loss.eval(feed_dict={X: scaled_data})

            print(step, "\tLoss:", err)

            output_2d = hidden_layer.eval(feed_dict={X: scaled_data})

    

    output_2d_test = hidden_layer.eval(feed_dict={X: scaled_test_data})
plt.figure(figsize=(20,8))

plt.scatter(output_2d[:,0],output_2d[:,1],c=sample_data['Class'],alpha=0.7)
plt.figure(figsize=(20,8))

plt.scatter(output_2d_test[:,0],output_2d_test[:,1],c=test_data['Class'],alpha=1)