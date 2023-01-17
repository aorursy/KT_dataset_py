# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
raw_train = pd.read_csv('../input/train.csv')

raw_test = pd.read_csv('../input/test.csv')
def to_one_hot(label):

    base = np.zeros([label.shape[0],10])

    base[np.arange(label.shape[0]),label[:,0].tolist()] = 1

    return base
train_255 = raw_train.iloc[:,1:].values

raw_label = raw_train.iloc[:,0].values.reshape([train_255.shape[0],1])

label = to_one_hot(raw_label)

train = train_255.astype(np.float)/255.0

print(train.shape)

print(label.shape)

test_255 = raw_test.values

test = test_255.astype(np.float)/255.0

print(test.shape)
def next_batch(num, train, label):

    idx = np.arange(0 , len(train))

    np.random.shuffle(idx)

    idx=idx[:2]

    data_shuffle = train[idx,:]

    label_shuffle = label[idx,:]

    return data_shuffle, label_shuffle

# Parameters

learning_rate = 0.01

training_epochs = 200

batch_size = 128

display_step = 1



# tf Graph Input

x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784

y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes



# Set model weights

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))



# Construct model

pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax



# Minimize error using cross entropy

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)



# Initialize the variables (i.e. assign their default value)

init = tf.global_variables_initializer()
# Start training

with tf.Session() as sess:



    # Run the initializer

    sess.run(init)



    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        total_batch = int(train.shape[0]/batch_size)

        # Loop over all batches

        for i in range(total_batch):

            batch_xs, batch_ys = next_batch(batch_size,train,label)

            # Run optimization op (backprop) and cost op (to get loss value)

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,

                                                          y: batch_ys})

            # Compute average loss

            avg_cost += c / total_batch

        # Display logs per epoch step

        if (epoch+1) % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))



    print("Optimization Finished!")



    # eval test data

    test_prediction = tf.argmax(pred, 1)

    result = test_prediction.eval({x: test})
dtype = [('ImageId','int32'), ('Label','int32')]

values = np.zeros(test.shape[0], dtype=dtype)

index = np.arange(1,test.shape[0]+1)

df = pd.DataFrame(values, index=index)

df['ImageId'] = index

df['Label'] = result

df.to_csv('./submission_logistic_regression.csv', sep=',', encoding='utf-8', index=False)