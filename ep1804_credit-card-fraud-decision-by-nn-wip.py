import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns



import tensorflow as tf
d1 = pd.read_csv("../input/creditcard.csv")
# Data shape

print(d1.shape)

d1.head()
# Chech NAs

d1.isnull().sum()
# Basic statistic of normal card transactions

d1[d1.Class == 0].describe()
# Basic statistic of fraud card transactions

d1[d1.Class == 1].describe()
# Histograms

def showHist(d):

    plt.figure(figsize=(12, 30 * 4))

    gs = gridspec.GridSpec(30, 1)

    for i, col in enumerate(d.loc[:, 'Time':'Amount']):

        ax = plt.subplot(gs[i])

        fraud = d[col][d.Class == 1].values

        normal = d[col][d.Class == 0].values

        ax.hist(fraud, bins = 50, alpha = 0.5, normed = True, color = 'tomato')

        ax.hist(normal, bins = 50, alpha = 0.5, normed = True, color = 'teal')

        ax.set_xlabel('')

        ax.set_title(str(col))

    plt.show()



showHist(d1)

    

# color table: https://matplotlib.org/examples/color/named_colors.html
# Correlations

corr = d1.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=5, yticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

plt.show()



# c.f. 

# As the description on data said, V1 ~ V28 are PCA reduced.

# Thus, no correlation between them.
# This whole process is a preliminary analysis. So we'll sample.

# Meanwhile, fraud data will be over-sampled for balance between classes.



d2_normal = d1[d1.Class == 0].sample(11000)

d2_normal_tr = d2_normal.iloc[:10000,:] # normal data for training

d2_normal_ob = d2_normal.iloc[10000:,:] # normal data for testing



d2_fraud = d1[d1.Class == 1].sample(492) # sampling for shuffle

d2_fraud_tr = d2_fraud.iloc[:400,:].sample(10000, replace = True) # fraud data for training

d2_fraud_ob = d2_fraud.iloc[400:,:].sample(1000, replace = True) # fraud data for testing



d2_tr = d2_normal_tr.append(d2_fraud_tr).sample(frac = 1) # training dataset

d2_ob = d2_normal_ob.append(d2_fraud_ob).sample(frac = 1) # testing dataset
# As you can see in the histogram, normal time data is periodic.

# We'll use second-in-a-day instead of absolute second

d2_tr.Time = d2_tr.Time % 86400

d2_ob.Time = d2_ob.Time % 86400
d2_tr.describe()
d2_ob.describe()
# showHist(d2_tr)
# showHist(d2_ob)
# trim dataset for tensorflow input



# convert class labels from scalars to one-hot vectors

# ref: https://www.kaggle.com/kakauandme/digit-recognizer/tensorflow-deep-nn/run/164725

# ex)

# 0 => [1 0 0 0 0 0 0 0 0 0]

# 1 => [0 1 0 0 0 0 0 0 0 0]

# ...

# 9 => [0 0 0 0 0 0 0 0 0 1]

def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return pd.DataFrame(labels_one_hot)



x_tr = d2_tr.iloc[:, :30]

x_ob = d2_ob.iloc[:, :30]



y_tr = dense_to_one_hot(d2_tr.iloc[:, 30], 2)

y_ob = dense_to_one_hot(d2_ob.iloc[:, 30], 2)
x_tr.head()
y_tr.head()
d2_tr.head()
# parameters

learning_rate = 0.01

data_size = x_tr.shape[0]

batch_size = 2000



print(data_size)
# input place holders

x = tf.placeholder(tf.float32, [None, 30])

y = tf.placeholder(tf.float32, [None, 2])

keep_prob = tf.placeholder(tf.float32)
# weights, bias, and activation function for nn layers

W1 = tf.Variable(tf.random_normal([30, 30]))

b1 = tf.Variable(tf.random_normal([30]))

a1 = tf.nn.relu(tf.matmul(x, W1) + b1)

a1_out = tf.nn.dropout(a1, keep_prob = keep_prob)



W2 = tf.Variable(tf.random_normal([30, 30]))

b2 = tf.Variable(tf.random_normal([30]))

a2 = tf.nn.relu(tf.matmul(a1_out, W2) + b2)

a2_out = tf.nn.dropout(a2, keep_prob = keep_prob)



W3 = tf.Variable(tf.random_normal([30, 30]))

b3 = tf.Variable(tf.random_normal([30]))

a3 = tf.nn.relu(tf.matmul(a2_out, W3) + b3)

a3_out = tf.nn.dropout(a3, keep_prob = keep_prob)



W4 = tf.Variable(tf.random_normal([30, 2]))

b4 = tf.Variable(tf.random_normal([2]))

z4 = tf.matmul(a3_out, W4) + b4

a4 = tf.nn.softmax(z4)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z4, labels=y))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Prediction and check accuracy

correct = tf.equal(tf.argmax(a4, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
print(a1_out.get_shape())

print(a2_out.get_shape())

print(a3_out.get_shape())

print(a4.get_shape())
# variables for next_batch function

next_batch__x = x_tr

next_batch__y = y_tr

next_batch__data_size = data_size

next_batch__batch_size = batch_size

next_batch__start = 0

next_batch__epoch_done = False

next_batch__epochs_completed = 0



# serve data by batches

def next_batch():

    

    global next_batch__x

    global next_batch__y

    global next_batch__data_size

    global next_batch__batch_size

    global next_batch__start

    global next_batch__epoch_done

    global next_batch__epochs_completed    

    

    next_batch__epoch_done = False

    

    # end of epoch

    if next_batch__start > next_batch__data_size:

        # shuffle the data

        arr = np.arange(next_batch__data_size)

        np.random.shuffle(arr)

        next_batch__x = next_batch__x.iloc[arr, :]

        next_batch__y = next_batch__y.iloc[arr, :]

        # end epoch

        next_batch__start = 0

        next_batch__epoch_done = True

        next_batch__epochs_completed += 1



    start = next_batch__start

    end = start + next_batch__batch_size

    next_batch__start = end

    return next_batch__x[start:end], next_batch__y[start:end]
# Initialize tensorflow session

sess = tf.Session()

sess.run(tf.global_variables_initializer())



# Initialize next_batch function

next_batch__start = 0

next_batch__epochs_completed = 0



# Train

while(next_batch__epochs_completed < 20):

    x_batch, y_batch = next_batch()

    sess.run(optimizer, feed_dict={x: x_batch, y: y_batch, keep_prob: 0.8})

    if (next_batch__epoch_done):

        c_b, ac_b = sess.run([cost, accuracy], feed_dict={x: x_batch, y: y_batch, keep_prob: 1.0})

        c, ac, pred = sess.run([cost, accuracy, a3], feed_dict={x: x_ob, y: y_ob, keep_prob: 1.0})

        print('Epochs: %2d, Batch cost: %f, accuracy: %f, Test cost: %s, accuracy: %f' % 

              (next_batch__epochs_completed, c_b, ac_b, c, ac))

print('Learning Finished!')
# Predict

ct, accu = sess.run([cost, accuracy], feed_dict={x: x_ob, y: y_ob, keep_prob: 1.0})

print('Test result: cost: %f, accuracy: %f' % (ct, accu))