nb_name = 'mnist-lenet-spatial-domain'
#################### Output ####################

log_path_dev = f'log-{nb_name}-dev.csv'

log_path_train = f'log-{nb_name}-train.csv'

params_path = f'params-{nb_name}.csv'

submission_path = f'submisison.csv'
import os

import pickle as pkl

import time



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf
def load_dataset(val_split = 0.25):

    np.random.seed(0)

    train_data = pd.read_csv('../input/train.csv')

    test_data = pd.read_csv("../input/test.csv")

    train_y = train_data['label'].values

    train_x = train_data.drop(columns=['label']).values

    train_x = train_x.reshape(-1,28,28)

    test_x = test_data.values

    test_x = test_x.reshape(-1,28,28)



    # Data Normalization

    mean, std = train_x.mean(), train_x.std()

    train_x = (train_x - mean)/std

    test_x = (test_x - mean)/std

    

    train_x = train_x.reshape((-1, 1, 28, 28))

    test_x = test_x.reshape((-1, 1, 28, 28))

    

    # Validation Set

    indices = np.arange(len(train_x))

    np.random.shuffle(indices)

    pivot = int(len(train_x) * (1 - val_split))

    train_x, val_x = train_x[indices[:pivot]], train_x[indices[pivot:]]

    train_y, val_y = train_y[indices[:pivot]], train_y[indices[pivot:]]

    

    return train_x, train_y, val_x, val_y, test_x
train_x, train_y, val_x, val_y, test_x = load_dataset(1 - 28/42)

train_x.shape, val_x.shape, test_x.shape
perm=[0, 3, 2, 1] # Back and forth between NHWC and NCHW

train_x, val_x, test_x = train_x.transpose(perm), val_x.transpose(perm), test_x.transpose(perm)

train_x.shape, val_x.shape, test_x.shape
def conv_layer(name, x, k, f1, f2, s=1, padding='SAME'):

    with tf.variable_scope(name):

        value = tf.truncated_normal([k, k, f1, f2], stddev=1e-1)

        w = tf.get_variable('w', initializer=value)

        conv = tf.nn.conv2d(x, w, [1, 1, s, s], padding)



        value = tf.constant(1e-1, tf.float32, [f2])

        bias = tf.get_variable('bias', initializer=value)

        out = tf.nn.relu(tf.nn.bias_add(conv, bias))

        

        tf.summary.histogram('weights', w)

        tf.summary.histogram('bias', bias)

        tf.summary.histogram('activations', out)

        

        return out



def pool_layer(name, x, stride=2, padding='SAME'):

    with tf.variable_scope(name):

        param = [1, stride, stride, 1]

        x = tf.nn.max_pool(x, param, param, padding)

        return x



def conv_net(x, out=10, is_training=True):

    k, f1, f2, h1, h2 = 5, 6, 16, 120, 84 # LeNet

    

    # Define a scope for reusing the variables

    with tf.variable_scope('ConvNet', reuse=tf.AUTO_REUSE):

        # Input Channels     

        c = x.shape.as_list()[-1] # For NHWC



        x = conv_layer('Conv1', x, k, c, f1)

        x = pool_layer('Pool1', x)

        x = conv_layer('Conv2', x, k, f1, f2)

        x = pool_layer('Pool2', x)



        x = tf.contrib.layers.flatten(x, scope='Flatten')

        x = tf.contrib.layers.fully_connected(x, h1, tf.nn.relu, scope='Dense1')

        x = tf.contrib.layers.fully_connected(x, h2, tf.nn.relu, scope='Dense2')

        y = tf.contrib.layers.fully_connected(x, out, None, scope='Logits')

        return y
#################### Dataset Iteration ####################

def batch(x, y, batch_size=256):

    num_steps = len(x) // batch_size

    remainder = len(x) % batch_size

    samples = np.arange(len(x))

    np.random.shuffle(samples)

    

    for step in range(num_steps):

        a, b = step * batch_size, (step + 1) * batch_size

        yield x[samples[a:b]], y[samples[a:b]]



    '''

    a, b = num_steps * batch_size, num_steps * batch_size + remainder

    yield x[samples[a:b]], y[samples[a:b]]

    '''
def get_trainable_params(params_path = None):

    params = list()

    columns = ['name', 'shape', 'params']

    for var in tf.trainable_variables():

        params.append([

            var.name,

            var.shape,

            np.prod(var.shape.as_list())

        ])

    return pd.DataFrame(params, columns=columns)
num_epochs = 100

val_step = int(num_epochs/10)

learning_rate = 1e-3
tf.reset_default_graph()



image_shape = train_x.shape[1:]

    

x = tf.placeholder(tf.float32, shape=(None, *image_shape))

y = tf.placeholder(tf.int64, shape=(None,))



# Building Model

logits = conv_net(x)



with tf.name_scope('loss'):

    loss = tf.losses.sparse_softmax_cross_entropy(y, logits)

    tf.summary.scalar('loss', loss)



with tf.name_scope('train'):

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



with tf.name_scope('accuracy'):

    prediction = tf.argmax(logits, 1)

    true_pos = tf.equal(y, prediction)

    accuracy = tf.reduce_mean(tf.cast(true_pos, tf.float32))

    tf.summary.scalar('accuracy', accuracy)
sess = tf.Session()
%%time



# Initializing the variables

sess.run(tf.global_variables_initializer())



merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter('./summary')

writer.add_graph(sess.graph)



log = list()

print('Training Model')

for epoch in range(num_epochs):

    # Model Training

    tic = time.perf_counter()



    batches = list(batch(train_x, train_y))

    for i, (batch_x, batch_y) in enumerate(batches):

        if i % int(num_epochs * 1.5) == 0:

            s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y})

            writer.add_summary(s, epoch*len(batches) + i)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})



    tac = time.perf_counter()

    tictac = tac-tic



    # Model Validation

    if (epoch+1)%val_step == 0:

        train_loss, train_acc, val_loss, val_acc = [], [], [], []



        # Evaluation on Training Set

        for batch_x, batch_y in batch(train_x, train_y):

            loss_, acc_ = sess.run((loss, accuracy), feed_dict={x: batch_x, y: batch_y})

            train_loss.append(loss_)

            train_acc.append(acc_)



        # Evaluation on Validation Set

        for batch_x, batch_y in batch(val_x, val_y):

            loss_, acc_ = sess.run((loss, accuracy), feed_dict={x: batch_x, y: batch_y})

            val_loss.append(loss_)

            val_acc.append(acc_)



        train_loss = np.array(train_loss).sum()

        train_acc = np.array(train_acc).mean() * 100

        

        val_loss = np.array(val_loss).sum()

        val_acc = np.array(val_acc).mean() * 100



        params = [epoch+1, tictac, train_loss, train_acc, val_loss, val_acc]

        msg = 'epoch: {}'

        msg += ' time: {:.3f}s train_loss: {:.3f} train_acc: {:.3f}'

        msg += ' val_loss: {:.3f} val_acc: {:.3f}'

        print(msg.format(*params))

        log.append(params)
df_params = get_trainable_params()

df_params.to_csv(params_path)

df_params
df_params['params'].sum()
cols = ['epoch', 'time', 'train_loss', 'train_acc', 'val_loss', 'val_acc']

df_log_dev = pd.DataFrame(log, columns=cols)

df_log_dev = df_log_dev.set_index('epoch')

df_log_dev.to_csv(log_path_dev)

df_log_dev
df_log_dev.mean()
fig, ax_ = plt.subplots(2,1, figsize=(9,7))



df_error_rate = df_log_dev[['train_loss', 'val_loss']]

ax = df_error_rate.plot(title='Overfit Analysis', marker='.', ax=ax_[0])

ax.set_xlabel('Epoch')

ax.set_ylabel('Loss');



df_error_rate = df_log_dev[['train_acc', 'val_acc']]

ax = df_error_rate.plot(marker='.', ax=ax_[1])

ax.set_xlabel('Epoch')

ax.set_ylabel('Accuracy');
data_x = np.concatenate((train_x, val_x))

data_y = np.concatenate((train_y, val_y))
%%time



log = list()

print('Training Model')

for epoch in range(num_epochs):

    # Model Training

    tic = time.perf_counter()



    batches = list(batch(data_x, data_y))

    for i, (batch_x, batch_y) in enumerate(batches):

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})



    tac = time.perf_counter()

    tictac = tac-tic



    # Model Validation

    if (epoch+1)%val_step == 0:

        train_loss, train_acc = list(), list()



        # Evaluation on Training Set

        for batch_x, batch_y in batch(train_x, train_y):

            loss_, acc_ = sess.run((loss, accuracy), feed_dict={x: batch_x, y: batch_y})

            train_loss.append(loss_)

            train_acc.append(acc_)



        train_loss = np.array(train_loss).sum()

        train_acc = np.array(train_acc).mean() * 100



        params = [epoch+1, tictac, train_loss, train_acc]

        msg = 'epoch: {}'

        msg += ' time: {:.3f}s train_loss: {:.3f} train_acc: {:.3f}'

        print(msg.format(*params))

        log.append(params)
cols = ['epoch', 'time', 'train_loss', 'train_acc']

df_log_train = pd.DataFrame(log, columns=cols)

df_log_train = df_log_train.set_index('epoch')

df_log_train.to_csv(log_path_train)

df_log_train
fig, ax_ = plt.subplots(2,1, figsize=(9,7))



df_error_rate = df_log_train[['train_loss']]

ax = df_error_rate.plot(title='Overfit Analysis', marker='.', ax=ax_[0])

ax.set_xlabel('Epoch')

ax.set_ylabel('Loss');



df_error_rate = df_log_train[['train_acc']]

ax = df_error_rate.plot(marker='.', ax=ax_[1])

ax.set_xlabel('Epoch')

ax.set_ylabel('Accuracy');
test_y = sess.run(prediction, feed_dict={x: test_x})
output = pd.DataFrame(test_y, columns=['Label'])

output.index = np.arange(1, len(output) + 1)

output.index.names = ['ImageId']

output.to_csv(submission_path)