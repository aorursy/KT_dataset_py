%matplotlib inline

import preprocessing



import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import os

import pickle

import tensorflow as tf



# to make this notebook's output stable across runs

def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)

    



def log_dir(prefix=''):

    from datetime import datetime

    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')

    root_logdir = os.path.join(tmp_dir, prefix, 'logs')

    if prefix:

        prefix += '-'

    name = prefix + 'run-' + now

    return '{}/{}/'.format(root_logdir, name)





# permute whole training set and generate batches from it

# better alternative => tf.data.Dataset.shuffle

def shuffle_batch(X, y, batch_size):

    rnd_idx = np.random.permutation(len(X))

    n_batches = len(X) // batch_size

    for batch_idx in np.array_split(rnd_idx, n_batches):

        batch_X, batch_y = X[batch_idx], y[batch_idx]

        yield batch_X, batch_y
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_train.dtype
X_train[0], y_train[0]
def plot_digit(data):

    plt.imshow(data, cmap=mpl.cm.binary, interpolation='nearest')

    plt.axis("off")

    plt.show()





def plot_digits(instances, images_per_row=10, **options):

    size = 28

    n_instances = len(instances)

    images_per_row = min(n_instances, images_per_row)

    n_rows = (n_instances - 1) // images_per_row + 1

    n_empty = n_rows * images_per_row - n_instances

    np.append(instances, np.zeros((size, size * n_empty)))

    row_images = []

    for row in range(n_rows):

        row_images.append(np.concatenate(

            instances[row * images_per_row: (row + 1) * images_per_row],

            axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap=mpl.cm.binary, **options)

    plt.axis('off')
plt.figure(figsize=(9, 9))

example_images = X_train[np.random.choice(X_train.shape[0], 100)]

plot_digits(example_images, images_per_row=10)

plt.show()
# deep feed forward network layers

n_hidden1 = 300

n_hidden2 = 100



# training

n_epochs = 1000

n_batch_size = 32



# early stopping

max_epochs_without_progress = 30
reset_graph()



n_inputs = 28 * 28 # single image is loaded as 28x28 2D array, but input to network is 1D

n_outputs = 10 # classify digits from 0-9



# reshape and normalize input

X_train = X_train.astype(np.float32).reshape(-1, n_inputs) / 255.0

X_test = X_test.astype(np.float32).reshape(-1, n_inputs) / 255.0

y_train = y_train.astype(np.int32)

y_test = y_test.astype(np.int32)



# split training data to validation and training data

X_valid, X_train = X_train[:5000], X_train[5000:] 

y_valid, y_train = y_train[:5000], y_train[5000:]
def dense_layer(X, n_neurons, name, kernel_init=tf.truncated_normal):

    with tf.name_scope(name):

        fan_in = int(X.get_shape()[1])

        W = tf.Variable(kernel_init(shape=(fan_in, n_neurons)), name="weights")

        b = tf.Variable([tf.zeros(n_neurons)], name="bias")

        return tf.add(tf.matmul(X, W), b)        

    

    

def batch_norm(Z, mean, var, epsilon=0.001):

    # apply the initial batch normalizing transform

    Z_hat = (Z - mean) / tf.sqrt(var + epsilon)



    fan_in = int(Z.get_shape()[1])



    # create two new parameters, scale and beta (shift)

    scale = tf.Variable(tf.ones([fan_in]), name='scale')

    beta = tf.Variable(tf.zeros([fan_in]), name='beta')



    # scale and shift to obtain the final output of the batch normalization

    # this value is fed into the activation function 

    return scale * Z_hat + beta

    

    

def batch_norm_layer(Z, is_training, epsilon=0.001, decay=0.999):

    with tf.name_scope('batch_norm'):                

        pop_mean = tf.Variable(tf.zeros([Z.get_shape()[-1]]), trainable=False, 

                               name='pop_mean')

        pop_var = tf.Variable(tf.ones([Z.get_shape()[-1]]), trainable=False, 

                              name='pop_var')

        

        def mean_var_with_update():

            # calculate batch mean and variance

            batch_mean, batch_var = tf.nn.moments(Z, [0])        

            

            # estimate the population mean and variance during training,

            # use an exponential moving average

            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))

            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

                        

            with tf.control_dependencies([train_mean, train_var]):

                return tf.identity(batch_mean), tf.identity(batch_var)

        

        # use batch mean and var for training but whole population mean and var

        # for prediction

        mean, var = tf.cond(is_training, 

                            mean_var_with_update,

                            lambda: (pop_mean, pop_var))

        

        return batch_norm(Z, mean, var, epsilon)





def he_normal(shape):

    return tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[0]))
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')

y = tf.placeholder(tf.int32, shape=(None), name='y')
with tf.name_scope('dnn'):

    training = tf.placeholder(tf.bool, name='training')

    hidden1 = tf.nn.relu(

        batch_norm_layer(

            dense_layer(X, n_hidden1, 'hidden1', kernel_init=he_normal),

            is_training=training))

    hidden2 = tf.nn.relu(

        batch_norm_layer(

            dense_layer(hidden1, n_hidden2, 'hidden2', kernel_init=he_normal),

            is_training=training))

    logits = dense_layer(hidden2, n_outputs, 'outputs')
with tf.name_scope('loss'):

    # we could also use categorical_cross_entropy with labels encoded

    # to one-hot vector

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,

                                                              logits=logits)

    # compute mean loss for whole batch

    loss = tf.reduce_mean(xentropy, name='loss') 

    # creates a node in the graph that will evaluate the MSE value and write it

    # to a TensorBoard-compatible binary log string called a summary

    loss_summary = tf.summary.scalar('log_loss', loss)
with tf.name_scope('train'):           

    optimizer = tf.train.AdamOptimizer()

    training_op = optimizer.minimize(loss) # apply gradients
with tf.name_scope('eval'):

    # binary table with true for every correct label

    correct = tf.nn.in_top_k(logits, y, 1) 

    # mean accuracy for whole batch

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # log progress for tensorboard

    accuracy_summary = tf.summary.scalar('accuracy', accuracy)    
model_name = 'mnist_ff'



# training state

tmp_dir = '.tmp'



# prepare tensorboard statistic logging

logdir = log_dir(model_name)



# logdir and (optional) graph to visualize

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
# prepare savers for model checkpoints

model_dir = os.path.join(tmp_dir, model_name)



os.makedirs(model_dir, exist_ok=True)



checkpoint_dir = os.path.join(model_dir, 'checkpoints')

best_dir = os.path.join(model_dir, 'models')

meta_file = os.path.join(model_dir, '.meta.pkl')



# by default saver uses graph that is created *before* saver creation

ckpt_saver = tf.train.Saver()

best_saver = tf.train.Saver()
init = tf.global_variables_initializer()



# set or restore training meta data

best_loss = np.infty

epochs_without_progress = 0

global_epoch = 0



if os.path.isfile(meta_file):

    with open(meta_file, 'rb') as meta:

        global_epoch, epochs_without_progress, best_loss = pickle.load(meta)



print('Starting from (epoch: {}, epoch without progress: {}, best_loss: {:.5f})'

      .format(global_epoch, epochs_without_progress, best_loss))

        

# start/restore training session

with tf.Session() as sess:

    # you could also use tf.train.import_meta_graph() for importing graph structure

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:

        # initialize variables from last checkpoint

        chpnt_saver.restore(sess, latest_checkpoint)

    else:

        init.run()  # actually initialize all the variables

        

    # run actual training starting from last checkpoint

    for epoch in range(global_epoch, n_epochs):

        

        if epochs_without_progress > max_epochs_without_progress:

            print('Early stopping')

            break        

        

        # train for whole training sets batch per batch

        for X_batch, y_batch in shuffle_batch(X_train, y_train, n_batch_size):

            sess.run(training_op, feed_dict={training: True, X: X_batch, y: y_batch})                            

                                

        # compute accuracy, loss and summary ops for tensorboard on validation set

        acc_val, loss_val, acc_summary_str, loss_summary_str = sess.run(

            [accuracy, loss, accuracy_summary, loss_summary],

            feed_dict={training: False, X: X_valid, y: y_valid})

        

        file_writer.add_summary(acc_summary_str, epoch)

        file_writer.add_summary(loss_summary_str, epoch)        

        

        if epoch % 5 == 0:

            print('Epoch:', epoch,

                  '\tValidation accuracy: {:.3f}%'.format(acc_val * 100),

                  '\tLoss: {:.5f}'.format(loss_val))

            

        ckpt_saver.save(sess, os.path.join(checkpoint_dir, 'ckpt'))

                 

        # early stopping

        if loss_val < best_loss:

            epochs_without_progress = 0

            best_saver.save(sess, os.path.join(best_dir, 'best'))

            best_loss = loss_val

        else:

            epochs_without_progress += 1

        

        with open(meta_file, 'wb') as meta:

            pickle.dump([epoch, epochs_without_progress, best_loss], meta)
# evaluate best model on test set

with tf.Session() as sess:

    best_saver.restore(sess, tf.train.latest_checkpoint(best_dir))

    accuracy_val = accuracy.eval(feed_dict={training: False, X: X_test, y: y_test})
accuracy_val