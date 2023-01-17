# These are all the modules we'll be using later. Make sure you can import them

# before proceeding further.

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv

plt.rcParams['image.cmap'] = 'Greys'

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['figure.figsize'] = (2,2)
LABELS = 10 # Number of different types of labels (1-10)

PIXELS = 28 # width / height of the image

CHANNELS = 1 # Number of colors in the image (greyscale)



TRAIN = 21000  #40000 # Train data size

VALID = 42000 - TRAIN # Validation data size



STEPS = 5000 #20001   # Number of steps to run

BATCH = 100 # Stochastic Gradient Descent batch size

PATCH = 5 # Convolutional Kernel size

DEPTH = 12 #32 # Convolutional Kernel depth size == Number of Convolutional Kernels

HIDDEN = 100 #1024 # Number of hidden neurons in the fully connected layer



LEARNING_RATE = 0.003 # Initial Learning rate

DECAY_FACTOR = 0.95 # Continuous Learning Rate Decay Factor (per 1000 steps)
def acc(pred, labels):

    return 100.0 * np.mean(np.float32(np.argmax(pred, axis=1) == np.argmax(labels, axis=1)), axis=0)
def shuffle(data, labels):

    rnd = np.random.get_state()

    np.random.shuffle(data)

    np.random.set_state(rnd)

    np.random.shuffle(labels)
data = read_csv('../input/train.csv') # Read csv file in pandas dataframe

labels = np.array(data.pop('label')) # Remove the labels as a numpy array from the dataframe

labels = np.array([np.arange(LABELS) == label for label in labels])

data = np.array(data, dtype=np.float32)/255.0-1.0# Convert the dataframe to a numpy array

data = data.reshape(len(labels), PIXELS, PIXELS, CHANNELS) # Reshape the data into 42000 2d images

train_data = data[:TRAIN]

train_labels = labels[:TRAIN]

valid_data = data[TRAIN:]

valid_labels = labels[TRAIN:]

test_data = np.array(read_csv('../input/test.csv'), dtype=np.float32)/255.0-1.0

test_data = test_data.reshape(test_data.shape[0], PIXELS, PIXELS, CHANNELS)



shuffle(train_data, train_labels) # Randomly shuffle the training data



print('train data shape = ' + str(train_data.shape) + ' = (TRAIN, PIXELS, PIXELS, CHANNELS)')

print('labels shape = ' + str(labels.shape) + ' = (TRAIN, LABELS)')
tf_train_data = tf.placeholder(tf.float32, shape=(BATCH, PIXELS, PIXELS, CHANNELS))

tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH, LABELS))

tf_valid_data = tf.constant(valid_data)

tf_test_data = tf.constant(test_data)
global_step = tf.Variable(0)

w1 = tf.Variable(tf.truncated_normal([PATCH, PATCH, CHANNELS, DEPTH], stddev=0.1))

b1 = tf.Variable(tf.zeros([DEPTH]))

w2 = tf.Variable(tf.truncated_normal([PATCH, PATCH, DEPTH, 2*DEPTH], stddev=0.1))

b2 = tf.Variable(tf.constant(1.0, shape=[2*DEPTH]))

w3 = tf.Variable(tf.truncated_normal([PIXELS // 4 * PIXELS // 4 * 2*DEPTH, HIDDEN], stddev=0.1))

b3 = tf.Variable(tf.constant(1.0, shape=[HIDDEN]))

w4 = tf.Variable(tf.truncated_normal([HIDDEN, LABELS], stddev=0.1))

b4 = tf.Variable(tf.constant(1.0, shape=[LABELS]))
def logits(data):

    conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')

    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    hidden = tf.nn.relu(pool + b1)

    conv = tf.nn.conv2d(hidden, w2, [1, 1, 1, 1], padding='SAME')

    pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    hidden = tf.nn.relu(pool + b2)

    reshape = tf.reshape(hidden, (-1, PIXELS // 4 * PIXELS // 4 * 2*DEPTH))

    hidden = tf.nn.relu(tf.matmul(reshape, w3) + b3)

    return tf.matmul(hidden, w4) + b4
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits(tf_train_data), tf_train_labels))
#learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, 1000, DECAY_FACTOR, staircase=False)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
train_prediction = tf.nn.softmax(logits(tf_train_data))

valid_prediction = tf.nn.softmax(logits(tf_valid_data))

test_prediction = tf.nn.softmax(logits(tf_test_data))
session = tf.Session()

tf.global_variables_initializer().run(session=session)
_step = 0

for step in np.arange(STEPS):

    _step += 1

    if _step*BATCH > TRAIN: # Reshuffle data

        shuffle(train_data, train_labels)

        _step = 0

    start = (step * BATCH) % (TRAIN - BATCH); stop = start + BATCH

    batch_data = train_data[start:stop]

    batch_labels = train_labels[start:stop, :]



    feed_dict = {tf_train_data:batch_data, tf_train_labels:batch_labels}

    opt, batch_loss, batch_prediction = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)



    if (step % 1000 == 0):

        b_acc = acc(batch_prediction, batch_labels) # Batch Accuracy

        v_acc = acc(valid_prediction.eval(session=session), valid_labels) # Valid Accuracy

        print('Step %i'%step, end='\t')

        print('Loss = %.2f'%batch_loss, end='\t')

        print('Batch Acc. = %.1f'%b_acc, end='\t\t')

        print('Valid. Acc. = %.1f'%v_acc, end='\n')

        #print('learning rate = %.4f'%learning_rate.eval())
test_labels = np.argmax(test_prediction.eval(session=session), axis=1)
k = 0 # Try different images indices k

plt.imshow(test_data[k,:,:,0])

plt.axis('off')

plt.show()

print("Label Prediction: %i"%test_labels[k])
submission = DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})

submission.to_csv('submission.csv', index=False)

submission.head()
#session.close()