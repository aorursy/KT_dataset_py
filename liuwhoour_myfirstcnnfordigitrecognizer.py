%matplotlib inline

import matplotlib.pyplot as plt

import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split

import time

from datetime import timedelta

import math
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)
# plot the first 25 digits in the training set. 

f, ax = plt.subplots(5, 5)

# plot some 4s as an example

for i in range(1,26):

    # Create a 1024x1024x3 array of 8 bit unsigned integers

    data = train.iloc[i,1:785].values #this is the first number

    nrows, ncols = 28, 28

    grid = data.reshape((nrows, ncols))

    n=math.ceil(i/5)-1

    m=[0,1,2,3,4]*5

    ax[m[i-1], n].imshow(grid)
i=1

img = train.iloc[:,1:].iloc[i].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[:,0].iloc[i])
images = train.iloc[:,1:].values.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)

y = train["label"].values.ravel()
y = np.array(y)

y = LabelEncoder().fit_transform(y)[:, None]

y = OneHotEncoder().fit_transform(y).todense()

image_train, image_val, y_train, y_val = train_test_split(images, y, test_size=0.1, random_state=0)
print(y_train.shape)

print(image_val.shape)
def build_weight(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))



def build_biases(length):

    return tf.Variable(tf.constant(0.05, shape=[length]))
def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)



def maxpool2d(x, k=2):

    # MaxPool2D wrapper

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],

                          padding='SAME')
index_in_epoch = 0

num_examples = image_train.shape[0]

epochs_completed = 0 

def next_batch(batch_size):

    

    global image_train

    global y_train

    global index_in_epoch

    global epochs_completed

    

    start = index_in_epoch

    index_in_epoch += batch_size

    

    # when all trainig data have been already used, it is reorder randomly    

    if index_in_epoch > num_examples:

        # finished epoch

        epochs_completed += 1

        # shuffle the data

        perm = np.arange(num_examples)

        np.random.shuffle(perm)

        image_train = image_train[perm]

        y_train = y_train[perm]

        # start next epoch

        start = 0

        index_in_epoch = batch_size

        assert batch_size <= num_examples

    end = index_in_epoch

    return image_train[start:end], y_train[start:end]
# Number of colour channels for the images: 1 channel for gray-scale.

num_channels = 1

num_classes = 10

img_size = 28

img_size_flat = img_size * img_size# MNIST data input (img shape: 28*28)

image_shape = (img_size, img_size)

train_batch_size = 64
x = tf.placeholder(tf.float32, shape=[None, img_size_flat])

# (?,784) => (?,28,28,1)

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_out = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_out')

y_cls = tf.argmax(y_out, dimension=1)
# first convolutional layer

wconv1 = build_weight([5, 5, 1, 32])

bconv1 = build_biases(32)

# convnet (?, 28, 28, 32)

conv1 = conv2d(x_image, wconv1, bconv1)

print(conv1.get_shape())

# maxpoll 2x2 (?, 14, 14, 32)

pool1 = maxpool2d(conv1)

print(pool1.get_shape())
# second convolutional layer

wconv2 = build_weight([5, 5, 32, 64])

bconv2 = build_biases(64)

# convnet2 (?, 14, 14, 64)

conv2 = conv2d(pool1, wconv2, bconv2)

print(conv2.get_shape())

# maxpoll 2x2 (?, 7, 7, 64)

pool2 = maxpool2d(conv2)

print(pool2.get_shape())
# Fully connected layer

wfc1 = build_weight([7*7*64, 1024])

bfc1 = build_biases(1024)

# (?, 7, 7, 64) => (?, 3136)

fc1 = tf.reshape(pool2, [-1, 7*7*64])

fc1 = tf.nn.relu(tf.matmul(fc1, wfc1) + bfc1)

# Apply Dropout

dropout = 0.65 # keep_prob

fc1 = tf.nn.dropout(fc1, dropout)
# Output, softmax layer, class prediction

wfc2 = build_weight([1024, num_classes])

bfc2 = build_biases(num_classes)

out = tf.matmul(fc1, wfc2) + bfc2

print(out.get_shape()) # (?, 10) 

predict = tf.nn.softmax(out)

#The class-number is the index of the largest element

predict_cls = tf.argmax(out, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_out)

cost = tf.reduce_mean(cross_entropy)



learning_rate = 1e-4

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)



correct_prediction = tf.equal(predict_cls, y_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Create TensorFlow session

session = tf.Session()

session.run(tf.global_variables_initializer())
# Counter for total number of iterations performed so far.

total_iterations = 0



def optimize(num_iterations):

    global total_iterations

    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):

        x_batch, y_true_batch = next_batch(train_batch_size)

        feed_dict_train = {x: x_batch, y_out: y_true_batch}

        # Run the optimizer using this batch of training data.

        # TensorFlow assigns the variables in feed_dict_train

        # to the placeholder variables and then runs the optimizer.

        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 50 == 0:

            # Calculate the accuracy on the training-set.

            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.

            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.

            print(msg.format(i + 1, acc))

            #validation_accuracy = session.run(accuracy,feed_dict={x: image_val, y_out: y_val}) 

            #print('validation_accuracy => %.4f'%validation_accuracy)

            

    # Update the total number of iterations performed.

    total_iterations += num_iterations

    # Ending time.

    end_time = time.time()

    # Difference between start and end-times.

    time_dif = end_time - start_time

    # Print the time-usage.

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
# default accuracy

optimize(num_iterations=1000)
#check final accuracy on validation set

validation_accuracy = session.run(accuracy,feed_dict={x: image_val, y_out: y_val}) 

print('validation_accuracy => %.4f'%validation_accuracy)
BATCH_SIZE = 64

test_images = test.iloc[:,0:].values.astype(np.float)

test_images = np.multiply(test_images, 1.0 / 255.0)
#print(test_images.shape)
# using batches is more resource efficient

predicted_lables = np.zeros(test_images.shape[0])

for i in range(0,test_images.shape[0]//BATCH_SIZE):

    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = session.run(predict_cls, feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE, :]})
#print(predicted_lables.shape)

predicted_lables = predicted_lables.astype(np.int32)
submission = pd.DataFrame(data={'ImageId':(np.arange(predicted_lables.shape[0])+1), 'Label':predicted_lables})

submission.to_csv('submission.csv', index=False)