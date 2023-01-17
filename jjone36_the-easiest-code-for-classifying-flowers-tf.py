import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

from skimage.data import imread

import cv2

from sklearn.metrics import accuracy_score



import tensorflow as tf 

seed = 100
os.listdir('../input/flowers/flowers/daisy')[:5]
path = '../input/flowers/flowers/'

flower_labels = os.listdir(path)

flower_labels
num_per_class = {}



for i in flower_labels:

    num_per_class[i] = len(os.listdir(path + i))



num_per_class
im_size = 64

n_class = len(num_per_class)



images = []

labels = []



for i in flower_labels:

    data_path = path + str(i)

    filenames = [i for i in os.listdir(data_path)

                 if i.endswith('.jpg')]

    for f in filenames:

        img = imread(data_path + '/' + f)

        img = cv2.resize(img, (im_size, im_size))

        images.append(img)

        labels.append(i)
examples = np.random.randint(0, len(images), 9)

plt.figure(figsize = (10, 10))

i = 1



for ex in examples:

    img = images[ex]

    plt.subplot(3, 3, i)

    plt.subplots_adjust(wspace = 0.5)

    plt.axis('off')

    plt.title("Label: {}".format(labels[ex]))

    plt.imshow(img)

    i += 1
# Converting into numpy array 

images = np.array(images)



# Reducing the pixel values between 0 ~ 1

images = images.astype('float32') / 255.
# Label encodding

def one_hot(labels):

    labels = pd.DataFrame(labels)

    labels = pd.get_dummies(labels)

    return np.array(labels)
# Shuffle

def img_shuffle(images, labels_oh, frt):

    a = im_size*im_size*3

    X_train = images.reshape((images.shape[0], a))

    X_y_train = np.hstack((X_train, labels_oh))

    

    np.random.shuffle(X_y_train)

    cut = int(len(X_y_train) * frt)

    

    X_val = X_y_train[:cut, :a]

    y_val = X_y_train[:cut, a:]

    X_train = X_y_train[cut:, :a]

    y_train = X_y_train[cut:, a:]

    

    X_train = X_train.reshape((X_train.shape[0], im_size, im_size, 3))

    X_val = X_val.reshape((X_val.shape[0], im_size, im_size, 3))

    

    return X_train, X_val, y_train, y_val
labels_oh = one_hot(labels)

X_train, X_val, y_train, y_val = img_shuffle(images, labels_oh, frt = .2)



print("The input shape of train set is {}".format(X_train.shape))

print("The input shape of validation set is {}".format(X_val.shape))

print("The output shape of train set is {}".format(y_train.shape))

print("The output shape of validation set is {}".format(y_val.shape))
# Initialize placeholders 

X = tf.placeholder(tf.float32, [None, im_size, im_size, 3])

y = tf.placeholder(tf.float32, [None, n_class])



print("X = ", X)

print("y = ", y)
# ConvNet_1

Z1 = tf.layers.conv2d(X, filters = 32, kernel_size = 7, strides = [2, 2], padding = 'VALID')

A1 = tf.nn.relu(Z1)

P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')



# ConvNet_2

Z2 = tf.layers.conv2d(P1, filters = 64, kernel_size = 3, strides = [1, 1], padding = 'VALID')

A2 = tf.nn.relu(Z2)

P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')



# Flattening 

P2 = tf.contrib.layers.flatten(P2)



# Fully-connected

Z3 = tf.contrib.layers.fully_connected(P2, n_class, activation_fn = None)
# Cost function 

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = y))
# Backpropagation with adam optimizer

optimizer = tf.train.AdamOptimizer(learning_rate = .01).minimize(loss)
epochs = 100

batch_size = 50



def create_batch(X_train, y_train, batch_size):

    m = X_train.shape[0]

    

    # Shuffling

    NUM = list(np.random.permutation(m))

    X_shuffled = X_train[NUM, :]

    y_shuffled = y_train[NUM, :]

    

    # Number of batches

    n_batch = int(m/batch_size)

    batches = []

    

    # Splitting the data  

    for i in range(0, n_batch):

        X_batch = X_shuffled[i*batch_size:(i+1)*batch_size, :, :, :]

        y_batch = y_shuffled[i*batch_size:(i+1)*batch_size, :]



        batch = (X_batch, y_batch)

        batches.append(batch)

    

    # Handling the tail of the data 

    X_batch_end = X_shuffled[n_batch*batch_size+1:, :, :, :]

    y_batch_end = y_shuffled[n_batch*batch_size+1:, :]

    batch = (X_batch_end, y_batch_end)

    batches.append(batch)

    

    return batches
# Checking 

batches = create_batch(X_train, y_train, batch_size)

(X_batch, y_batch) = batches[0]

print(X_batch.shape)

print(y_batch.shape)
costs = []



# Initialization all the variables globally

init = tf.global_variables_initializer()



# Run the session and compute

with tf.Session() as sess:

    

    sess.run(init)

    

    for epoch in range(epochs):



        # mini-batch gradient descents

        batches = create_batch(X_train, y_train, batch_size)

        batch_cost = 0

        m = X_train.shape[0]

        n_batch = int(m/batch_size)

        

        for batch in batches:

            

            (X_batch, y_batch) = batch           

            _, temp_cost = sess.run([optimizer, loss], feed_dict = {X : X_batch, y : y_batch})

            batch_cost += temp_cost/n_batch



        # Print the cost per each epoch

        if epoch % 10 == 0:

            print("Cost after {0} epoch: {1}".format(epoch, batch_cost))

        if epoch % 1 == 0:

            costs.append(batch_cost)



    # plot the cost

    plt.plot(np.squeeze(costs))

    plt.ylabel('cost')

    plt.xlabel('iterations')

    plt.show()



    pred_op = tf.argmax(Z3, 1)

    actual = tf.argmax(y_train, 1)

    correct_pred = tf.equal(pred_op, actual)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))