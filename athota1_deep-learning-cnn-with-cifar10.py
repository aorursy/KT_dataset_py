import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import cifar10

import tensorflow as tf

from tqdm import tqdm_notebook as tqdm
#Load CIFAR10 dataset

(x_train, y_train),(x_test, y_test) = cifar10.load_data()

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
Num_Classes = 10

EPOCHS = 10

BATCH_SIZE = 32
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, 32, 32, 3])

Y = tf.placeholder(tf.int64, [None,1])

X_extend = tf.reshape(X, [-1, 32,32,3])

Y_onehot = tf.one_hot(indices = Y, depth = Num_Classes)

# First cnn layer

conv1_w = tf.get_variable("conv1_w", [3,3,3,10], initializer= tf.random_normal_initializer(stddev=1e-2))

conv1_b = tf.get_variable("conv1_b", [10], initializer= tf.random_normal_initializer(stddev=1e-2))

conv1 = tf.nn.conv2d(X_extend, conv1_w, strides = [1,1,1,1], padding= 'SAME')+conv1_b

relu1= tf.nn.relu(conv1)

pool1 = tf.nn.max_pool(value=relu1, ksize= [1,2,2,1], strides = [1,2,2,1], padding= 'SAME')

# Second cnn layer

conv2_w = tf.get_variable("conv2_w", [3,3,10,10], initializer= tf.random_normal_initializer(stddev=1e-2))

conv2_b = tf.get_variable("conv2_b", [10], initializer= tf.random_normal_initializer(stddev=1e-2))

conv2 = tf.nn.conv2d(pool1, conv2_w, strides = [1,1,1,1], padding= 'SAME')+conv2_b

relu2= tf.nn.relu(conv2)

pool2 = tf.nn.max_pool(value=relu2, ksize= [1,2,2,1], strides = [1,2,2,1], padding= 'SAME')

# Third cnn layer

conv3_w = tf.get_variable("conv3_w", [3,3,10,10], initializer= tf.random_normal_initializer(stddev=1e-2))

conv3_b = tf.get_variable("conv3_b", [10], initializer= tf.random_normal_initializer(stddev=1e-2))

conv3 = tf.nn.conv2d(pool2, conv3_w, strides = [1,1,1,1], padding= 'SAME')+conv3_b

relu3= tf.nn.relu(conv3)

# Flatten layer

flatten = tf.reshape(relu3, [-1, 8*8*10])

#first fully connect layer

fc1 = tf.layers.dense(inputs= flatten, units = 512, activation = tf.nn.relu, use_bias =True)

# first fully connect layer

fc2 = tf.layers.dense(inputs= fc1, units = 512, activation = tf.nn.relu, use_bias =True)

# Output layer

output = tf.layers.dense(inputs=fc2, units = Num_Classes, activation = None, use_bias= True)

# Loss function

loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_onehot, logits=output))

# Accuracy function

accu = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, axis = 1), Y[:,0]), dtype = tf.float32))

# Optimizer

opt = tf.train.AdamOptimizer(0.001).minimize(loss)
# Initiate the parameters

sess = tf.Session()

init = tf.global_variables_initializer()

sess.run(init)
# Training loop

for epoch in range(0, EPOCHS):

    for step in tqdm(range(int(len(x_train)/BATCH_SIZE)), desc=('Epoch '+str(epoch))):

        # Get next batch of training data

        x_batch = x_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        y_batch = y_train[step*BATCH_SIZE:step*BATCH_SIZE+BATCH_SIZE]

        # train

        loss_value, _ = sess.run([loss, opt], feed_dict={X: x_batch, Y: y_batch})

        # Use the first 1000 images from testing dataset to test the CNN model

        loss_value, accuracy_value = sess.run([loss, accu], feed_dict= {X:x_test[:1000], Y: y_test[:1000]})

        print('Epoch ', epoch + 1, ' loss: ', loss_value,'  accuracy:  ', accuracy_value)
# Extract first convolution layer filters

conv1_w_extract = sess.run(conv1_w)

print(conv1_w_extract.shape)

plt.figure(figsize = (20,20))



# Plot first 10 filters

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(np.reshape(conv1_w_extract[:,:,:,i]*100, [3,3,3]))
# Extract first convolution layer feature maps

conv1_fmaps = sess.run(relu1, feed_dict = {X: [x_train[0]]})

print(conv1_fmaps.shape)



plt.figure(figsize = (20,20))

# Plot first 10 feature maps

for i in range(10):

    plt.subplot(3, 10, i+1)

    plt.imshow(np.reshape(conv1_fmaps[0,:,:,i], [32,32]))