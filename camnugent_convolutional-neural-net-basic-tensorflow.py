import pandas as pd

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split
#####

# Load in the data

#####

print('loading data')

# load function from: https://www.kaggle.com/kmader/exploring-the-icebergs-with-skimage-and-keras

# b/c I didn't want to reinvent the wheel

def load_and_format(in_path):

    """ take the input data in .json format and return a df with the data and an np.array for the pictures """

    out_df = pd.read_json(in_path)

    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)

    out_images = np.stack(out_images).squeeze()

    return out_df, out_images





train_df, train_images = load_and_format('../input/train.json')



test_df, test_images = load_and_format('../input/test.json')



X_train, X_valid, y_train, y_valid = train_test_split(train_images,

                                                   train_df['is_iceberg'].as_matrix(),

                                                   test_size = 0.3

                                                   )

print('Train', X_train.shape, y_train.shape)

print('Validation', X_valid.shape, y_valid.shape)
#convert to np.float32 for use in tensorflow

X_train = X_train.astype(np.float32)

y_train = y_train.astype(np.float32)

X_valid = X_valid.astype(np.float32)

y_valid= y_valid.astype(np.float32)
#for stability

def reset_graph(seed=42):

    tf.reset_default_graph()

    tf.set_random_seed(seed)

    np.random.seed(seed)



reset_graph()
print('designing model')

# Training Parameters

learning_rate = 0.005

n_epochs = 2500 # changed to 2500 for a LB score of ~2.69





# Network Parameters

num_input = 75*75 #size of the images

num_classes = 2 # Binary

dropout = 0.4 # Dropout, probability to keep units
X = tf.placeholder(tf.float32, shape=(None, 75, 75, 2), name="X")

y = tf.placeholder(tf.int64, shape=(None), name="y")





with tf.variable_scope('ConvNet'):



    he_init = tf.contrib.layers.variance_scaling_initializer()



    # Convolution Layer with 32 filters and a kernel size of 5

    conv1 = tf.layers.conv2d(X, filters=32,  kernel_size=[5, 5], activation=tf.nn.relu)

    # Max Pooling 

    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)



    conv2 = tf.layers.conv2d(pool1, filters=64,  kernel_size=[3,3], activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)



    conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3], activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)



    conv4 = tf.layers.conv2d(pool3, filters=256, kernel_size=[3,3], activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2)

    

    # Flatten the data to a 1-D vector for the fully connected layer

    fc1 = tf.contrib.layers.flatten(pool4)



    # Fully connected layer 

    fc2 = tf.layers.dense(fc1, 32, 

                        kernel_initializer=he_init, activation=tf.nn.relu)



    # Apply Dropout 

    fc3 = tf.layers.dropout(fc2, rate=dropout)



    logits = tf.layers.dense(fc3, num_classes, activation=tf.nn.sigmoid)
with tf.name_scope("loss"):

    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

    loss = tf.reduce_mean(xentropy, name="loss")
with tf.name_scope("train"):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):

    correct = tf.nn.in_top_k(logits, y, 1)

    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

#saver = tf.train.Saver()
print('training model\n')

with tf.Session() as sess:

    init.run()

    for epoch in range(n_epochs):

        sess.run(training_op, feed_dict={X: X_train, y: y_train})   

        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})

        acc_test = accuracy.eval(feed_dict={X: X_valid,

                                            y: y_valid})

    

        print(epoch, "Train accuracy:", acc_train, "Validation accuracy:", acc_test)

    save_path = saver.save(sess, "./cam_iceberg_model_final.ckpt")
#convert the test images to float32

test_images =test_images.astype(np.float32) 

test_images.shape



print('making predictions\n')

#make external predictions on the test_dat

with tf.Session() as sess:

    saver.restore(sess, "./cam_iceberg_model_final.ckpt") # or better, use save_path

    Z = logits.eval(feed_dict={X: test_images}) #outputs switched to logits

    y_pred = Z[:,1]



output = pd.DataFrame(test_df['id'])

output['is_iceberg'] = y_pred



output.to_csv('cam_tf_cnn.csv', index=False)