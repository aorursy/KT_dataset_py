import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

import numpy as np

import warnings

import tensorflow as tf

warnings.filterwarnings('ignore')

%matplotlib inline

import os

os.listdir('../input')
train_file = "../input/train.csv"

test_file = "../input/test.csv"
test = pd.read_csv(test_file)
images_train = pd.read_csv(train_file)

images_test = pd.read_csv(test_file)
images_test.head(3)
images_train.head()
labels_train=np.array(images_train['label']).reshape(-1,1)

#labels_test=np.array(images_test['label']).reshape(-1,1)

labels_train.shape
onehot=OneHotEncoder()

onehot.fit(labels_train)

labels_train=onehot.transform(labels_train)

#labels_test=onehot.transform(labels_test)

labels_train=labels_train.toarray()

#labels_test=labels_test.toarray()



# convert train and test data to an array

images_train=np.array(images_train.iloc[:, 1:]).reshape(42000,784)

images_test=np.array(images_test.iloc[:, :]).reshape(28000,784)



# Convert all the values between 0 and 1

images_train=images_train/255

images_test=images_test/255
images_train.shape
labels_train.shape
x_ph = tf.placeholder(tf.float32, shape=[None, 784]) 

    

y_ph = tf.placeholder(tf.float32, shape=[None, 10])
def weight_variable(shape):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial)



def bias_variable(shape):

    initial = tf.constant(0.1, shape=shape)

    return tf.Variable(initial)
def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')



def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')
# originally an image is simply a flat array of 784 numbers representing pixels

# we are reshaping it to original dimension of 28x28X1 , had this been a colored image. 

# there will be 3 channels instead of just 1 on the third dimension

x_image = tf.reshape(x_ph, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])

b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])

b_conv2 = bias_variable([64])



h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])



W_fc1 = weight_variable([7 * 7 * 64, 1024]) # the layer has 1024 nodes

b_fc1 = bias_variable([1024])



h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)



h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 200]) # second layer has 200 nodes

b_fc2 = bias_variable([200])



h_fc2= tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop=tf.nn.dropout(h_fc2,keep_prob)



W_fc3 = weight_variable([200, 100]) # the output has 10 nodes(the target labels)

b_fc3 = bias_variable([100])





h_fc3= tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop=tf.nn.dropout(h_fc3,keep_prob)



W_fc4 = weight_variable([100, 10]) # the output has 10 nodes(the target labels)

b_fc4 = bias_variable([10])





y_conv_logits=tf.matmul(h_fc3_drop, W_fc4) + b_fc4
loss = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_ph, 

                                            logits=y_conv_logits))



train_step = tf.train.AdamOptimizer().minimize(loss)



correct_prediction = tf.equal(tf.argmax(y_conv_logits, 1),

                              tf.argmax(y_ph, 1))



accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_labels_predicted=tf.argmax(y_conv_logits, 1)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    

    for i in range(1000):

        rand_int=np.random.choice(range(42000),100)

        x_train_batch=images_train[rand_int]

        y_train_batch=labels_train[rand_int]

        # checking accuracy for every 100 iteration

        if i % 100 == 0:

            train_accuracy = accuracy.eval(feed_dict={x_ph: x_train_batch, y_ph: y_train_batch, keep_prob: 1.0})

            #test_accuracy=accuracy.eval(feed_dict={x_ph: images_test, y_ph: labels_test, keep_prob: 1.0})

            print('step %d, training accuracy %g & testing accuracy g' % (i, train_accuracy))

            

        train_step.run(feed_dict={x_ph: x_train_batch, y_ph: y_train_batch, keep_prob: 0.5})



   # print('test accuracy %g' % accuracy.eval(feed_dict={x_ph: images_test, y_ph: labels_test, keep_prob: 1.0}))

    test_labels_predicted=sess.run(test_labels_predicted,feed_dict={x_ph:images_test,keep_prob:1.0})
#pd.crosstab(np.argmax(labels_test,axis=1),test_labels_predicted)
#t=np.argmax(labels_test,axis=1)!=test_labels_predicted
#len([i for i,x in enumerate(t) if x])
ind=2352

sample_image = images_test[ind] 

sample_image = np.array(sample_image, dtype='float')

pixels = sample_image.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()

print('predicted label:',test_labels_predicted[ind])

#print('real label:',np.argmax(labels_test[ind]))
submission = pd.DataFrame({

        "ImageId": list(test.index+1),

        "Label": test_labels_predicted

        })

submission.to_csv('submission.csv', index=False)