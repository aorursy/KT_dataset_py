import tensorflow as tf

import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline
df = pd.read_csv('train.csv')

df_test = pd.read_csv('test.csv')
numbers_full = df.drop('label', axis = 1)

labels_full = df['label']
from sklearn.model_selection import train_test_split

numbers, X_test, labels, y_test = train_test_split(numbers_full, labels_full, test_size=0.05, random_state=42)
X_test = np.array(X_test, dtype = 'int32')

numbers = np.array(numbers, dtype = 'int32')

test = np.array(df_test, dtype = 'int32')

test1 = test[0:int(len(test)/4)]

test2 = test[int((len(test))/4):int((2*len(test)/4))]

test3 = test[int((2*len(test))/4):int((3*len(test)/4))]

test4 = test[int((3*len(test))/4):int((4*len(test)/4))]
nb_classes = 10

labels = np.eye(nb_classes)[labels.reshape(-1)]
number = numbers[0].reshape(28,28)

sns.heatmap(number)
labels[415]
def init_weights(shape):

    init_random_dist = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(init_random_dist)
def init_bias(shape):

    init_bias_vals = tf.constant(0.1, shape=shape)

    return tf.Variable(init_bias_vals)
def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2by2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                          strides=[1, 2, 2, 1], padding='SAME')
def convolutional_layer(input_x, shape):

    W = init_weights(shape)

    b = init_bias([shape[3]])

    return tf.nn.relu(conv2d(input_x, W) + b)
def normal_full_layer(input_layer, size):

    input_size = int(input_layer.get_shape()[1])

    W = init_weights([input_size, size])

    b = init_bias([size])

    return tf.matmul(input_layer, W) + b
x = tf.placeholder(tf.float32, shape = [None,784])
y_true = tf.placeholder(tf.float32, [None,10])
x_image = tf.reshape(x,[-1,28,28,1])
convo_1 = convolutional_layer(x_image,shape=[6,6,1,64])

convo_1_pooling = max_pool_2by2(convo_1)
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,64,128])

convo_2_pooling = max_pool_2by2(convo_2)
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*128])

full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
hold_prob = tf.placeholder(tf.float32)

full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,10)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
y_test.shape
steps = 5000

with tf.Session() as sess:

    sess.run(init)



    for step in range(steps):

        

        batch_x = numbers[step*100:(1+step)*100]

        batch_y = labels[step*100:(1+step)*100]

        

        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})

        

        # Test the Train Model

        if step%100 == 1:

            pred = tf.argmax(y_pred,1)

            predict_test = sess.run(pred, feed_dict = {x: X_test, hold_prob: 1})

            predictions_test = tf.cast(predict_test, tf.float32)

            predictions_test = predictions_test.eval()

            predictions_test = predictions_test.tolist()

            predictions_test = list(map(int,predictions_test))

            acc = sum(predictions_test == y_test) / 2100

            

            print(step)

            print(acc)

            

    pred = tf.argmax(y_pred,1)

    #test1

    predict1 = sess.run(pred, feed_dict={x: test1, hold_prob:1})

    predictions1 = tf.cast(predict1, tf.float32)

    predictions1 = predictions1.eval()

    

    #test2

    predict2 = sess.run(pred, feed_dict={x: test2, hold_prob:1})

    predictions2 = tf.cast(predict2, tf.float32)

    predictions2 = predictions2.eval()

    

    #test3

    predict3 = sess.run(pred, feed_dict={x: test3, hold_prob:1})

    predictions3 = tf.cast(predict3, tf.float32)

    predictions3 = predictions3.eval()



    #test4

    predict4 = sess.run(pred, feed_dict={x: test4, hold_prob:1})

    predictions4 = tf.cast(predict4, tf.float32)

    predictions4 = predictions4.eval()
predictions1
sns.heatmap(test[556].reshape(28,28))
y_pred.shape
batch_x.shape
test[0:100]
predictions = predictions1.tolist() + predictions2.tolist() + predictions3.tolist() + predictions4.tolist()

predictions = list(map(int,predictions))
predictions
sns.heatmap(test[0].reshape(28,28))
results = pd.DataFrame({'ImageID': df_test.index+1,'Label': predictions})

results = results.set_index('ImageID')
results
results.to_csv('results CNN v6')