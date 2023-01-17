import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
dtrain  =  pd.read_csv('../input/train.csv')
dtest = pd.read_csv('../input/test.csv')
Xfull, yfull = dtrain.drop('label',axis =1), dtrain['label']
Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xfull, yfull, stratify = yfull, test_size=0.1)
ytrain = pd.get_dummies(ytrain)
yvalid = pd.get_dummies(yvalid)
Xtrain = Xtrain.values
Xvalid = Xvalid.values
ytrain = ytrain.values
yvalid = yvalid.values
Xtest = dtest.values
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
def get_next_batch(features, labels, j, batch_size):
    j = j%756
    return features[j*batch_size:j*batch_size + batch_size, :], labels[j*batch_size:j*batch_size + batch_size, :]
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
x_image = tf.reshape(x,[-1,28,28,1])
# Convolutional Layer 1 and Pool Layer 1
convo_1 = convolutional_layer(x_image,shape=[5,5,1,32])
convo_1_pooling = max_pool_2by2(convo_1)
# Convolutional Layer 2 and Pool Layer 2
convo_2 = convolutional_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)
# Flatten
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
# Dense 1
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))
# Dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred = normal_full_layer(full_one_dropout,10)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
steps = 7560

with tf.Session() as sess:
    
    sess.run(init)
    
    for i in range(steps):
        
        batch_x , batch_y = get_next_batch(Xtrain, ytrain, i, 50)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:Xvalid,y_true:yvalid,hold_prob:1.0}))
            print('\n')
            
    result = sess.run(y_pred,feed_dict={x:Xtest,hold_prob:0.5})
submission = pd.DataFrame(np.arange(1,28001), columns = ['ImageId'])
submission['Label'] = result.argmax(axis = 1)
submission.to_csv("evaluation_submission.csv",index=False)