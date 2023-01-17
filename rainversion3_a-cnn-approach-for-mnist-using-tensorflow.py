import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
array_train_data = np.array(train_data)
train_data_images = np.array(array_train_data[:,1:])
train_data_labels = np.array(array_train_data[:,0])

from sklearn.preprocessing import LabelBinarizer 
label_binarizer = LabelBinarizer()
label_binarizer.fit(train_data_labels)

train_data_onehot_labels = label_binarizer.transform(train_data_labels)
test_data = pd.read_csv('../input/test.csv')
test_data_images = np.array(test_data)
#Visulizing dataset
plt.figure(figsize=(16,6))
plt.suptitle("Visualizing dataset")
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(train_data_images[i].reshape(-1, int(np.sqrt(train_data_images[0].shape[0]))), cmap='Greys')
    plt.axis('off')
plt.show()
import tensorflow as tf

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

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
graph = tf.Graph()

with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, train_data_images.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, train_data_onehot_labels.shape[1]])
    
    #First Convolution Layer (1 layer -> 6 layers) with kernel of size 5x5
    W_conv1 = weight_variable([5,5,1,6])
    b_conv1 = bias_variable([6])
    
    x_image = tf.reshape(x, [-1,int(np.sqrt(train_data_images[0].shape[0])),int(np.sqrt(train_data_images[0].shape[0])),1])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    #Second Convolution Layer (6 layers -> 16 layers) with kernel of size 5x5
    W_conv2 = weight_variable([5,5,6,16])
    b_conv2 = bias_variable([16])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    #Densely connected Layer1
    W_fc1 = weight_variable([7*7*16, 120])
    b_fc1 = bias_variable([120])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    #Dropout Layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    #Densely connected Layer2
    W_fc_temp = weight_variable([120, 84])
    b_fc_temp = weight_variable([84])
    
    h_pool_temp_flat = tf.reshape(h_fc1_drop, [-1,120])
    h_fc_temp = tf.nn.relu(tf.matmul(h_pool_temp_flat, W_fc_temp) + b_fc_temp)
    
    #Dropout Layer
    h_fc_temp_drop = tf.nn.dropout(h_fc_temp, keep_prob)
    
    #Final prediction layer
    W_fc2 = weight_variable([84,10])
    b_fc2 = bias_variable([10])
    
    y_logits = tf.matmul(h_fc_temp_drop, W_fc2) + b_fc2
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_logits))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #Save variables of model in graph
    saver = tf.train.Saver()
with tf.Session(graph = graph) as sess:
    tf.global_variables_initializer().run()
    for i in range(10000):
        batch = next_batch(50, train_data_images, train_data_onehot_labels)
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            val_accuracy = accuracy.eval(feed_dict={x: train_data_images, y: train_data_onehot_labels, keep_prob: 1.0})
            print('step {0:5d}, training accuracy {1:1.2f}, validation accuracy {2:1.4f}'.format(i, train_accuracy, val_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1})

    #Final Model evaluation
    print('test accuracy {0:1.4f}'.format(accuracy.eval(feed_dict={x: train_data_images, y: train_data_onehot_labels, keep_prob: 1.0})))
    save_path = saver.save(sess, "model_weights_asg/model_nodrop.ckpt")
    print("Model saved in path: {}".format(save_path))
with tf.Session(graph=graph) as sess:
    saver.restore(sess, "model_weights_asg/model_nodrop.ckpt")
    preds = tf.argmax(tf.nn.softmax(y_logits), 1).eval(feed_dict={x: test_data_images, keep_prob: 1.0})
#Write submission file
with open('submission.csv', 'w') as file:
    file.write("ImageId" + ',' + "Label" + '\n')
    for i in range(preds.shape[0]):
        file.write(str(i+1) + ',' + str(preds[i]) + '\n')