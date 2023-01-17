# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!ls ../input/fruits-360_dataset_2018_02_08/fruits-360/
!ls
DIR = '../input/fruits-360_dataset_2018_02_08/fruits-360/'
TRAIN_DIR = DIR + 'Training/'
TEST_DIR = DIR + 'Validation/'

# checkpoints and tfrecords go here
SAVE_DIR = 'saves'
CP_DIR = SAVE_DIR + '/checkpoint'
import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
%matplotlib inline
import time
from PIL import Image

# tf.enable_eager_execution()
# resizing all the input images in to the following dims
HEIGHT = 64
WIDTH = 64
CHANNELS = 3

CLASSES = 60

from tqdm import tqdm
def encode_TFRecords(srcdir, recordname="data.tfrecords"):
    # converting our dataset into tf's recommended format
    labels = []
    sample_size = 0
    index = 0
    
    writer = tf.python_io.TFRecordWriter(recordname)
    
    for folder in tqdm(os.listdir(srcdir)):
        path = srcdir + folder
        labels.append(folder)
        index += 1
        
        for img in os.listdir(path):
            img_path = path + '/' + img
            im = Image.open(img_path)
            im = im.resize((HEIGHT, WIDTH))
            im_raw = im.tobytes()

            '''
            obviously the final tfrecord file contiaining raw images were way larger.
            so using the raw compressed images would keep the file size same. try using this and tf.image.decode_image with this. might work
            
            >> with open(img_path, 'rb') as im:
            >>    im_raw = im.read()
            '''
            
            # create example objects to write into the protobuf
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw]))
                    }
                )
            )
            
            writer.write(example.SerializeToString())
            sample_size += 1
            
    writer.close()
    return sample_size, labels
train_sample_size, labels = encode_TFRecords(TRAIN_DIR, 'train.tfrecords')
test_sample_size, _ = encode_TFRecords(TEST_DIR, 'test.tfrecords')
print('Serialized to tfrecords')
!ls && du -ah
def decode_TFRecords(recordname):
    file_queue = tf.train.string_input_producer([recordname])
    reader = tf.TFRecordReader()
    
    _, examples = reader.read(file_queue)
    features = tf.parse_single_example(
        examples,
        features = {
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [HEIGHT,WIDTH,CHANNELS])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    
    return img, label
def inputs(tfrecordname, batch_size, queuesize, shuffle=False):
    with tf.name_scope('input'):
        img, label = decode_TFRecords(tfrecordname)
        print('deserialized tfrecords')

        if shuffle:        
            images, labels = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size,
                                                    capacity=queuesize+batch_size,
                                                    min_after_dequeue=queuesize//4
                                                   )
        else:
            images, labels = tf.train.batch([img, label],
                                           batch_size=batch_size,
                                           capacity=batch_size*2)
    return images, labels
def conv_layer(X, filters, kernel_size=3, strides=1):
    '''
    thin wrapper for the conv layers in the CNN
    '''
    return tf.layers.conv2d(X, filters, kernel_size, strides, padding='same',
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer()
                           )

def max_pooling(X, pool_size=2, strides=2):
    '''
    thin wrapper for the max pooling layers
    '''
    return tf.layers.max_pooling2d(X, pool_size, strides, padding='same')
def conv_net_model(X):
    X = tf.reshape(X, shape=[-1, HEIGHT, WIDTH, CHANNELS])
    
    with tf.name_scope('conv1'):
        print('X: ', X)
        conv1 = conv_layer(X, 16)
        conv1 = max_pooling(conv1)
        conv1 = tf.nn.local_response_normalization(conv1)
        
    with tf.name_scope('conv2'):
        print('conv1: ', conv1)
        conv2 = conv_layer(conv1, 32)
        conv2 = max_pooling(conv2)
        conv2 = tf.nn.local_response_normalization(conv2)
        
    with tf.name_scope('conv3'):
        print('conv2: ', conv2)
        conv3 = conv_layer(conv2, 64)
        conv3 = max_pooling(conv3)
        conv3 = tf.nn.local_response_normalization(conv3) # normalization of layers to prevent vanishing gradients
        
    with tf.name_scope('conv4'):
        print('conv3: ', conv3)
        conv4 = conv_layer(conv3, 128)
        conv4 = max_pooling(conv4)
        conv4 = tf.nn.local_response_normalization(conv4)
        
#     with tf.name_scope('conv5'):
#         print('conv4: ', conv4)
#         conv5 = conv_layer(conv4, 512)
#         conv5 = max_pooling(conv5)
#         conv5 = tf.layers.dropout(conv5, 0.8)
        
    with tf.name_scope('fully_connected'):
        print('conv4: ', conv4)
#         fc = tf.reshape(conv5, shape=[-1, 512])
        fc = tf.reshape(conv4, shape=[-1, 4*4*128])
        print('fc reshape: ', fc)
#         fc = tf.layers.dense(fc, 2048, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        fc = tf.layers.dense(fc, 512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        fc = tf.layers.dropout(fc, 0.5)
        print('fc: ', fc)
        
    with tf.name_scope('output'):
        logits = tf.layers.dense(fc, CLASSES, kernel_initializer=tf.contrib.layers.xavier_initializer())
        print('out: ', logits)
        logits = tf.reshape(logits, shape=[-1, CLASSES])
        print('out: ', logits)
        
    return logits
def optimize(logits, Y):
    pred = tf.nn.softmax(logits, name='pred_op')
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name='loss_op')
    optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer_op')
    train_op = optimizer.minimize(loss_op, name='train_op')
    return pred, loss_op, train_op
def get_accuracy(pred, Y):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy_op')
    return accuracy
import time

# required configuration
lr_start = 0.001
lr_end = 0.0001
learning_rate = lr_end

CHECKPOINT_DIR = 'checkpoint/model.ckpt'
use_ckpt = True
checkpoint_step = 100

num_steps = 3000 # 3000 gave good acc
batch_size = 64
update_step = 5
display_step = 100
train_acc_target = 1
train_acc_target_cnt = train_sample_size / batch_size
def update_learning_rate(acc, lr):
    return lr - acc * lr * 0.9
def train_model(sess):
    acc_meet_target_cnt = 0
    start_time = time.time()
    accuracies = []
    losses = []
    
    for step in tqdm(range(1, num_steps+1)):
            if train_acc_target_cnt <= acc_meet_target_cnt:
                break
            
            oh_label = tf.one_hot(labels_train, CLASSES)
            batch_x, batch_y, y = sess.run([images_train, oh_label, labels_train])
            
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            
            if step % update_step == 0 or step == 1:
                loss, acc, pred = sess.run([loss_op, acc_op, pred_op], feed_dict={X: batch_x, Y: batch_y})
                learning_rate = update_learning_rate(acc, lr_start)
                
                if train_acc_target <= acc:
                    acc_meet_target_cnt += 1
                else:
                    acc_meet_target_cnt = 0
                    
                end_time = time.time()
                
            if step % display_step == 0 or step == 1:
                accuracies.append(acc) # for plotting purposes
                losses.append(loss)
                
                print("{:.4f}".format(end_time - start_time)+ "s,", "step " + str(step) + ", minibatch loss = " +
                      "{:.4f}".format(loss) + ", training accuracy = " +
                      "{:.4f}".format(acc) , ", acc_meet_target_cnt = " + "{:.4f}".format(acc_meet_target_cnt))
            
            start_time = end_time
            
            if use_ckpt:
                if step % checkpoint_step == 0 or train_acc_target_cnt <= acc_meet_target_cnt:
                    print('saving session...')
                    saver.save(sess, CHECKPOINT_DIR)
                    
    # plot the final loss graph
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('loss')
    plt.subplot(2, 1, 2)
    plt.plot(accuracies)
    plt.title('accuracy')
num_inputs = HEIGHT*WIDTH*CHANNELS
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, CHANNELS], name='X')
Y = tf.placeholder(tf.float32, [None, CLASSES], name='Y')
with tf.Session() as sess:    
    images_train, labels_train = inputs('train.tfrecords', 64, train_sample_size, True)
    logits = conv_net_model(X)
    pred_op, loss_op, train_op = optimize(logits, Y)
    acc_op = get_accuracy(pred_op, Y)
    
    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)
        
    saver = tf.train.Saver()
                
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    train_model(sess)
    
    coord.request_stop()
    coord.join(threads)

!du -ah -d 1
!ls checkpoint
!ls
# dropout = 1

def test_model(sess):
    samples_untested = test_sample_size
    acc_sum = 0
    test_sample_sum = 0
    
    with tf.Graph().as_default():
        if use_ckpt:
            ckpt = tf.train.get_checkpoint_state('checkpoint/')

            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.import_meta_graph(CHECKPOINT_DIR + '.meta')
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('restored')
            else:
                pass

            while samples_untested > 0:
                oh_label = tf.one_hot(labels_test, CLASSES)
                batch_x, batch_y, y = sess.run([images_test, oh_label, labels_test])
                batch_size = len(y)

                acc_op = sess.graph.get_tensor_by_name('accuracy_op:0')
                X = sess.graph.get_tensor_by_name('X:0')
                Y = sess.graph.get_tensor_by_name('Y:0')

                acc = sess.run(acc_op, feed_dict={X:batch_x, Y:batch_y})
                print(acc)
                acc_sum += acc * batch_size

                samples_untested -= batch_size
                test_sample_sum += batch_size
            
    print('Testing accuracy = ', acc_sum / test_sample_sum)
with tf.Session() as sess:
    images_test, labels_test = inputs('test.tfrecords', 64, test_sample_size, True)
    
    sess.run(tf.local_variables_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    test_model(sess)
    
    coord.request_stop()
    coord.join(threads)
