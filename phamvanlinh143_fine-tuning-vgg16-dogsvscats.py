def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
print(os.listdir("../input"))

import glob
import numpy as np
from skimage import io, transform
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
import time
from random import shuffle
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import math
from datetime import timedelta
import tensorflow.contrib.slim as slim
# Any results you write to the current directory are saved as output.
ROWS = 150
COLS = 150
CHANNELS = 3

BATCH_SIZE = 64
LR_BASE = 1e-3
num_classes = 2
train = glob.glob('../input/dogs-vs-cats-redux-kernels-edition/train/*.jpg')
NUM_TRAINS = len(train)
split_factor = 0.1

# Shuffle training data
shuffle(train)

NUM_TRAININGS = int(NUM_TRAINS*(1 - split_factor))

TRAINING_DATA = train[:NUM_TRAININGS]

VALIDATION_DATA = train[NUM_TRAININGS:]

(len(TRAINING_DATA), len(VALIDATION_DATA))
# Reshape color images
def reshaped_image(image):
    return transform.resize(image,(ROWS, COLS, CHANNELS))
# One-hot Encoding for training data: cat [1 0], dog [0 1]
def process_Data(DIR_PATH):
    path_Imgs = []
    labels = []
    for pathImg in DIR_PATH:
        path_Imgs.append(pathImg)
        imgName = pathImg.split('/')[-1]
        label = imgName.split('.')[0]
        if label == 'cat':
            label = [1., 0.]
        else:
            label = [0., 1.]
        labels.append(label)
    
    return np.array(path_Imgs), np.array(labels)
def loadBatchImages(BatchDataPath):
    BatchImages = []
    for path in BatchDataPath:
        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/ 255.
        BatchImages.append(reshaped_image(img))
    return np.array(BatchImages)
TRAIN_PATH_DATAS, TRAIN_LABELS = process_Data(TRAINING_DATA)
VAL_PATH_DATAS, VALID_LABELS = process_Data(VALIDATION_DATA)
TRAIN_PATH_DATAS.shape, VAL_PATH_DATAS.shape
def loading_Data(DataPath, DataLabels, Num_Datas, current_index, batch_size=64):
    begin_index = current_index*batch_size
    end_index = -1
    if begin_index + batch_size < Num_Datas:
        end_index = begin_index + batch_size
    else:
        end_index = Num_Datas
    Batch_Imgs = loadBatchImages(DataPath[begin_index : end_index])
    Batch_Labels = DataLabels[begin_index : end_index]
    
    return Batch_Imgs, Batch_Labels
Batch_Imgs, Batch_Labels = loading_Data(TRAIN_PATH_DATAS, TRAIN_LABELS, NUM_TRAININGS, current_index = 0, batch_size=9)
Batch_Imgs.shape, Batch_Labels.shape
def plot_image(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
    
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
    
        ax.set_xlabel(xlabel)
    
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
Batch_Labels_cls = np.argmax(Batch_Labels, axis=1)
plot_image(Batch_Imgs, Batch_Labels_cls)
VGG_Weights = np.load('../input/vgg16-pretrained/vgg16.npy', encoding='latin1').item()
x_image = tf.placeholder(tf.float32, shape=[None, ROWS, COLS, CHANNELS], name='x_image')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)
def conv2d(layer, name, n_filters, trainable, k_size=3):
        return tf.layers.conv2d(layer, n_filters, kernel_size=(k_size, k_size),
                                activation=tf.nn.relu, padding='SAME', name=name, trainable=trainable,
                                kernel_initializer=tf.constant_initializer(VGG_Weights[name][0], dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(VGG_Weights[name][1], dtype=tf.float32),
                                use_bias=True)
net = x_image
# Block 1
net = conv2d(net, 'conv1_1', 64, trainable = False)
layer_conv1_1 = net
net = conv2d(net, 'conv1_2', 64, trainable = False)
layer_conv1_2 = net
net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='same')
# Block 2
net = conv2d(net, 'conv2_1', 128, trainable = False)
layer_conv2_1 = net
net = conv2d(net, 'conv2_2', 128, trainable = False)
layer_conv2_2 = net
net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='same')
# Block 3
net = conv2d(net, 'conv3_1', 256, trainable = False)
layer_conv3_1 = net
net = conv2d(net, 'conv3_2', 256, trainable = False)
layer_conv3_2 = net
net = conv2d(net, 'conv3_3', 256, trainable = False)
layer_conv3_3 = net
net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='same')
# Block 4
net = conv2d(net, 'conv4_1', 512, trainable = True)
layer_conv4_1 = net
net = conv2d(net, 'conv4_2', 512, trainable = True)
layer_conv4_2 = net
net = conv2d(net, 'conv4_3', 512, trainable = True)
layer_conv4_3 = net
net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='same')
# Block 5
net = conv2d(net, 'conv5_1', 512, trainable = True)
layer_conv5_1 = net
net = conv2d(net, 'conv5_2', 512, trainable = True)
layer_conv5_2 = net
net = conv2d(net, 'conv5_3', 512, trainable = True)
layer_conv5_3 = net
net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), padding='same')
layer_conv5_3, net
net = tf.contrib.layers.flatten(net)
net
net = tf.layers.dense(inputs=net, 
                      name='layer_fc1',
                      units=512, 
                      activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, 
                      name='layer_fc2',
                      units=64, 
                      activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, 
                      name='layer_fc_out',
                      units=num_classes, 
                      activation=None)
logits = net
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
y_pred_cls
def model_summary():
    model_vars = tf.global_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
model_summary()
def model_trainable_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
# summary trainable weights
model_trainable_summary()
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)
# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
# Classification Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(var)
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')
    return variable
weights_conv1_1 = get_weights_variable(layer_name='conv1_1')
weights_conv1_2 = get_weights_variable(layer_name='conv1_2')

weights_conv2_1 = get_weights_variable(layer_name='conv2_1')
weights_conv2_2 = get_weights_variable(layer_name='conv2_2')

weights_conv3_1 = get_weights_variable(layer_name='conv3_1')
weights_conv3_2 = get_weights_variable(layer_name='conv3_2')
weights_conv3_3 = get_weights_variable(layer_name='conv3_3')

weights_conv4_1 = get_weights_variable(layer_name='conv4_1')
weights_conv4_2 = get_weights_variable(layer_name='conv4_2')
weights_conv4_3 = get_weights_variable(layer_name='conv4_3')

weights_conv5_1 = get_weights_variable(layer_name='conv5_1')
weights_conv5_2 = get_weights_variable(layer_name='conv5_2')
weights_conv5_3 = get_weights_variable(layer_name='conv5_3')
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
test_batch_size = 128

def print_test_accuracy(show_example_error=False, show_confusion_matrix=False):
    num_test = len(VALIDATION_DATA)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)

        k = int(i // test_batch_size)

        x_batch, y_true_batch = loading_Data(VAL_PATH_DATAS, 
                                           VALID_LABELS, 
                                           len(VALIDATION_DATA), 
                                           current_index = k, 
                                           batch_size=test_batch_size)

        feed_dict = {x_image: x_batch, y_true: y_true_batch}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j
    cls_true = np.argmax(VALID_LABELS, axis=1)
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
train_batch_size = 128
total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()
    num_batchs = math.ceil(NUM_TRAININGS / train_batch_size)
    for epoch in range(total_iterations, num_iterations + total_iterations):
    
        for batch_step in range(num_batchs):
            x_batch, y_true_batch = loading_Data(TRAIN_PATH_DATAS, 
                                               TRAIN_LABELS, 
                                               NUM_TRAININGS, 
                                               current_index = batch_step, 
                                               batch_size=train_batch_size)

            feed_dict_train = {x_image: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict = feed_dict_train)
            if batch_step % 10 == 0:
                train_acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: {0:>6}, Training Accuracy : {1:>6.1%}"
                print(msg.format(batch_step + 1, train_acc))
        print('===============================')
        print('Epoch : {}'.format(epoch))
        print_test_accuracy()
        print('===============================')
    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
#print_test_accuracy()
optimize(num_iterations = 1)
#optimize(num_iterations = 9)
#print_test_accuracy()
#optimize(num_iterations = 10)
