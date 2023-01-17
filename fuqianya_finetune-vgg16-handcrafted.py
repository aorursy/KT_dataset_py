#### Libraries

# Third-party libraries

import tensorflow.compat.v1 as tf

from PIL import Image

import numpy as np

import cv2, os, random

from datetime import datetime



tf.disable_v2_behavior()



def parse_fn(example):

    keys_to_features = {

        'label': tf.FixedLenFeature((), tf.int64),

        'img': tf.FixedLenFeature((), tf.string)

    }

    parsed = tf.parse_single_example(example, keys_to_features)

    images = tf.decode_raw(parsed["img"], tf.uint8)

    images = tf.reshape(images, [224, 224, 3])

    labels = tf.cast(parsed["label"], tf.int32)

    labels = tf.one_hot(labels, num_classes)

    return images, labels





def input_fn(path, batch_size):

    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.shuffle(buffer_size=batch_size * 2)

    dataset = dataset.repeat() # 注意这里的顺序

    dataset = dataset.map(map_func=parse_fn)

    dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.prefetch(batch_size)

    return dataset
class vgg16:

    def __init__(self, imgs, keep_prob, num_classes, skip_layer, weights=None):

        self.imgs = imgs

        self.NUM_CLASSES = num_classes

        self.SKIP_LAYER = skip_layer

        self.KEEP_PROB = keep_prob

        self.WEIGHTS_PATH = weights

        self.convlayers()

        self.fc_layers()

        #self.probs = tf.nn.softmax(self.fc3l)





    def convlayers(self):

        self.parameters = []



        # zero-mean input

        with tf.name_scope('preprocess') as scope:

            mean = tf.constant([187.4, 187.4, 187.4], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')

            images = self.imgs-mean



        # conv1_1

        with tf.name_scope('conv1_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),

                    trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv1_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv1_2

        with tf.name_scope('conv1_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv1_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # pool1

        self.pool1 = tf.nn.max_pool(self.conv1_2,

                               ksize=[1, 2, 2, 1],

                               strides=[1, 2, 2, 1],

                               padding='SAME',

                               name='pool1')



        # conv2_1

        with tf.name_scope('conv2_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv2_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv2_2

        with tf.name_scope('conv2_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv2_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # pool2

        self.pool2 = tf.nn.max_pool(self.conv2_2,

                               ksize=[1, 2, 2, 1],

                               strides=[1, 2, 2, 1],

                               padding='SAME',

                               name='pool2')



        # conv3_1

        with tf.name_scope('conv3_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv3_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv3_2

        with tf.name_scope('conv3_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv3_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv3_3

        with tf.name_scope('conv3_3') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv3_3 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # pool3

        self.pool3 = tf.nn.max_pool(self.conv3_3,

                               ksize=[1, 2, 2, 1],

                               strides=[1, 2, 2, 1],

                               padding='SAME',

                               name='pool3')



        # conv4_1

        with tf.name_scope('conv4_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv4_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv4_2

        with tf.name_scope('conv4_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv4_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv4_3

        with tf.name_scope('conv4_3') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv4_3 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # pool4

        self.pool4 = tf.nn.max_pool(self.conv4_3,

                               ksize=[1, 2, 2, 1],

                               strides=[1, 2, 2, 1],

                               padding='SAME',

                               name='pool4')



        # conv5_1

        with tf.name_scope('conv5_1') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv5_1 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv5_2

        with tf.name_scope('conv5_2') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv5_2 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # conv5_3

        with tf.name_scope('conv5_3') as scope:

            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,

                                                     stddev=1e-1), name='weights')

            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')

            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),

                                 trainable=True, name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.conv5_3 = tf.nn.relu(out, name=scope)

            self.parameters += [kernel, biases]



        # pool5

        self.pool5 = tf.nn.max_pool(self.conv5_3,

                               ksize=[1, 2, 2, 1],

                               strides=[1, 2, 2, 1],

                               padding='SAME',

                               name='pool4')



    def fc_layers(self):

        # fc1

        with tf.name_scope('fc1') as scope:

            shape = int(np.prod(self.pool5.get_shape()[1:]))

            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],

                                                         dtype=tf.float32,

                                                         stddev=1e-1), name='weights')

            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),

                                 trainable=True, name='biases')

            pool5_flat = tf.reshape(self.pool5, [-1, shape])

            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)

            self.fc1 = tf.nn.relu(fc1l)

            self.fc1 = dropout(self.fc1, self.KEEP_PROB)

            self.parameters += [fc1w, fc1b]



        # fc2

        with tf.name_scope('fc2') as scope:

            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],

                                                         dtype=tf.float32,

                                                         stddev=1e-1), name='weights')

            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),

                                 trainable=True, name='biases')

            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)

            self.fc2 = tf.nn.relu(fc2l)

            self.fc2 = dropout(self.fc2, self.KEEP_PROB)

            self.parameters += [fc2w, fc2b]



        # fc3

        with tf.name_scope('fc3') as scope:

            fc3w = tf.Variable(tf.truncated_normal([4096, self.NUM_CLASSES],

                                                         dtype=tf.float32,

                                                         stddev=1e-1), name='weights')

            fc3b = tf.Variable(tf.constant(1.0, shape=[self.NUM_CLASSES], dtype=tf.float32),

                                 trainable=True, name='biases')

            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

            self.parameters += [fc3w, fc3b]



    def load_weights(self, session):

        weights = np.load(self.WEIGHTS_PATH)

        keys = sorted(weights.keys())

        for i, k in enumerate(keys):

            if k not in self.SKIP_LAYER:

                print(i, k, np.shape(weights[k]))

                sess.run(self.parameters[i].assign(weights[k]))



def dropout(x, keep_prob):

    return tf.nn.dropout(x, keep_prob)
# path to storage DataSet

path_storage_train_1 = "../input/handcrafted-vgg16/train_1.tfrecorder"

path_storage_train_2 = "../input/handcrafted-vgg16-1/train_2.tfrecorder"

path_storage_test = "../input/handcrafted-vgg16/test.tfrecorder"



# Learning params

learning_rate = 0.0001

num_epochs = 2

batch_size = 64



# Network params

dropout_rate = 0.5

num_classes = 256

train_layers = ['fc8_W', 'fc8_b']



# num of data

num_train_1_data = 100000

num_train_2_data = 70000

num_test_data = 20000



x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

y = tf.placeholder(tf.float32, [None, num_classes])

keep_prob = tf.placeholder(tf.float32)



checkpoint_path = "/kaggle/working/model/"

acc_file = "/kaggle/working/model/acc.txt"

model_path = "/kaggle/input/finetune-vgg/model/"



if not os.path.isdir(checkpoint_path):

    os.makedirs(checkpoint_path)

    

# Initialize model

model = vgg16(x, keep_prob, num_classes, train_layers, weights='../input/vgg16-weights/vgg16_weights.npz')



# the output of the network

score = model.fc3l



# List of trainable variables of the layers we want to train

var_list = [v for v in tf.trainable_variables()]



# Op for calculating the loss

with tf.name_scope("loss"):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))



# We can use optimizer.minimize() straightforward

# But like this, we can grab the gradients and show them in Tensorboard



# Train op

with tf.name_scope("train"):

    gradients = tf.gradients(loss, var_list)

    gradients = list(zip(gradients, var_list))



    # Create optimizer and apply gradient descent to the trainable variables

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.apply_gradients(grads_and_vars=gradients)



with tf.name_scope('accuracy'):

    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Get the number of training/validation/ steps per epoch

train_1_batches_per_epoch = int(num_train_1_data / batch_size)

train_2_batches_per_epoch = int(num_train_2_data / batch_size)

test_batches_per_epoch = int(num_test_data / batch_size)



# create iterator for read data from the pipeline

Train_1_DataSet = input_fn(path_storage_train_1, batch_size)

train_1_iterator = Train_1_DataSet.make_one_shot_iterator()

train_1_image_batch, train_1_label_batch = train_1_iterator.get_next()



Train_2_DataSet = input_fn(path_storage_train_2, batch_size)

train_2_iterator = Train_2_DataSet.make_one_shot_iterator()

train_2_image_batch, train_2_label_batch = train_2_iterator.get_next()



Test_DataSet = input_fn(path_storage_test, batch_size)

test_iterator = Test_DataSet.make_one_shot_iterator()

test_image_batch, test_label_batch = test_iterator.get_next()



# Initialize on saver for store model checkpoints

saver = tf.train.Saver()



# mean of target dataset in BGR

# bgr_mean_train = tf.constant([237.81, 237.81, 237.81])

# bgr_mean_test = tf.constant([237.86, 237.86, 237.86])

bgr_mean_train = np.array([187.4, 187.4, 187.4], dtype=np.float32)

#bgr_mean_test = np.array([187.21, 187.21, 187.21], dtype=np.float32)



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    # Load the pretrained weights into non-trainable layer

    #model.load_weights(sess)

    model_file = tf.train.latest_checkpoint(model_path)

    saver.restore(sess, model_file)



    print("{} Start training...".format(datetime.now()))



    for epoch in range(num_epochs):

        f = open(acc_file, 'a')

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))



        for i in range(train_1_batches_per_epoch):

            train_x, train_y = sess.run([train_1_image_batch, train_1_label_batch])

            train_x = train_x - bgr_mean_train

            sess.run([train_op], feed_dict={

                x: train_x, y: train_y, keep_prob: dropout_rate

            })



        for i in range(train_2_batches_per_epoch):

            train_x, train_y = sess.run([train_2_image_batch, train_2_label_batch])

            train_x = train_x - bgr_mean_train

            sess.run([train_op], feed_dict={

                x: train_x, y: train_y, keep_prob: dropout_rate

            })



        # after each epoch, accuracy will be calculate on ValDataset

        test_accuracy = 0.0



        for j in range(test_batches_per_epoch):

            test_x, test_y = sess.run([test_image_batch, test_label_batch])

            test_x = test_x - bgr_mean_train

            accuracy_temp = sess.run([accuracy], feed_dict={

                x: test_x, y: test_y, keep_prob: 1.0

            })

            test_accuracy += accuracy_temp[0]

        test_accuracy = round(test_accuracy / test_batches_per_epoch, 5)

        info = "epoch:  " + str(epoch + 1) + "   val_accuracy: " + str(test_accuracy)

        print(info)

        f.write(info + '\n')

        f.close()

        checkpoint_name = os.path.join(checkpoint_path, "model_epoch" + str(epoch) + ".ckpt")

        saver.save(sess, checkpoint_name)