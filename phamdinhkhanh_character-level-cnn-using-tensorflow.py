import pandas as pd

train = pd.read_csv('../input/train_batch.csv')

test = pd.read_csv('../input/test_batch.csv')

print('train shape:', train.shape)

print('test shape:', test.shape)



train.head()
import numpy as np

import tensorflow as tf

from sklearn.utils import shuffle

import pandas as pd

from math import ceil

import logging



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



# Lấy số lượng classes

def get_num_classes(data_path):

    return len(pd.read_csv(data_path, header = None, usecols = [0])[0].unique())
def create_dataset(data_path, alphabet="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",

                   max_length=1014, batch_size=128, is_training=True):

    label_converter = lambda x: int(x) - 1

    data = pd.read_csv(data_path, header = None, converters = {0: label_converter})

    num_iter = ceil(data.shape[0] / batch_size)

    if is_training:

        data = shuffle(data, random_state = 42)

    num_columns = data.shape[1]



    for idx in range(2, num_columns):

        # collect whole data comment into column 1

        data[1] += data[idx]

    # drop another column of data except 0, 1 and convert data into numpy

    data = data.drop([idx for idx in range(2, num_columns)], axis = 1).values

    alphabet = list(alphabet)

    # create the indentity matrice

    identity_mat = np.identity(len(alphabet))

    

    def generator():

        for row in data:

            label, text = row

            text = np.array([identity_mat[alphabet.index(i)] for i in list(str(text)) if i in alphabet], dtype = np.float32)

            if len(text) > max_length:

                text = text[:max_length]

            elif 0 < len(text) < max_length:

                text = np.concatenate((text, np.zeros((max_length-len(text), len(alphabet)), dtype = np.float32)))

            elif len(text) == 0:

                text = np.zeros((max_length, len(alphabet)), dtype = np.float32)

            # yield the embedding matrix at characters level

            # height is number of basic characters set (68)

            # width is maximum_length_characters in sentence (300)

            yield text.T, label

    return tf.data.Dataset.from_generator(generator, \

                                          (tf.float32, tf.int32), \

                                          ((len(alphabet), max_length), (None))).batch(batch_size), num_iter
alphabet = """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""

train_path = '../input/train_batch.csv'

test_path = '../input/test_batch.csv'

test_interval = 1

max_length = 1014

feature = 'small'

batch_size = 128

num_epochs = 2

lr = 1e-2

optimizer = 'sgd'

dropout = 0.5

log_path = 'tensorboard_khanh/char_level_cnn'

saved_path = 'trained_models_khanh'

es_min_delta = 0.

es_patience = 3

allow_soft_placement = True

log_device_placement = False
class Char_level_cnn():

    def __init__(self, batch_size = 128, num_classes = 14, feature = 'small',

                 kernel_size = [7, 7, 3, 3, 3, 3], padding = 'VALID'):

        # with python version 2

        # supper(char_level_cnn, self).__init__()

        self.batch_size = batch_size

        self.num_classes = num_classes

        if feature == 'small':

            self.num_filters = 256

            self.stddev_initialization = 0.05

            self.num_fully_connected_features = 1024

        else:

            self.num_filters = 1024

            self.stddev_initialization = 0.02

            self.num_fully_connected_features = 2048

        self.kernel_size = kernel_size

        self.padding = padding

        

    def forward(self, input, keep_prob):

        # expand dims in the deepest inside of tensor to transform shape from 3 to 4. 

        # For example from shape (3, 3, 4) to (3, 3, 4, 1)

        output = tf.expand_dims(input, -1)

        logging.info('input shape: {}'.format(output.get_shape()))

        output = self._create_conv(output, 

                                   [output.get_shape().as_list()[1], self.kernel_size[0], 1, self.num_filters],

                                   'conv1', 3)

        logging.info('output shape 1: {}'.format(output.get_shape()))

        # shape below are the shape of weight matrix, which have heights equal 1 and lengths equal kernel size. 

        # This mean that convolution product only impose on the length of each sentence regardless of characters dimension.

        output = self._create_conv(output, [1, self.kernel_size[1], self.num_filters, self.num_filters], 'conv2', 3)

        logging.info('output shape 2: {}'.format(output.get_shape()))

        output = self._create_conv(output, [1, self.kernel_size[2], self.num_filters, self.num_filters], 'conv3')

        logging.info('output shape 3: {}'.format(output.get_shape()))

        output = self._create_conv(output, [1, self.kernel_size[3], self.num_filters, self.num_filters], 'conv4')

        logging.info('output shape 4: {}'.format(output.get_shape()))

        output = self._create_conv(output, [1, self.kernel_size[4], self.num_filters, self.num_filters], 'conv5')

        logging.info('output shape 5: {}'.format(output.get_shape()))

        output = self._create_conv(output, [1, self.kernel_size[5], self.num_filters, self.num_filters], 'conv6', 3)

        logging.info('output shape 6: {}'.format(output.get_shape()))

        new_feature_size = int(self.num_filters*((input.get_shape().as_list()[2] - 96)/27))

        flatten = tf.reshape(output,[-1, new_feature_size])

        

        output = self._create_fc(flatten, [new_feature_size, self.num_fully_connected_features], 'fc1', keep_prob)

        logging.info('output shape 7: {}'.format(output.get_shape()))

        output = self._create_fc(output, [self.num_fully_connected_features, self.num_fully_connected_features], 'fc2',

                                 keep_prob)

        logging.info('output shape 8: {}'.format(output.get_shape()))

        output = self._create_fc(output, [self.num_fully_connected_features, self.num_classes], 'fc3')

        logging.info('output shape 9: {}'.format(output.get_shape()))

        return output

    

    def _create_conv(self, input, shape, name_scope, pool_size = None):

        '''

        shape: shape of weight

        name_scope: name of layer

        '''

        with tf.name_scope(name_scope):

            weight = self._initialize_weight(shape, self.stddev_initialization)

            bias = self._initialize_bias([shape[-1]])

            # shape NHWC

            # input shape: [batch, in_height, in_width, in_channel]

            # filter shape: [f_height, f_width, in_channel, out_channel]

            # stride shape: [1, stride, stride, 1]

            # output: [batch, out_height, out_width, out_channel]

            conv = tf.nn.conv2d(input=input, filter=weight, strides=[1, 1, 1, 1], padding=self.padding, name='conv')

            # tf.nn.bias_add will plus bias into whole element of ouput conv2d.

            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")

            if pool_size:

                return tf.nn.max_pool(value=activation, ksize=[1, 1, pool_size, 1], strides=[1, 1, pool_size, 1],

                                      padding=self.padding, name='maxpool')

            else:

                return activation

        

    def _create_fc(self, input, shape, name_scope, keep_prob = None):

        with tf.name_scope(name_scope):

            weight = self._initialize_weight(shape, self.stddev_initialization)

            bias = self._initialize_bias([shape[-1]])

            dense = tf.nn.bias_add(tf.matmul(input, weight), bias, name="dense")

            if keep_prob is not None:

                return tf.nn.dropout(dense, keep_prob, name="dropout")

            else:

                return dense

            

    def _initialize_weight(self, shape, stddev):

        return tf.Variable(tf.truncated_normal(shape = shape, stddev = stddev, dtype = tf.float32, name = 'weight'))

    

    def _initialize_bias(self, shape):

        return tf.Variable(tf.constant(0, shape = shape, dtype = tf.float32, name = 'bias'))

    

    def loss(self, logits, labels):

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    

    def accuracy(self, logits, labels):

        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), dtype = tf.float32))

    

    def confusion_matrix(self, logits, labels):

        return tf.confusion_matrix(tf.cast(labels, tf.int64), tf.argmax(logits, 1), num_classes = self.num_classes)
from math import pow

import shutil

import os

import numpy as np

import logging

import sys

from functools import reduce



# Create logger

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

# Create STDERR handler

handler = logging.StreamHandler(sys.stderr)

# Create formatter and add it to the handler

formatter = logging.Formatter('%(asctime)s : %(name)s: %(levelname)s : %(message)s')

handler.setFormatter(formatter)

# Set STDERR handler as the only handler 

logger.handlers = [handler]



def classifier(model, optimizer):

    with tf.Graph().as_default():

        batch_size = 128

        session_conf = tf.ConfigProto(

            allow_soft_placement = allow_soft_placement, 

            log_device_placement = log_device_placement

        )



        session_conf.gpu_options.allow_growth = True

        training_set, num_training_iters = create_dataset(train_path, alphabet, max_length, 

                                                         batch_size, True)

        test_set, num_test_iters = create_dataset(test_path, alphabet, 

                                                  max_length, batch_size, False)

        train_iterator = training_set.make_initializable_iterator()

        test_iterator = test_set.make_initializable_iterator()



        handle = tf.placeholder(tf.string, shape = [])

        keep_prob = tf.placeholder(tf.float32, name = 'dropout_prob')



        iterator = tf.data.Iterator.from_string_handle(handle, training_set.output_types, training_set.output_shapes)

        texts, labels = iterator.get_next()



        logits = model.forward(texts, keep_prob)

        loss = model.loss(logits, labels)

        # record loss into summary

        loss_summary = tf.summary.scalar('loss', loss)

        # record accuracy into summary

        accuracy = model.accuracy(logits, labels)

        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        # unstack to convert tensor rank R into rank R-1 depend on order is declared in axis

        batch_size = tf.unstack(tf.shape(texts))[0]

        confusion = model.confusion_matrix(logits, labels)

        global_step = tf.Variable(0, name='global_step', trainable = False)



        if optimizer == 'sgd':

            values = [lr]

            boundaries = []

            for i in range(1, 10):

                # decrease learning rate twice time in each boundary steps which show off in tf.train.piecewise_constant()

                values.append(lr/pow(2, i))

                # num_training_iters = number of records in dataset/batch_size

                boundaries.append(3 * num_training_iters * i)

            # create constant learning rate at boundaries and values.

            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

            # add momentum for learning_rate

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9)

        else:

            optimizer = tf.train.AdamOptimizer(lr)



        train_op = optimizer.minimize(loss, global_step = global_step)

        merged = tf.summary.merge([loss_summary, accuracy_summary])

        # init whole global variable

        init = tf.global_variables_initializer()

        # save training process

        saver = tf.train.Saver()

        if os.path.isdir(log_path):

            shutil.rmtree(log_path)

        os.makedirs(log_path)

        if os.path.isdir(saved_path):

            shutil.rmtree(saved_path)

        os.makedirs(saved_path)

        output_file = open(saved_path + os.sep + "logs.txt", "w")

        output_file.write("Model's parameters: {}".format('FLAG values dict'))

        best_loss = 1e5

        best_epoch = 0



        # create session

        with tf.Session(config = session_conf) as sess:

            # write into tensorboard

            train_writer = tf.summary.FileWriter(log_path + os.sep + 'train', sess.graph)

            test_writer = tf.summary.FileWriter(log_path + os.sep + 'test')

            sess.run(init)

            for epoch in range(num_epochs):

                sess.run(train_iterator.initializer)

                sess.run(test_iterator.initializer)

                train_handle = sess.run(train_iterator.string_handle())

                test_handle = sess.run(test_iterator.string_handle())

                train_iter = 0

                while True:

                    try:

                        _, tr_loss, tr_accuracy, summary, step = sess.run(

                            [train_op, loss, accuracy, merged, global_step],

                            feed_dict={handle: train_handle, keep_prob: dropout})

                        print("Epoch: {}/{}, Iteration: {}/{}, Loss: {}, Accuracy: {}".format(

                                epoch + 1,

                                num_epochs,

                                train_iter + 1,

                                num_training_iters,

                                tr_loss, tr_accuracy))

                        train_writer.add_summary(summary, step)

                        train_iter += 1

                    except (tf.errors.OutOfRangeError, StopIteration):

                        break



                if epoch % test_interval == 0:

                    loss_ls = []

                    loss_summary = tf.Summary()

                    accuracy_ls = []

                    accuracy_summary = tf.Summary()

                    confusion_matrix = np.zeros([num_classes, num_classes], np.int32)

                    num_samples = 0

                    while True:

                        try:

                            test_loss, test_accuracy, test_confusion, samples = sess.run(

                                [loss, accuracy, confusion, batch_size],

                                feed_dict={handle: test_handle, keep_prob: 1.0})

                            # test_loss are mean loss of each batch_size

                            loss_ls.append(test_loss * samples)

                            accuracy_ls.append(test_accuracy * samples)

                            confusion_matrix += test_confusion

                            num_samples += samples

                        except (tf.errors.OutOfRangeError, StopIteration):

                            break



                    mean_test_loss = sum(loss_ls) / num_samples

                    # add mean_test_loss into loss in summary log

                    loss_summary.value.add(tag='loss', simple_value=mean_test_loss)

                    test_writer.add_summary(loss_summary, epoch)

                    mean_test_accuracy = sum(accuracy_ls) / num_samples

                    # add mean_test_accuracy into accuracy in summary log

                    accuracy_summary.value.add(tag='accuracy', simple_value=mean_test_accuracy)

                    test_writer.add_summary(accuracy_summary, epoch)



                    # write into ouput_file logs

                    output_file.write(

                            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(

                                epoch + 1, num_epochs,

                                mean_test_loss,

                                mean_test_accuracy,

                                confusion_matrix))



                    print("Epoch: {}/{}, Final loss: {}, Final accuracy: {}".format(epoch + 1, num_epochs,

                                                                                    mean_test_loss,

                                                                                    mean_test_accuracy))



                    # save model at best epoch in case it gain best loss

                    if mean_test_loss + es_min_delta < best_loss:

                        best_loss = mean_test_loss

                        best_epoch = epoch

                        saver.save(sess, saved_path + os.sep + "char_level_cnn")



                    # stop training model when number of epoch exceed the best_epoch

                    if epoch - best_epoch > es_patience > 0:

                        print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, best_loss))

                        break



            output_file.close()
num_classes = get_num_classes('../input/test_batch.csv')

model = Char_level_cnn(batch_size = batch_size, num_classes = num_classes, feature = feature)

classifier(model, optimizer = optimizer)
class char_level_cnn_est():

    def __init__(self, num_classes = 14, feature = 'small',

                 kernel_size = [7, 7, 3, 3, 3, 3], padding = 'valid', keep_prob = [0.5, 0.5]):

            self.input = input

            self.num_classes = num_classes

            self.feature = feature

            self.kernel_size = kernel_size

            self.padding = padding

            if self.feature == 'small':

                self.num_filters = 256

                self.num_fully_connected_features = 1024

            else:

                self.num_filters = 1024

                self.num_fully_connected_features = 2048

            self.keep_prob = keep_prob

            

                

    def forward(self, input, keep_prob):

        input_layer = tf.expand_dims(input, -1)

        logging.info('input shape {}'.format(input_layer.get_shape()))

        

        conv1 = tf.layers.conv2d(

            inputs = input_layer,

            filters = 32, 

            kernel_size = [input_layer.get_shape().as_list()[1], self.kernel_size[0]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv1'

        )

        logging.info('conv1 shape {}'.format(conv1.get_shape()))

        

        max_pool1 = tf.layers.max_pooling2d(

            inputs = conv1,

            pool_size = [1, 3],

            strides = [1, 3],

            name = 'max_pool1'

        )

        logging.info('maxpool1 shape {}'.format(max_pool1.get_shape()))

        

        conv2 = tf.layers.conv2d(

            inputs = max_pool1,

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[1]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv2'

        )

        logging.info('conv2 shape {}'.format(conv2.get_shape()))

        

        max_pool2 = tf.layers.max_pooling2d(

            inputs = conv2,

            pool_size = [1, 3],

            strides = [1, 3],

            name = 'max_pool2'

        )

        logging.info('maxpool2 shape {}'.format(max_pool2.get_shape()))

        

        conv3 = tf.layers.conv2d(

            inputs = max_pool2,

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[2]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv3'

        )

        logging.info('conv3 shape {}'.format(conv3.get_shape()))

        

        conv4 = tf.layers.conv2d(

            inputs = conv3,

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[3]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv4'

        )

        logging.info('conv4 shape {}'.format(conv4.get_shape()))

        

        conv5 = tf.layers.conv2d(

            inputs = conv4,

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[4]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv5'

        )

        logging.info('conv5 shape {}'.format(conv5.get_shape()))

        

        conv6 = tf.layers.conv2d(

            inputs = conv5,

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[5]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv6'

        )

        logging.info('conv6 shape {}'.format(conv6.get_shape()))

        

        max_pool3 = tf.layers.max_pooling2d(

            inputs = conv6,

            pool_size = [1, 3],

            strides = [1, 3],

            name = 'max_pool3'

        )

        logging.info('maxpool3 shape {}'.format(max_pool3.get_shape()))

        

        shape = max_pool3.get_shape().as_list()

        # Calculate the shape size when flat tensor into vector

        new_size = reduce(lambda x, y: x*y, shape[1:]) 

        flat = tf.reshape(max_pool3, [-1, new_size])

        logging.info('flat shape {}'.format(flat.get_shape()))

        

        fc1 = tf.layers.dense(

            inputs = flat,

            units = self.num_fully_connected_features,

            name = 'fc1'

        )

        

        logging.info('fc1 shape {}'.format(fc1.get_shape()))

        if keep_prob is not None:

            fc1 = tf.layers.dropout(

                inputs = fc1, 

                rate = self.keep_prob[0],

                name = 'drop1'

            )

        

        fc2 = tf.layers.dense(

            inputs = fc1,

            units = self.num_fully_connected_features,

            name = 'fc2'

        )

        logging.info('fc2 shape {}'.format(fc2.get_shape()))

        

        if keep_prob is not None:

            fc2 = tf.layers.dropout(

                inputs = fc2, 

                rate = self.keep_prob[1],

                name = 'drop2'

            )

        

        logits = tf.layers.dense(

            inputs = fc2,

            units = self.num_classes,

            name = 'ouput'

        )

        logging.info('logits shape {}'.format(logits.get_shape()))

        

        return logits

    

    def loss(self, logits, labels):

        return tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels)

    

    def accuracy(self, logits, labels):

        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), dtype = tf.float32))

    

    def confusion_matrix(self, logits, labels):

        return tf.confusion_matrix(tf.cast(labels, tf.int64), tf.argmax(logits, 1), num_classes = self.num_classes)
num_classes = get_num_classes('../input/test_batch.csv')

model2 = char_level_cnn_est(num_classes = num_classes, feature = feature)

classifier(model2, optimizer = optimizer)
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Reshape



class char_level_cnn_keras():

    def __init__(self, num_classes = 14, feature = 'small',

                 kernel_size = [7, 7, 3, 3, 3, 3], padding = 'valid', keep_prob = [0.5, 0.5]):

            self.input = input

            self.num_classes = num_classes

            self.feature = feature

            self.kernel_size = kernel_size

            self.padding = padding

            if self.feature == 'small':

                self.num_filters = 256

                self.num_fully_connected_features = 1024

            else:

                self.num_filters = 1024

                self.num_fully_connected_features = 2048

            self.keep_prob = keep_prob

            

                

    def forward(self, input, keep_prob):

        input_layer = tf.expand_dims(input, -1)

        logging.info('input shape {}'.format(input_layer.get_shape()))

        

        conv1 = Conv2D(

            filters = 32, 

            kernel_size = [input_layer.get_shape().as_list()[1], self.kernel_size[0]],

            strides = [1, 1],

            padding = 'valid',

            activation = 'relu',

            name = 'conv1'

        )(input_layer)

        

        logging.info('conv1 shape {}'.format(conv1.get_shape()))

        

        max_pool1 = MaxPooling2D(

            pool_size = [1, 3],

            strides = [1, 3],

            name = 'max_pool1'

        )(conv1)

        logging.info('maxpool1 shape {}'.format(max_pool1.get_shape()))

        

        conv2 = Conv2D(

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[1]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv2'

        )(max_pool1)

        logging.info('conv2 shape {}'.format(conv2.get_shape()))

        

        max_pool2 = MaxPooling2D(

            pool_size = [1, 3],

            strides = [1, 3],

            name = 'max_pool2'

        )(conv2)

        logging.info('maxpool2 shape {}'.format(max_pool2.get_shape()))

        

        conv3 = Conv2D(

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[2]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv3'

        )(max_pool2)

        logging.info('conv3 shape {}'.format(conv3.get_shape()))

        

        conv4 = Conv2D(

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[3]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv4'

        )(conv3)

        logging.info('conv4 shape {}'.format(conv4.get_shape()))

        

        conv5 = Conv2D(

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[4]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv5'

        )(conv4)

        logging.info('conv5 shape {}'.format(conv5.get_shape()))

        

        conv6 = Conv2D(

            filters = self.num_filters,

            kernel_size = [1, self.kernel_size[5]],

            strides = [1, 1],

            padding = 'valid',

            activation = tf.nn.relu,

            name = 'conv6'

        )(conv5)

        logging.info('conv6 shape {}'.format(conv6.get_shape()))

        

        max_pool3 = MaxPooling2D(

            pool_size = [1, 3],

            strides = [1, 3],

            name = 'max_pool3'

        )(conv6)

        logging.info('maxpool3 shape {}'.format(max_pool3.get_shape()))

        

        shape = max_pool3.get_shape().as_list()

        # Calculate the shape size when flat tensor into vector

        new_size = reduce(lambda x, y: x*y, shape[1:]) 

        flat = Reshape([new_size])(max_pool3)

        logging.info('flat shape {}'.format(flat.get_shape()))

        

        fc1 = Dense(

            units = self.num_fully_connected_features,

            name = 'fc1'

        )(flat)

        

        logging.info('fc1 shape {}'.format(fc1.get_shape()))

        if keep_prob is not None:

            fc1 = Dropout(

                rate = self.keep_prob[0],

                name = 'drop1'

            )(fc1)

        

        fc2 = Dense(

            units = self.num_fully_connected_features,

            name = 'fc2'

        )(fc1)

        logging.info('fc2 shape {}'.format(fc2.get_shape()))

        

        if keep_prob is not None:

            fc2 = Dropout(

                rate = self.keep_prob[1],

                name = 'drop2'

            )(fc2)

        

        logits = Dense(

            units = self.num_classes,

            name = 'ouput'

        )(fc2)

        logging.info('logits shape {}'.format(logits.get_shape()))

        

        return logits

    

    def loss(self, logits, labels):

        return tf.losses.sparse_softmax_cross_entropy(logits = logits, labels = labels)

    

    def accuracy(self, logits, labels):

        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), dtype = tf.float32))

    

    def confusion_matrix(self, logits, labels):

        return tf.confusion_matrix(tf.cast(labels, tf.int64), tf.argmax(logits, 1), num_classes = self.num_classes)
num_classes = get_num_classes('../input/test_batch.csv')

model3 = char_level_cnn_keras(num_classes = num_classes, feature = feature)

classifier(model3, optimizer = optimizer)