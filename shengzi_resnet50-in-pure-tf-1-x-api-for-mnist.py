import numpy as np

import tensorflow.compat.v1 as tf

import csv

import pandas

import matplotlib.pyplot as plt

from random import randint



path_train = '/kaggle/input/digit-recognizer/train.csv'

path_test = '/kaggle/input/digit-recognizer/test.csv'



tf.disable_eager_execution()
train_num = 42000

raw = pandas.read_csv(path_train)

X = np.array(raw.iloc[0:, 1:785])

y = np.array(raw.iloc[0:, 0:1])

X = (X*(1/255)).reshape((42000, 28, 28, 1))



y_bi=(y==0)

for ybi in range(1,10):

    y_bi = np.concatenate((y_bi,(y==ybi)),axis=1)



X_train = X[0:train_num, ...]

y_train = y_bi[0:train_num, ...]

X_valid = X[train_num: , ...]

y_valid = y_bi[train_num:, ...]



print('training set shape: ',X_train.shape,y_train.shape)

print('cross validation set shape: ',X_valid.shape,y_valid.shape)



test = pandas.read_csv(path_test)

testset = np.array(test.iloc[0:, 0:784])

test_img = (testset*(1/255)).reshape((28000, 28, 28, 1))

print('test set shape: ', test_img.shape)
class ResNet50(object):

    def __init__(self, inputs, num_classes=10, is_training=True, scope="resnet50"):

        self.inputs = inputs

        self.is_training = is_training

        self.num_classes = num_classes

        self.logits = None

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            with tf.variable_scope("Conv1"):

                net = conv2d(inputs, 64, 8, 2, scope="conv1")        # ---> [batch,14,14,64]

                net = batch_norm(net, is_training=is_training, scope="bn1")

                net = tf.nn.relu(net)

            print("Layer_input   ", net.shape)

            for i in range(1, 4):

                net = self._bottleneck(net, 64, 256, stride=1,

                                       is_training=is_training,

                                       scope="Conv2_btlNeck%d" % i)  # ---> [batch,14,14,256]

            print("Layer_Block1  ", net.shape)

            for i in range(1, 5):

                net = self._bottleneck(net, 128, 512,

                                       is_training=is_training,

                                       scope="Conv3_btlNeck%d" % i)  # ---> [batch,7,7,512]

            print("Layer_Block2  ",net.shape)

            for i in range(1, 6):

                net = self._bottleneck(net, 256, 1024,stride=1,

                                       is_training=is_training,

                                       scope="Conv4_btlNeck%d" % i)  # ---> [batch,7,7,1024]

            print("Layer_Block2  ",net.shape)

            for i in range(1, 4):

                net = self._bottleneck(net, 512, 2048,stride=1,

                                       is_training=is_training,

                                       scope="Conv5_btlNeck%d" % i)  # ---> [batch,7,7,2048]

            print("Layer_Block4  ",net.shape)

            net = avg_pool(net, 7, scope="avgpool5")  # ---> [batch,1,1,2048]

            print("Layer_AvgPool ", net.shape)

            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")     # ---> [batch,784]

            self.logits = fc(net, self.num_classes, scope="fc6") # ---> [batch,num_classes]

            print("logits_shape  ",self.logits.shape)

            self.sflogits = tf.nn.softmax(self.logits)

            

    def _bottleneck(self, x, h_out, n_out, stride=None, is_training=True, scope="bottleneck"):

        """a ResNet bottleneck unit"""

        n_in = x.get_shape()[-1]

        if stride is None:

            stride = 1 if n_in == n_out else 2

        with tf.variable_scope(scope):

            h = conv2d(x, h_out, 1, stride=stride, scope="conv_1")

            h = batch_norm(h, is_training=is_training, scope="bn1")

            h = tf.nn.relu(h)

            h = conv2d(h, h_out, 3, stride=1, scope="conv_2")

            h = batch_norm(h, is_training=is_training, scope="bn2")

            h = tf.nn.relu(h)

            h = conv2d(h, n_out, 1, stride=1, scope="conv_3")

            h = batch_norm(h, is_training=is_training, scope="bn3")

            if n_in != n_out:

                shortcut = conv2d(x, n_out, 1, stride=stride, scope="shortcut")

                shortcut = batch_norm(shortcut, is_training=is_training, scope="shortcut")

            else:

                shortcut = x

            return tf.nn.relu(shortcut + h)

def create_var(name, shape, initializer, trainable=True):

    return tf.get_variable(name,

                             shape=shape,

                             dtype=tf.float32,

                             initializer=initializer,

                             trainable=trainable)

def conv2d(x, num_outputs, kernel_size, stride=1, scope="conv2d"):

    conv2d_initializer = tf.random_normal_initializer

    num_inputs = x.get_shape()[-1]

    with tf.variable_scope(scope):

        kernel = create_var("kernel",

                            [kernel_size, kernel_size, num_inputs, num_outputs],

                            conv2d_initializer())

        return tf.nn.conv2d(x, kernel,

                            strides=[1, stride, stride, 1],

                            padding="SAME")

def fc(x, num_outputs, scope="fc"):

    fc_initializer = tf.random_normal_initializer

    num_inputs = x.get_shape()[-1]

    with tf.variable_scope(scope):

        weight = create_var("weight", [num_inputs, num_outputs], fc_initializer())

        bias = create_var("bias", [num_outputs, ], tf.zeros_initializer())

        return tf.nn.xw_plus_b(x, weight, bias)

def max_pool(x, pool_size, stride, scope):

    with tf.variable_scope(scope):

        return tf.nn.max_pool2d(x,

                                [1, pool_size, pool_size, 1],

                                [1, stride, stride, 1],

                                padding="SAME")

def avg_pool(x, pool_size, scope):

    with tf.variable_scope(scope):

        return tf.nn.avg_pool2d(x,

                                [1, pool_size, pool_size, 1],

                                strides=[1, pool_size, pool_size, 1],

                                padding="VALID")

def batch_norm(x, is_training, scope):

    with tf.variable_scope(scope):

        return tf.layers.batch_normalization(inputs=x, training=is_training, fused=True)
def Fitting(NetName, xset, yset, B_SZ, epochs, rate_base=0.001, From0=False, CVnum=512):

    SavePath = "./SAVED/"

    DATAnum = xset.shape[0]

    STEPS = ((DATAnum-CVnum)//B_SZ)+1

    RATE_BASE = rate_base



    RATE_DECAY = 0.99

    ME_DECAY = 0.99



    tf.reset_default_graph()

    g1 = tf.Graph()

    with g1.as_default():

        x1 = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])

        y1 = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        resnet1 = ResNet50(x1, 10, True, scope=NetName)

        with tf.name_scope("Fitting"):

            global_step = tf.Variable(0, trainable=False)



            y = resnet1.logits

          #  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=y)) + 0.00001 * tf.add_n(

          #      [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'batch_normalization' not in v.name])

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=y))



            learning_rate = tf.train.exponential_decay(RATE_BASE, global_step, STEPS, RATE_DECAY, staircase=False)

            ema = tf.train.ExponentialMovingAverage(ME_DECAY, global_step)

            ema_op = ema.apply(tf.trainable_variables())

            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)



            with tf.control_dependencies(update_op):

                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

            with tf.control_dependencies([train_step, ema_op]):

                train_op = tf.no_op(name='train')



            var_list = tf.global_variables()

            

            cv_pre = tf.cast(tf.equal(tf.argmax(resnet1.sflogits, axis=-1), tf.argmax(y1, axis=-1)), tf.float32)

            cv_acc = tf.reduce_mean(cv_pre)



        saver = tf.train.Saver(var_list=var_list, max_to_keep=2)

        #tf.summary.FileWriter(r"./GRAPH/", g1)



        gpu_op = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_op)) as sess:

            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(SavePath)

            if (From0 is False) and ckpt and ckpt.model_checkpoint_path:

                saver.restore(sess, save_path=ckpt.model_checkpoint_path)

            

            for ep in range(epochs):

                permutation = np.random.permutation(STEPS)

                for i in permutation:

                    _, step = sess.run([train_op, global_step], 

                                        feed_dict={x1: xset[i*B_SZ:(i+1)*B_SZ, :, :, :], 

                                                   y1: yset[i*B_SZ:(i+1)*B_SZ, :]})             

                saver.save(sess, save_path=SavePath + NetName + ".ckpt")

                acc, loss_value = sess.run([cv_acc, loss], 

                                           feed_dict={x1: xset[DATAnum-CVnum:, ...],

                                                      y1: yset[DATAnum-CVnum:, :]})

                print("%d epochs, cv loss is %.4f." % (ep+1, loss_value))

                print("%d epochs, cv accuracy is %.4f." % (ep+1, acc))





def Test(NetName, xset, yset, B_SZ=512):

    SavePath = "./SAVED/"

    DATAnum = xset.shape[0]



    tf.reset_default_graph()

    g1 = tf.Graph()

    with g1.as_default():

        x1 = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])

        resnet1 = ResNet50(x1, 10, False, scope=NetName)

      

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=2)

        gpu_op = tf.GPUOptions(allow_growth=True)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_op)) as sess:

            init_op = tf.global_variables_initializer()

            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(SavePath)

            if ckpt and ckpt.model_checkpoint_path:

                saver.restore(sess, save_path=ckpt.model_checkpoint_path)



            pred = []

            for i in range((DATAnum//B_SZ)+1):

                cur_pred = sess.run(tf.argmax(resnet1.sflogits, axis=-1), 

                                feed_dict={x1: xset[i*B_SZ:(i+1)*B_SZ, :, :, :]})

                pred.append(cur_pred)

            pred = np.concatenate(pred, axis=0)

            

            if not yset is None:

                acc = np.sum((pred==np.argmax(yset, axis=-1)))/DATAnum

                print("validation acc is: %.4f"%acc)            

            else:

                inde = np.array(range(1,pred.shape[0]+1))

                out = np.concatenate([inde[:, np.newaxis], pred[:, np.newaxis]], axis=-1)

                head = np.array([["ImageId","Label"]])

                out = np.concatenate([head, np.array(out, dtype=np.str)], axis=0)

                np.savetxt('/kaggle/working/submission_ResNet50_no_pre.csv', out, fmt='%s', delimiter=',')

                print("test prediction saved\n")
Fitting(NetName='ResNet50', xset=X_train, yset=y_train, B_SZ=32, epochs=20, rate_base=0.001, From0=True)

Fitting(NetName='ResNet50', xset=X_train, yset=y_train, B_SZ=16, epochs=20, rate_base=0.0003, From0=False)

Fitting(NetName='ResNet50', xset=X_train, yset=y_train, B_SZ=8, epochs=10, rate_base=0.0003, From0=False)
Test(NetName='ResNet50', xset=X_train, yset=y_train, B_SZ=512)
Test(NetName='ResNet50', xset=test_img, yset=None, B_SZ=512)