import struct
import numpy as np
from collections import namedtuple
import pathlib
import gzip

Dataset = namedtuple('Dataset_fields', ('name', 'num', 'shape'))
basepath = pathlib.Path('../input')
file_name = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    't10k_img': 't10k-images-idx3-ubyte.gz',
    't10k_label': 't10k-labels-idx1-ubyte.gz'
}

data_dic = dict.fromkeys(file_name.keys())
fields_dic = dict.fromkeys(file_name.keys())
class_num = 10
for key, value in file_name.items():
    value = pathlib.Path(value)
    with gzip.open(basepath / value, 'rb') as f:
        data = f.read()
    mdata = memoryview(data)
    if key.endswith('img'):
        meta_fmt = '>4s3i' # 这里为了方便我们将magic number按照字节解析而非int
        size = struct.calcsize(meta_fmt)
        magic, images, rows, columns = struct.unpack(meta_fmt, mdata[:size])
        data_dic[key] = np.array(bytearray(mdata[size:])).reshape(images, rows, columns)
        fields_dic[key] = Dataset(key, images, (images, rows, columns, 1))
    else:
        meta_fmt = '>4si'
        size = struct.calcsize(meta_fmt)
        magic, images = struct.unpack(meta_fmt, mdata[:size])
        rows = columns = 1
        data_dic[key] = np.array(bytearray(mdata[size:])).reshape(images, rows)
        fields_dic[key] = Dataset(key, images, (images, rows))
BATCH_ROUND = 600
training_meta = fields_dic.get('train_img')
image_num, rows, columns, channels = training_meta.shape
test_num = 10000
train_images = data_dic.get('train_img').reshape(image_num, rows, columns, channels)
train_labels = data_dic.get('train_label').reshape(image_num)
label_meta = fields_dic.get('train_label')
BATCH_SIZE = image_num // BATCH_ROUND
test_images = data_dic.get('t10k_img').reshape(test_num, rows, columns, channels)
test_labels = data_dic.get('t10k_label').reshape(test_num)
test_meta = fields_dic.get('t10k_img')
numbers = rows*columns
import tensorflow as tf
from collections import namedtuple
import matplotlib.pyplot as plt

ConvLayers = namedtuple('ConvLayer', ('layer', 'type', 'kernel', 'strides', 'number', 'channels', 'stddev', 'bias'))
PoolLayers = namedtuple('PoolLayer', ('layer', 'type', 'ksize', 'strides'))
# Local Response Normalizations
LRNLayers = namedtuple('LRNLayer', ('layer', 'type', 'radius', 'bias', 'alpha', 'beta'))
FCLayers = namedtuple('FCLayer', ('layer', 'type', 'shape', 'stddev', 'bias', 'regularizer', 'regularizer_weight', 'activation'))

tf.reset_default_graph()

REGULARIZATION_RATE = 0.03
regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

batch_size = 128
batch_num = image_num // batch_size

ipt = tf.placeholder(tf.float32, shape=(batch_size, numbers))
ipt = tf.reshape(ipt, [batch_size, rows, columns, channels])
labels = tf.placeholder(tf.int32, (batch_size,))

def gen_weights(scope_name, shape, bshape, stddev=.1, bias=.1, regularizer=regularizer, wl=None):
    '''
    bshape: shape for bias
    '''
    weight_init = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
    bias_init = tf.constant_initializer(bias)
    weights = tf.get_variable('{}-weights'.format(scope_name), shape=shape, initializer=weight_init)
    biases = tf.get_variable('{}-biases'.format(scope_name), shape=bshape, initializer=bias_init)
    if regularizer is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(weights), wl, name='weights-loss')
        tf.add_to_collection('losses', weights_loss)
    return weights, biases

def build_layer(ilayer, olayer):
    scope_name = '{type}-{layer}'.format(type=olayer.type, layer=olayer.layer)
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        if olayer.type == 'conv':
            weights_shape = [*olayer.kernel, olayer.channels, olayer.number]
            weights, biases = gen_weights(
                scope_name,
                weights_shape,
                bshape=olayer.number,
                regularizer=None,
                stddev=olayer.stddev,
                bias=olayer.bias
            )
            clayer = tf.nn.conv2d(ilayer, weights, strides=olayer.strides, padding='SAME')
            clayer = tf.nn.relu(tf.nn.bias_add(clayer, biases))
        elif olayer.type == 'pool':
            clayer = tf.nn.max_pool(ilayer, ksize=olayer.ksize, strides=olayer.strides, padding='SAME')
        elif olayer.type == 'lrn':
            clayer = tf.nn.lrn(ilayer, bias=olayer.bias, alpha=olayer.alpha, beta=olayer.beta)
        elif olayer.type == 'fc':
            reshape = tf.reshape(ilayer, [batch_size, -1])
            dim = reshape.get_shape()[-1].value
            weights_shape = (dim, olayer.shape)
            weights, biases = gen_weights(
                scope_name,
                weights_shape,
                bshape=[olayer.shape],
                stddev=olayer.stddev,
                bias=olayer.bias,
                regularizer=olayer.regularizer,
                wl=olayer.regularizer_weight
            )
            if olayer.activation:
                clayer = tf.nn.relu(tf.matmul(reshape, weights) + biases)
            else:
                clayer = tf.add(tf.matmul(reshape, weights), biases)
        return clayer

clayer1 = ConvLayers(layer=1, type='conv', kernel=(5, 5), strides=(1, 1, 1, 1), number=64, channels=1, stddev=5e-2, bias=0.0)
clayer2 = ConvLayers(layer=2, type='conv', kernel=(3, 3), strides=(1, 1, 1, 1), number=128, channels=64, stddev=5e-2, bias=0.1)
player1 = PoolLayers(layer=1, type='pool', ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1))
player2 = PoolLayers(layer=2, type='pool', ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1))
llayer1 = LRNLayers(layer=1, type='lrn', radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
llayer2 = LRNLayers(layer=2, type='lrn', radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)
flayer1 = FCLayers(layer=1, type='fc', shape=384, stddev=0.04, bias=0.1, regularizer=regularizer, regularizer_weight=0.004, activation=True)
flayer2 = FCLayers(layer=2, type='fc', shape=192, stddev=0.04, bias=0.1, regularizer=regularizer, regularizer_weight=0.004, activation=True)
flayer3 = FCLayers(layer=3, type='fc', shape=10, stddev=1/192.0, bias=0.0, regularizer=None, regularizer_weight=0.0, activation=False)

layers = [clayer1, player1, llayer1, clayer2, llayer2, player2, flayer1, flayer2, flayer3]
layer = ipt
for nlayer in layers:
    layer = build_layer(layer, nlayer)

labels = tf.cast(labels, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=layer,
    labels=labels,
)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
tf.add_to_collection('losses', cross_entropy_mean)
loss = tf.add_n(tf.get_collection('losses'))

learning_rate = 1e-3
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
top_k = tf.nn.in_top_k(layer, labels, 1)
steps = 3000
loss_val = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for s in range(steps):
    batch = batch_size * (s % batch_num)
    image_batch = train_images[batch:batch + batch_size]
    label_batch = train_labels[batch:batch + batch_size]
    _, closs = sess.run([train, loss], {ipt: image_batch, labels: label_batch})
    loss_val.append(closs)
    if s % 100 == 0:
        print('Round: {}, average_loss: {}, current_loss: {}'.format(s, sum(loss_val)/(s + 1), closs))
        
plt.figure()
plt.plot(loss_val, 'r')
plt.title('Loss curve')
plt.show()

# Test
true_count_test = 0
true_count_train = 0
step = 0
batch_size = 128
batch_num = test_num // batch_size
tbatch_num = image_num // batch_size
while (step < batch_num):
    batch = step * batch_size
    test_batch = test_images[batch:batch + batch_size]
    test_label = test_labels[batch:batch + batch_size]
    predictions = sess.run(top_k, {ipt: test_batch, labels: test_label})
    true_count_test += np.sum(predictions)
    step += 1
print('Testing Precision: {:.2f}%'.format(true_count_test / test_num * 100))
step = 0
while (step < tbatch_num):
    batch = step * batch_size
    train_batch = train_images[batch:batch + batch_size]
    train_label = train_labels[batch:batch + batch_size]
    predictions = sess.run(top_k, {ipt: train_batch, labels: train_label})
    true_count_train += np.sum(predictions)
    step += 1
    
print('Training Precision: {:.2f}%'.format(true_count_train / image_num * 100))