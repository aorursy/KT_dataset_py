import numpy as np

import tensorflow as tf

from scipy.sparse import vstack

from functools import partial

from sklearn.datasets import load_svmlight_files



from shutil import copyfile

copyfile(src = "../input/libscript/flip_gradient.py", dst = "../working/flip_gradient.py")

from flip_gradient import flip_gradient



def load_amazon(source_name, target_name, data_folder=None, verbose=False):

    if data_folder is None:

        data_folder = './data/'

    source_file = data_folder + source_name + '_train.svmlight'

    target_file = data_folder + target_name + '_train.svmlight'

    test_file = data_folder + target_name + '_test.svmlight'

    if verbose:

        print('source file:', source_file)

        print('target file:', target_file)

        print('test file:  ', test_file)



    xs, ys, xt, yt, xt_test, yt_test = load_svmlight_files([source_file, target_file, test_file])

    ys, yt, yt_test = (np.array((y + 1) / 2, dtype=int) for y in (ys, yt, yt_test))



    return xs, ys, xt, yt, xt_test, yt_test



data_folder = '../input/amazontransfer/'

source_name = 'kitchen'

target_name = 'books'

xs, ys, xt, yt, xt_test, yt_test = load_amazon(source_name, target_name, data_folder, verbose=True)

# MMD Comutation



def compute_pairwise_distances(x, y):

    if not len(x.get_shape()) == len(y.get_shape()) == 2:

        raise ValueError('Both inputs should be matrices.')

    if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:

        raise ValueError('The number of features should be the same.')



    norm = lambda x: tf.reduce_sum(tf.square(x), 1)

    return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))



def gaussian_kernel_matrix(x, y, sigmas):

    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))



def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):

    cost = tf.reduce_mean(kernel(x, x))

    cost += tf.reduce_mean(kernel(y, y))

    cost -= 2 * tf.reduce_mean(kernel(x, y))

    cost = tf.where(cost > 0, cost, 0, name='value')

    return cost
def csr_2_sparse_tensor_tuple(csr_matrix):

    coo_matrix = csr_matrix.tocoo()

    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))

    values = coo_matrix.data

    shape = csr_matrix.shape

    return indices, values, shape



def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense'):

    with tf.name_scope(layer_name):

        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')

        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')

        if input_type == 'sparse':

            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)

        else:

            activations = act(tf.matmul(input_tensor, weight) + bias)

        return activations



def shuffle_aligned_list(data):

    num = data[0].shape[0]

    shuffle_index = np.random.permutation(num)

    return [d[shuffle_index] for d in data]



def batch_generator(data, batch_size, shuffle=True):

    if shuffle:

        data = shuffle_aligned_list(data)

    batch_count = 0

    while True:

        if batch_count * batch_size + batch_size >= data[0].shape[0]:

            batch_count = 0

            if shuffle:

                data = shuffle_aligned_list(data)

        start = batch_count * batch_size

        end = start + batch_size

        batch_count += 1

        yield [d[start:end] for d in data]
# mmd coefficient

mmd_param = 0

# dann coefficient

grl_lambda = 1

dann_param = 0.1



batch_size = 64

l2_param = 1e-4

lr = 1e-4

num_step = 10000

num_class = 2

n_input = xs.shape[1]

n_hidden = [500]



tf.set_random_seed(0)

np.random.seed(0)



with tf.name_scope('input'):

    X = tf.sparse_placeholder(dtype=tf.float32)

    y_true = tf.placeholder(dtype=tf.int32)

    train_flag = tf.placeholder(dtype=tf.bool)

    y_true_one_hot = tf.one_hot(y_true, num_class)



h1 = fc_layer(X, n_input, n_hidden[0], layer_name='hidden1', input_type='sparse')



with tf.name_scope('slice_data'):

    h1_s = tf.cond(train_flag, lambda: tf.slice(h1, [0, 0], [int(batch_size / 2), -1]), lambda: h1)

    h1_t = tf.cond(train_flag, lambda: tf.slice(h1, [int(batch_size / 2), 0], [int(batch_size / 2), -1]), lambda: h1)

    ys_true = tf.cond(train_flag, lambda: tf.slice(y_true_one_hot, [0, 0], [int(batch_size / 2), -1]), lambda: y_true_one_hot)



with tf.name_scope('classifier'):

    W_clf = tf.Variable(tf.truncated_normal([n_hidden[-1], num_class], stddev=1. / tf.sqrt(n_hidden[-1] / 2.)), name='clf_weight')

    b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')

    pred_logit = tf.matmul(h1_s, W_clf) + b_clf

    pred_softmax = tf.nn.softmax(pred_logit)

    clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=ys_true))

    clf_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys_true, 1), tf.argmax(pred_softmax, 1)), tf.float32))



with tf.name_scope('dann'):

    d_label = tf.concat(values=[tf.zeros(batch_size / 2, dtype=tf.int32), tf.ones(batch_size / 2, dtype=tf.int32)], axis=0)

    d_label_one_hot = tf.one_hot(d_label, 2)

    h1_grl = flip_gradient(h1, grl_lambda)

    h_dann_1 = fc_layer(h1_grl, n_hidden[-1], 100, layer_name='dann_fc_1')

    W_domain = tf.Variable(tf.truncated_normal([100, 2], stddev=1. / tf.sqrt(100 / 2.)), name='dann_weight')

    b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')

    d_logit = tf.matmul(h_dann_1, W_domain) + b_domain

    d_softmax = tf.nn.softmax(d_logit)

    domain_loss = dann_param * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=d_label_one_hot))

    domain_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(d_label_one_hot, 1), tf.argmax(d_softmax, 1)), tf.float32))



with tf.name_scope('mmd'):

    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

    gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(h1_s, h1_t, kernel=gaussian_kernel)

    mmd_loss = mmd_param * tf.maximum(1e-4, loss_value)



all_variables = tf.trainable_variables()

l2_loss = l2_param * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])

total_loss = clf_loss + l2_loss + mmd_loss + domain_loss

train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    S_batches = batch_generator([xs, ys], int(batch_size / 2), shuffle=True)

    T_batches = batch_generator([xt, yt], int(batch_size / 2), shuffle=True)



    for i in range(num_step):

        xs_batch_csr, ys_batch = S_batches.__next__()

        xt_batch_csr, yt_batch = T_batches.__next__()

        batch_csr = vstack([xs_batch_csr, xt_batch_csr])

        xb = csr_2_sparse_tensor_tuple(batch_csr)

        yb = np.hstack([ys_batch, yt_batch])

        sess.run(train_op, feed_dict={X: xb, y_true: yb, train_flag: True})



        if i % 30 == 0:

            whole_xs_stt = csr_2_sparse_tensor_tuple(xs)

            acc_xs, c_loss_xs = sess.run([clf_acc, clf_loss], feed_dict={X: whole_xs_stt, y_true: ys, train_flag: False})

            whole_xt_stt = csr_2_sparse_tensor_tuple(xt_test)

            acc_xt, c_loss_xt = sess.run([clf_acc, clf_loss], feed_dict={X: whole_xt_stt, y_true: yt_test, train_flag: False})

            print('step: ', i)

            print('Source classifier loss: %f, Target classifier loss: %f' % (c_loss_xs, c_loss_xt))

            print('Source label accuracy: %f, Target label accuracy: %f' % (acc_xs, acc_xt))