# from collections import namedtuple
# from easydict import EasyDict as edict

# SIZE = namedtuple('SIZE', ['h', 'w'])
# INFO = namedtuple('INFO', ['size', 'scale', 'aspect_ratios'])

# size_a = SIZE(16, 19)
# info_a = INFO(size_a, 2.5, '16:9')
# print(size_a.h, size_a.w)
# print(info_a.size.h)

# __C = edict()
# cfg = __C

# __C.BATCH = 32

# __C.TRAIN = edict()
# __C.TRAIN.BATCH = 16

# import play_ground.cfg as cfg
# cfg.TRAIN.BATCH
import re
import os
import time
import numpy as np
import tensorflow as tf
def slice_array(array, batch_size):
    return np.array([array[i:i + batch_size]
                     for i in range(0, array.shape[0], batch_size)])


def get_file_path(train_ratio, data_dir, extension='.JPG|.png|.jpeg'):
    target = re.compile('^[^\.].*(' + extension + ')$')
    path_list = []
    for path, folders, files in os.walk(data_dir):
        for file in files:
            try:
                full_name = target.match(file).group()
                full_path = os.path.join(path, full_name)
                path_list.append(full_path)
            except:
                pass

    # To convert the python list into numpy array.
    path_list = np.array(path_list)
    if train_ratio < 1:
        train_list, test_list = slice_array(
            path_list, int(path_list.shape[0] * train_ratio))
    else:
        train_ratio = 1
        train_list = slice_array(
            path_list, int(path_list.shape[0] * train_ratio))[0]
        test_list = np.array([])
    return train_list, test_list


def one_hot(labels, num_classes=10):
    return np.eye(num_classes, dtype=float)[labels]


def time_counter(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        print('Took {0:.4} sec on "{1}" func'.format(t2, func.__name__))
        return result
    return wrapper


def BN(inputs, epsilon=0.0001):
    size = inputs.shape[-1]
    beta = tf.Variable(tf.constant(0.01, shape=[size], dtype=tf.float32))
    gama = tf.Variable(tf.constant(0.997, shape=[size], dtype=tf.float32))
    mean, var = tf.nn.moment(inputs, axis=[0], keep_dims=False)
    inputs = tf.nn.batch_normalization(inputs, mean, var, beta, gama, epsilon)
    return inputs


def uninitialized_variables_initializer(sess):
    uninit_vars = []
    uninit_tensors = []

    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))

    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))


@time_counter
def bbox_loop(w, h):
    coords = np.zeros((w * h, 2))

    n = 0
    for i in np.arange(w):
        coord = [i]

        for j in np.arange(h):
            coord.append(j)
            coords[n, :] = coord
            coord.remove(j)
            n += 1

    return coords


@time_counter
def bbox(w, h):
    w, h = np.meshgrid(np.arange(w), np.arange(h))
    return np.vstack((w.ravel(), h.ravel())).transpose()
# a = np.arange(48).reshape(6, 8)
# print(a)
# print(slice_array(a, 4))

# path = '/Users/kcl/Desktop'
# b = get_file_path(0.7, path, extension='.pdf')
# print(b)

# c = np.random.randint(0, 10, 8)
# print(c)
# print(one_hot(c, num_classes=10))

N = 10000

bbox(N, N)
bbox_loop(N, N)

