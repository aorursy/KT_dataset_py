import os

import csv



import numpy as np 

import pandas as pd 



import PIL.Image

import tensorflow as tf

from tensorflow.python.training import moving_averages

from tensorflow.python.ops import control_flow_ops



%load_ext autoreload

%autoreload 2



TRAIN_DIR = "/kaggle/input/AIAxTainanDL/flower/flower_classification/train/"

TEST_DIR = "/kaggle/input/AIAxTainanDL/flower/flower_classification/test/"

CLASSES = "/kaggle/input/AIAxTainanDL/flower/flower_classification/mapping.csv"

SUBMISSION = "./submission.csv"



INPUT_SIZE = 64
classes = pd.read_csv(CLASSES)

classes.head()
imgs_path = []

labels = []

for i, row in classes.iterrows():

    data_dir = os.path.join(TRAIN_DIR, row["dirs"])

    img_path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    label = [row["class"] for l in range(len(img_path))]

    

    imgs_path.extend(img_path)

    labels.extend(label)
train_X = []



for img_path in imgs_path:

    image = PIL.Image.open(img_path)

        

    image = image.convert('RGB')

    image = image.resize((INPUT_SIZE, INPUT_SIZE), PIL.Image.ANTIALIAS)

            

    train_X.append(np.array(image))
train_X = np.asarray(train_X, dtype=np.float32)

train_Y = np.asarray(labels, dtype=np.int32)



idx = np.arange(train_X.shape[0])

np.random.shuffle(idx)



train_X = train_X[idx]

train_Y = train_Y[idx]



print(train_X.shape)

print(train_Y.shape)
test_imgs_path = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if os.path.isfile(os.path.join(TEST_DIR, f))]

test_X = []

for img_path in test_imgs_path:

    image = PIL.Image.open(img_path)

        

    image = image.convert('RGB')

    image = image.resize((INPUT_SIZE, INPUT_SIZE), PIL.Image.ANTIALIAS)

    

    test_X.append(np.asarray(image))

test_X = np.asarray(test_X)
class Dataset(object):

    def __init__(self, X, y, batch_size):

        self.X, self.y = X, y

        self.batch_size = batch_size

        

    def __iter__(self):

        N, B = self.X.shape[0], self.batch_size

        

        idxs = np.arange(N)

        np.random.shuffle = B

        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

    

train_dset = Dataset(train_X[:-200], train_Y[:-200], 64)

val_dset = Dataset(train_X[-200:], train_Y[-200:], 64)
def flatten(x):

    N = tf.shape(x)[0]

    return tf.reshape(x, (N, -1))
def batchnorm(layer, moving_mean, moving_var, offset, is_conv=True, is_training=True):

    axis = list(range(len(layer.shape) - 1))

    mean, var = tf.nn.moments(layer, axis)

    

    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9)

    update_moving_var = moving_averages.assign_moving_average(moving_var, var, 0.9)

    

    mean, var = control_flow_ops.cond(is_training, lambda: (mean, var), lambda: (moving_mean, moving_var))

    return tf.nn.batch_normalization(layer, mean, var, offset, None, 1e-5)
def convnet_init(conv_layers, fc_layers):

    params = None

    input_dim = 3

    

    conv_w = []

    conv_b = []

    conv_mean = []

    conv_var = []

    

    fc_w = []

    fc_b = []

    fc_mean = []

    fc_var = []

    

    feature_size = INPUT_SIZE

    

    for index, layer in enumerate(conv_layers):

        layer = tf.constant(layer)

        conv_w.append(tf.Variable(tf.random_normal((3, 3, input_dim, layer), stddev=1e-2)))

        conv_b.append(tf.Variable(tf.zeros((layer,))))

        

        conv_mean.append(tf.Variable(tf.zeros((layer,))))

        conv_var.append(tf.Variable(tf.zeros((layer,))))

        

        feature_size -= 2

        input_dim = layer

        

    input_dim = feature_size*feature_size*input_dim

    for index, layer in enumerate(fc_layers):

        fc_w.append(tf.Variable(tf.random_normal((input_dim, layer), stddev=1e-2)))

        fc_b.append(tf.Variable(tf.zeros((layer, ))))

        

        fc_mean.append(tf.Variable(tf.zeros((layer,))))

        fc_var.append(tf.Variable(tf.zeros((layer,))))

        

        input_dim = layer

    

    params = [conv_w, conv_b, fc_w, fc_b]

    mean_var = [conv_mean, conv_var, fc_mean, fc_var]

    return params, mean_var
def convnet(x, params, mean_var, is_training):

    conv_w, conv_b, fc_w, fc_b = params

    conv_mean, conv_var, fc_mean, fc_var = mean_var

    scores = None

    

    layers_n = len(conv_w)

    

    last_conv = x

    for index in range(layers_n):

        next_conv = tf.nn.relu(tf.nn.conv2d(input=last_conv, filter=conv_w[index], padding="VALID"))

        next_conv = batchnorm(next_conv, conv_mean[index], conv_var[index], conv_b[index], tf.constant(True, dtype=tf.bool), is_training)

        last_conv = next_conv

        

    last_fc = flatten(last_conv)

    

    fc_layers_n = len(fc_w)

    for index in range(fc_layers_n-1):

        next_fc = tf.nn.relu(tf.matmul(last_fc, fc_w[index]))

        next_fc = batchnorm(next_fc, fc_mean[index], fc_var[index], fc_b[index], tf.constant(False, dtype=tf.bool), is_training)

        last_fc = next_fc

        

    scores = tf.matmul(last_fc, fc_w[fc_layers_n-1]) + fc_b[fc_layers_n-1]

        

    return scores 
def training_step(scores, y, params, lr):

    param_list = []

    for l_t in params:

        for l in l_t:

            param_list.append(l)

    

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)

    loss = tf.reduce_mean(losses)

    

    grad_params = tf.gradients(loss, param_list)

    

    new_weights = []

    for w, grad_w in zip(param_list, grad_params):

        new_w = tf.assign_sub(w, lr*grad_w)

        new_weights.append(new_w)

    

    with tf.control_dependencies(new_weights):

        return tf.identity(loss)
def check_accuracy(sess, dset, x, scores, is_training=None):

    num_correct, num_samples = 0, 0

    for x_batch, y_batch in dset:

        feed_dict = {x: x_batch, is_training: 1}

        scores_np = sess.run(scores, feed_dict=feed_dict)

        y_pred = scores_np.argmax(axis=1)

        num_samples += x_batch.shape[0]

        num_correct += (y_pred == y_batch).sum()

    acc = float(num_correct) / num_samples

    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
def test_step(sess, x, scores, is_training=None):

    pred_label = []

    for test_x in test_X:

        W, H, C = test_x.shape

        test_x = np.reshape(test_x, (1, W, H, C))

        feed_dict = {x: test_x, is_training: 0}

        scores_np = sess.run(scores, feed_dict=feed_dict)

        y_pred = scores_np.argmax(axis=1)

        pred_label.append(y_pred)



    with open(SUBMISSION, 'w', newline='') as csvFile:

        writer = csv.writer(csvFile)

    

        writer.writerow(['id', 'class'])

        for index in range(len(pred_label)):

            file_name = test_imgs_path[index][61:-4]

            writer.writerow([file_name, pred_label[index][0]])
CONV_LAYER = np.array([32, 64, 256, 256, 512], dtype=np.int32)

FC_LAYER = np.array([200, 5], dtype=np.int32)

learning_rate = 5e-4
def train(model_fn, model_init, lr, epochs):

    tf.reset_default_graph()

    

    with tf.device('/device:GPU:0'):

        x = tf.placeholder(tf.float32, [None, INPUT_SIZE, INPUT_SIZE, 3])

        y = tf.placeholder(tf.int32, [None])

        is_training = tf.placeholder(tf.bool, name='is_training')

        params, mean_var = model_init(CONV_LAYER, FC_LAYER)

        scores = model_fn(x, params, mean_var, is_training)

        loss = training_step(scores, y, params, lr)

        

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):

            for t, (x_np, y_np) in enumerate(train_dset):

                feed_dict = {x:x_np, y:y_np, is_training:1}

                loss_np = sess.run(loss, feed_dict=feed_dict)

            if e % 5 == 0:

                print('Epoch %d, loss = %.4f' % (e, loss_np))

                check_accuracy(sess, val_dset, x, scores, is_training)

        test_step(sess, x, scores, is_training)
params = train(convnet, convnet_init, learning_rate, 15)