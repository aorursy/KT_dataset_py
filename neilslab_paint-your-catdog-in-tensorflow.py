TRAIN_DIR = '../input/train/'

import os

import numpy as np

import matplotlib.pyplot as plt

from skimage.transform import resize

from skimage import data

import tensorflow as tf

import warnings

warnings.filterwarnings("ignore", category=UserWarning) 
files = [os.path.join(TRAIN_DIR, fname)

        for fname in os.listdir(TRAIN_DIR)]



cats = files[:3]

dogs = files[-4:-1]



cat_imgs = [plt.imread(fname)[..., :3] for fname in cats]

cat_imgs = [resize(img_i, (100, 100)) for img_i in cat_imgs]

cat_imgs = np.array(cat_imgs).astype(np.float32)



dog_imgs = [plt.imread(fname)[..., :3] for fname in dogs]

dog_imgs = [resize(img_i, (100, 100)) for img_i in dog_imgs]

dog_imgs = np.array(dog_imgs).astype(np.float32)
plt.imshow(cat_imgs[1])
plt.imshow(dog_imgs[1])
imgs = np.concatenate((cat_imgs[1], dog_imgs[1]), axis=1)

plt.imshow(imgs)
def split_image(img):

    # positions, ie row/column tuple

    xs = []



    # 3 rgb colors

    ys = []



    for row_i in range(img.shape[0]):

        for col_i in range(img.shape[1]):

            xs.append([row_i, col_i])

            ys.append(img[row_i, col_i])

            

    xs = np.array(xs)

    ys = np.array(ys)

    return xs, ys
xs, ys = split_image(imgs)



xs.shape, ys.shape
xs = ((xs - np.mean(xs)) / np.std(xs))
print(np.min(ys), np.max(ys))
X = tf.placeholder(name='X', shape=(None, 2), dtype=tf.float32)
def linear(x, n_output, name=None, activation=None, reuse=None):



    n_input = x.get_shape().as_list()[1]



    with tf.variable_scope(name or "fully_connected", reuse=reuse):

        W = tf.get_variable(

            name='W',

            shape=[n_input, n_output],

            dtype=tf.float32,

            initializer=tf.contrib.layers.xavier_initializer())



        b = tf.get_variable(

            name='b',

            shape=[n_output],

            dtype=tf.float32,

            initializer=tf.constant_initializer(0.0))



        h = tf.nn.bias_add(

            name='h',

            value=tf.matmul(x, W),

            bias=b)



        if activation:

            h = activation(h)



        return h, W
h, W = linear(x=X, n_output=20, name='linear', activation=tf.nn.relu)
tf.reset_default_graph()

X = tf.placeholder(name='X', shape=(None, 2), dtype=tf.float32)

Y = tf.placeholder(name='Y', shape=(None, 3), dtype=tf.float32)
n_neurons = 100

h1, W1 = linear(X, n_neurons, name='layer1', activation=tf.nn.relu)

h2, W2 = linear(h1, n_neurons, name='layer2', activation=tf.nn.relu)

h3, W3 = linear(h2, n_neurons, name='layer3', activation=tf.nn.relu)

h4, W4 = linear(h3, n_neurons, name='layer4', activation=tf.nn.relu)

h5, W5 = linear(h4, n_neurons, name='layer5', activation=tf.nn.relu)



Y_pred, W6 = linear(h5, 3, activation=None, name='pred')
error = tf.squared_difference(Y, Y_pred)

sum_error = tf.reduce_sum(input_tensor=error, axis=1)

cost = tf.reduce_mean(input_tensor=sum_error)
optimizer = tf.train.AdamOptimizer(learning_rate=0.003).minimize(cost)

n_iterations = 300

batch_size = 50

sess = tf.Session()
print(imgs.shape)

img_shape = imgs.shape
sess.run(tf.global_variables_initializer())



imgs = []

display_step = n_iterations // 10



for it_i in range(n_iterations):

    

    idxs = np.random.permutation(range(len(xs)))

    

    n_batches = len(idxs) // batch_size

    

    for batch_i in range(n_batches):

        

        idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

        

        training_cost = sess.run([cost, optimizer],

                                feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})[0]

    

    if (it_i + 1) % display_step == 0:

        

        ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)

        img = np.clip(ys_pred.reshape(img_shape), 0, 1)

        imgs.append(img)

        

        ax = plt.imshow(img)

        plt.title('Iteration {}'.format(it_i))

        plt.show()
plt.figure(figsize=(8, 8))

plt.imshow(imgs[-1])
plt.imsave(fname='my_catdog.png', arr=imgs[-1])