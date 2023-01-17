import tensorflow as tf

tf.__version__
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import os

import numpy as np

from skimage import io

'''

提取mnist数据，并保存在7文件夹下

data数组为训练数据，数据shape为[-1,28,28]

data1数组为将数据合并成一个数组[28,28*batch]

'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist/", one_hot=True)





a=mnist.train.images

b=mnist.train.labels

# file=r'D:\pycharm projects\a,毕设\数据\7'

# data=[]

# count=0

# for i in range(1000):

#     if b[i][1]==1:

#         out_img=np.reshape(a[i],[28,28])

#         matplotlib.image.imsave(file+'{}.png'.format(i), out_img)

        # if i in [197,223,349,407,472,504,555,778,786,976]:

        #     data.append(out_img)

        #     if count==0:

        #         data1=out_img

        #     else:

        #         data1=np.concatenate([data1,out_img],axis=1)

        #     count=count+1



# print(np.shape(data))

# print(np.shape(data1))

# plt.imshow(data1)

# plt.show()

#



data=[]

for i in range(1000):

    if b[i][0]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [10,72,120,144,266,342,462,502,627,730]:

            data.append(out_img)

for i in range(1000):

    if b[i][1]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [4,58,123,168,234,301,429,500,610,708]:

            data.append(out_img)

for i in range(1000):

    if b[i][2]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [16,76,151,171,236,250,481,509,647,762]:

            data.append(out_img)

for i in range(1000):

    if b[i][3]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [11,67,139,207,337,434,493,531,605,748]:

            data.append(out_img)

for i in range(1000):

    if b[i][4]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [623,46,150,175,331,631,689,764,822,936]:

            data.append(out_img)

for i in range(1000):

    if b[i][6]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [619,98,466,517,576,632,671,693,846,910]:

            data.append(out_img)

for i in range(1000):

    if b[i][5]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [617,66,158,240,363,414,453,499,550,653]:

            data.append(out_img)

for i in range(1000):

    if b[i][7]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [197,223,349,407,472,504,555,778,786,976]:

            data.append(out_img)

for i in range(1000):

    if b[i][8]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [5,43,50,119,183,372,633,725,775,917]:

            data.append(out_img)

for i in range(1000):

    if b[i][9]==1:

        out_img=np.reshape(a[i],[28,28])

        if i in [17,93,125,185,209,212,391,532,592,645]:

            data.append(out_img)

data1=[]

co1=0

for i in range(1000):

    out_img=np.reshape(a[i],[28,28])

    if i in [10,4,16,11,623,617,619,197,5,17]:

        if co1==0:

            data1=out_img

        else:

            data1=np.concatenate([data1,out_img],axis=1)

        co1=co1+1

#检验数据集

# print(np.shape(data))

tf.reset_default_graph()

batch_size = 100

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')

Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')

Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])

keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')



dec_in_channels = 1

n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]

inputs_decoder = 48 * dec_in_channels/2





def lrelu(x, alpha=0.3):

    return tf.maximum(x, tf.multiply(x, alpha))





def encoder(X_in, keep_prob):

    activation = lrelu

    with tf.variable_scope("encoder", reuse=None):

        X = tf.reshape(X_in, shape=[-1, 28, 28, 1])

        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)

        x = tf.nn.dropout(x, keep_prob)

        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)

        x = tf.nn.dropout(x, keep_prob)

        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)

        x = tf.nn.dropout(x, keep_prob)

        x = tf.contrib.layers.flatten(x)

        mn = tf.layers.dense(x, units=n_latent)

        sd  = 0.5 * tf.layers.dense(x, units=n_latent)

        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))

        z  = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd





def decoder(sampled_z, keep_prob):

    with tf.variable_scope("decoder", reuse=None):

        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)

        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)

        x = tf.reshape(x, reshaped_dim)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)

        x = tf.nn.dropout(x, keep_prob)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)

        x = tf.nn.dropout(x, keep_prob)

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)



        x = tf.contrib.layers.flatten(x)

        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)

        img = tf.reshape(x, shape=[-1, 28, 28])

        return img



sampled, mn, sd = encoder(X_in, keep_prob)

dec = decoder(sampled, keep_prob)



unreshaped = tf.reshape(dec, [-1, 28*28])

img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)

latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)

loss = tf.reduce_mean(img_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)



#模型保存

saver = tf.train.Saver()

#初始化

init = tf.initialize_all_variables()

with tf.Session() as sess:

    sess.run(init)

    batch=data

    for i in range(10000):

        sess.run(optimizer, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})

        if not i % 1000:

            randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]

            imgs = sess.run(dec, feed_dict={sampled: randoms, keep_prob: 1.0})

            imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

            co=0

            for img in imgs:

                if co==0:

                    data2=img

                else:

                    data2=np.concatenate([data2,img],axis=1)

                co=co+1

            data1=np.concatenate([data1,data2],axis=0)

    save_path = saver.save(sess, 'mnist/save_net.ckpt')





plt.imshow(data1)

plt.show()
