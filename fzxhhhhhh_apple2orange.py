import warnings

warnings.filterwarnings("ignore")
import cv2

import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from glob import glob

from tensorflow.python.framework import graph_util
img_h = 256

img_w = 256

img_shape = (img_h, img_w, 3)

dirs = "../input/images"

dataset_name = "apple2orange"

batch_size = 1
class Data_Generator():

    """数据生成器"""

    def __init__(self, dirs, name, img_shape, batch_size, is_testing=False):

        """初始化数据集参数"""

        self.dirs = dirs

        self.name = name

        self.img_shape = img_shape

        self.batch_size = batch_size

        self.is_testing = is_testing

        self.h, self.w = self.img_shape[0], self.img_shape[1]

    

    def img_read(self, path):

        """读取一张图片"""

        img = cv2.imread(path)

        img = img[...,::-1]

        img = cv2.resize(img, (self.h, self.w), interpolation=cv2.INTER_LINEAR)

        return img

    

    def get_img_paths(self, is_testing=False):

        """取得所有图片的路径"""

        data_type = "train" if not is_testing else "test"

        imgs_A_log = "%s/%s/%sA/*" % (self.dirs, self.name, data_type)

        imgs_B_log = "%s/%s/%sB/*" % (self.dirs, self.name, data_type)

        imgs_A_path = glob(imgs_A_log)

        imgs_B_path = glob(imgs_B_log)

        return imgs_A_path, imgs_B_path

    

    def load_batch_imgs(self):

        """加载一个batch的数据，作为生成器"""

        A_paths, B_paths = self.get_img_paths(self.is_testing)

        max_batch = min(len(A_paths), len(B_paths)) // self.batch_size

        total = max_batch * self.batch_size

        A_paths = np.random.choice(A_paths, total, replace=False)

        B_paths = np.random.choice(B_paths, total, replace=False)

        A_paths = np.reshape(A_paths, (-1, self.batch_size))

        B_paths = np.reshape(B_paths, (-1, self.batch_size))

        for batch_imgs_A, batch_imgs_B in zip(A_paths, B_paths):

            imgs_A, imgs_B = [], []

            for img_A, img_B in zip(batch_imgs_A, batch_imgs_B):

                imgs_A.append(self.img_read(img_A))

                imgs_B.append(self.img_read(img_B))

            imgs_A = np.array(imgs_A, dtype=np.float32) / 127.5 -1.

            imgs_B = np.array(imgs_B, dtype=np.float32) / 127.5 -1.

            yield imgs_A, imgs_B

            

    def load_data(self):

        A_paths, B_paths = self.get_img_paths(self.is_testing)

        A_path = np.random.choice(A_paths, 1, replace=False)

        B_path = np.random.choice(B_paths, 1, replace=False)

        img_A = self.img_read(A_path[0]).astype(np.float32) / 127.5 -1

        img_B = self.img_read(B_path[0]).astype(np.float32) / 127.5 -1

        return img_A, img_B
def get_generator(inputs, scope, reuse=False):

    """创建A到B的生成网络"""

    def residual_block(inputs, filters, kernel_size):

        """残差块"""

        pad = kernel_size // 2

        x = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

        x = tf.layers.conv2d(x, filters, kernel_size, 1)

        x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.relu(x)



        x = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

        x = tf.layers.conv2d(x, filters, kernel_size, 1)

        x = tf.contrib.layers.instance_norm(x)

        x = tf.add(x, inputs)

        x = tf.nn.relu(x)

        return x

    

    with tf.variable_scope(scope, reuse=reuse):

        x = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]])

        x = tf.layers.conv2d(x, 64, 7, 1)

        x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.relu(x)

        

        # 下采样

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

        x = tf.layers.conv2d(x, 128, 3, 2)

        x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.relu(x)

        

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])

        x = tf.layers.conv2d(x, 256, 3, 2)

        x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.relu(x)

        

        # 残差部分

        for i in range(9):

            x = residual_block(x, 256, 3)

            

        # 上采样

        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=3, strides=2, padding="same")

        x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.relu(x)

        

        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=3, strides=2, padding="same")

        x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.relu(x)

        

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]])

        x = tf.layers.conv2d(x, 3, 7, 1)

        x = tf.nn.tanh(x, name="generator")

        return x
def get_discriminator(inputs, scope, reuse):

    """创建判别模型"""

    def conv2d(inputs, filters, kernel_size, strides, normal=True):

        x = tf.layers.conv2d(inputs, filters, kernel_size, strides, padding="same")

        if normal:

            x = tf.contrib.layers.instance_norm(x)

        x = tf.nn.leaky_relu(x)

        return x

    with tf.variable_scope(scope, reuse=reuse):

        x = conv2d(inputs, 64, 4, 2, False)

        x = conv2d(x, 128, 4, 2)

        x = conv2d(x, 256, 4, 2)

        x = conv2d(x, 512, 4, 2)

        x = tf.layers.conv2d(x, 1, 3, 1, padding="same")

        return x
X_A = tf.placeholder(dtype=tf.float32, shape=(None,)+img_shape, name="X_A")  # A域图片占位符

X_B = tf.placeholder(dtype=tf.float32, shape=(None,)+img_shape, name="X_B")  # B域图片占位符

fake_B = get_generator(X_A, "gen_A2B", False)  # 得到A2B的数据流图

fake_A = get_generator(X_B, "gen_B2A", False)  # 得到B2A的数据流图

re_A = get_generator(fake_B, "gen_B2A", True)  # 从fakeB重构为A

re_B = get_generator(fake_A, "gen_A2B", True)  # 从fakeA重构为B

id_A = get_generator(X_A, "gen_B2A", True)  # 要求A域不会被B2A改变

id_B = get_generator(X_B, "gen_A2B", True)  # 要求B域不会被A2B改变

d_fake_A = get_discriminator(fake_A, "d_A", False)  # 辨别fakeA

d_fake_B = get_discriminator(fake_B, "d_B", False)  # 辨别fakeB

d_real_A = get_discriminator(X_A, "d_A", True)  # 辨别真实A图片

d_real_B = get_discriminator(X_B, "d_B", True)  # 辨别真实B图片
lambda_cycle = 5

lambda_id = 2.5

lambda_d = 0.5
def genA2B_loss(re_B, id_B, X_B, d_fake_B):

    """genA2B损失函数"""

    valid_B = np.ones(shape=(batch_size,)+(16, 16, 1))

    re_loss = tf.reduce_mean(tf.abs(re_B-X_B))

    id_loss = tf.reduce_mean(tf.abs(id_B-X_B))

    fake_loss = tf.reduce_mean(tf.square(d_fake_B-valid_B))

    return lambda_cycle * re_loss + lambda_id * id_loss + lambda_d * fake_loss
def genB2A_loss(re_A, id_A, X_A, d_fake_A):

    """genB2A损失函数"""

    valid_A = np.ones(shape=(batch_size,)+(16, 16, 1))

    re_loss = tf.reduce_mean(tf.abs(re_A-X_A))

    id_loss = tf.reduce_mean(tf.abs(id_A-X_A))

    fake_loss = tf.reduce_mean(tf.square(d_fake_A-valid_A))

    return lambda_cycle * re_loss + lambda_id * id_loss + lambda_d * fake_loss
def discriminator_A_loss(d_fake_A, d_real_A):

    """d_A损失函数"""

    valid = np.ones(shape=(batch_size,)+(16, 16, 1))

    fake = np.zeros(shape=(batch_size,)+(16, 16, 1))

    fake_loss = tf.reduce_mean(tf.square(d_fake_A-fake))

    real_loss = tf.reduce_mean(tf.square(d_real_A-valid))

    return fake_loss + real_loss
def discriminator_B_loss(d_fake_B, d_real_B):

    """d_A损失函数"""

    valid = np.ones(shape=(batch_size,)+(16, 16, 1))

    fake = np.zeros(shape=(batch_size,)+(16, 16, 1))

    fake_loss = tf.reduce_mean(tf.square(d_fake_B-fake))

    real_loss = tf.reduce_mean(tf.square(d_real_B-valid))

    return fake_loss + real_loss
A2B_loss = genA2B_loss(re_B, id_B, X_B, d_fake_B)

B2A_loss = genB2A_loss(re_A, id_A, X_A, d_fake_A)

D_A_loss = discriminator_A_loss(d_fake_A, d_real_A)

D_B_loss = discriminator_B_loss(d_fake_B, d_real_B)
optmizier = tf.train.AdamOptimizer(0.0002, 0.5)
D_A_train_op = optmizier.minimize(D_A_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith("d_A")])

D_B_train_op = optmizier.minimize(D_B_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith("d_B")])

A2B_train_op = optmizier.minimize(A2B_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith("gen_A2B")])

B2A_train_op = optmizier.minimize(B2A_loss, var_list=[var for var in tf.trainable_variables() if var.name.startswith("gen_B2A")])
def sample_images(epoch, img_A, img_B, gen_A, gen_B):

        os.makedirs('trans_images/%s' % dataset_name, exist_ok=True)

        r, c = 2, 2

        

        img_A = np.expand_dims(img_A, axis=0)

        img_B = np.expand_dims(img_B, axis=0)

        gen_imgs = np.concatenate([img_A, gen_B, img_B, gen_A])

        gen_imgs = 0.5 * gen_imgs + 0.5



        titles = ['Original', 'Translated', 'Reconstructed']

        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt])

                axs[i, j].set_title(titles[j])

                axs[i,j].axis('off')

                cnt += 1

        fig.savefig("trans_images/%s/%d.png" % (dataset_name, epoch))

        plt.close()
config = tf.ConfigProto()

config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    epochs = 31

    data_loader = Data_Generator(dirs, dataset_name, img_shape, batch_size, False)

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):

        dA_losses, dB_losses, genA2B_losses, genB2A_losses = [], [], [], []

        for i, (img_A, img_B) in enumerate(data_loader.load_batch_imgs()):

            feed_dict = {X_A: img_A, X_B: img_B}

            _, genA2B_loss = sess.run([A2B_train_op, A2B_loss], feed_dict=feed_dict)

            _, genB2A_loss = sess.run([B2A_train_op, B2A_loss], feed_dict=feed_dict)

            _, dA_loss = sess.run([D_A_train_op, D_A_loss], feed_dict=feed_dict)

            _, dB_loss = sess.run([D_B_train_op, D_B_loss], feed_dict=feed_dict)

            dA_losses.append(dA_loss)

            dB_losses.append(dB_loss)

            genA2B_losses.append(genA2B_loss)

            genB2A_losses.append(genB2A_loss)

            if i % 50 == 0:

                str_log = "Epoch: %d \t Batch: %d \t d_loss: %f \t g_loss: %f"

                print(str_log % (epoch, i, np.mean(dA_losses+dB_losses)/2, np.mean(genA2B_losses+genB2A_losses)/2))

        img_A, img_B = Data_Generator(dirs, dataset_name, img_shape, 1, True).load_data()

        gen_B = sess.run(fake_B, feed_dict={X_A: np.expand_dims(img_A, axis=0)})

        gen_A = sess.run(fake_A, feed_dict={X_B: np.expand_dims(img_B, axis=0)})

        sample_images(epoch, img_A, img_B, gen_A, gen_B)

        

        if epoch % 10 ==0 and epoch !=0:

            pb_file_path = "train_models/epoch_%d" % epoch

            os.makedirs(pb_file_path, exist_ok=True)

            with tf.gfile.FastGFile(pb_file_path+'/apple2orange.pb', mode='wb') as f:

                A2B_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['gen_A2B/generator'])

                f.write(A2B_graph.SerializeToString())

            with tf.gfile.FastGFile(pb_file_path+'/orange2apple.pb', mode='wb') as f:

                B2A_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['gen_B2A/generator'])

                f.write(B2A_graph.SerializeToString())