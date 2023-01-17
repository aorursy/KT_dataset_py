import tensorflow as tf
import numpy as np
import scipy
from tensorflow.contrib.layers.python.layers import batch_norm, variance_scaling_initializer
def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size).astype(np.uint8))
def inverse_transform(image):
    return ((image + 1.)* 127.5).astype(np.uint8)
def center_crop(x, crop_h, crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    rate = np.random.uniform(0, 1, size=1)

    if rate < 0.5:
        x = np.fliplr(x)

    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

# the implements of leakyRelu

def lrelu(x, alpha=0.2, name="LeakyReLU"):
    return tf.maximum(x, alpha*x)


def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, padding='SAME',
           name="conv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=variance_scaling_initializer())
        if padding == 'Other':

            padding = 'VALID'
            input_ = tf.pad(
                input_, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")

        elif padding == 'VALID':
            padding = 'VALID'

        conv = tf.nn.conv2d(input_, w, strides=[
                            1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable(
            'biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        if with_w:
            return conv, w, biases

        else:
            return conv


def de_conv(input_, output_shape,
            k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
            name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=variance_scaling_initializer())
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        biases = tf.get_variable(
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:

            return deconv, w, biases

        else:

            return deconv


def fully_connect(input_, output_size, stddev=0.02, scope=None, with_w=False):

    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):

        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 variance_scaling_initializer())

        bias = tf.get_variable(
            "bias", [output_size], initializer=tf.constant_initializer(0.0))

        output = tf.matmul(input_, matrix) + bias

        if with_w:
            return output, with_w, bias

        else:
            return output


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])


def batch_normal(input, scope="scope", reuse=False):
    return batch_norm(input, epsilon=1e-5, decay=0.9, scale=True, scope=scope, reuse=reuse, updates_collections=None)


def resize_nearest_neighbor(x, new_size):
    x = tf.image.resize_nearest_neighbor(x, new_size)
    return x


def upscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h * scale, w * scale))


def downscale(x, scale):
    _, h, w, _ = get_conv_shape(x)
    return resize_nearest_neighbor(x, (h / scale, w / scale))


def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def avgpool2d(x, k=2):
    # avgpool wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def instance_norm(input, scope="instance_norm"):

    with tf.variable_scope(scope):

        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(
            1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable(
            "offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv

        return scale*normalized + offset


def Pixl_Norm(input, eps=1e-8):
    return input / tf.sqrt(tf.reduce_mean(input**2, axis=3, keep_dims=True) + eps)


class WScaleLayer(object):

    def __int__(self, weights, biases):

        self.scale = tf.sqrt(tf.reduce_mean(weights ** 2))
        self.bias = None
        self.we_assign = weights.assign(weights / self.scale)
        if biases is not None:
            self.bias = biases

    def getoutput_for(self, input):

        if self.bias is not None:
            input = input - self.bias

        return input * self.scale + self.bias, self.we_assign


def MinibatchstateConcat(input, averaging='all'):

    adjusted_std = lambda x, **kwargs: tf.sqrt(tf.reduce_mean(
        (x - tf.reduce_mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
    vals = adjusted_std(input, axis=0, keep_dims=True)
    if averaging == 'all':
        vals = tf.reduce_mean(vals, keep_dims=True)
    else:
        print("nothing")
    vals = tf.tile(vals, multiples=[tf.shape(input)[0], 4, 4, 1])
    return tf.concat([input, vals], axis=3)
class PGGAN(object):

    # build model
    def __init__(self, batch_size, max_iters, model_path, read_model_path, data, sample_size, sample_path, log_dir,
                 learn_rate, PG, t):

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gan_model_path = model_path
        self.read_model_path = read_model_path
        self.data_In = data
        self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.pg = PG
        self.trans = t
        self.log_vars = []
        self.channel = 3
        self.output_size = 4 * pow(2, PG - 1)
        self.images = tf.placeholder(
            tf.float32, [batch_size, self.output_size, self.output_size, self.channel])
        self.z = tf.placeholder(
            tf.float32, [self.batch_size, self.sample_size])
        self.alpha_tra = tf.Variable(
            initial_value=0.0, trainable=False, name='alpha_tra')

    def build_model_PGGan(self):

        self.fake_images = self.generate(
            self.z, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)

        _, self.D_pro_logits = self.discriminate(
            self.images, reuse=False, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.G_pro_logits = self.discriminate(
            self.fake_images, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)

        # the defination of loss for D and G
        self.D_loss = tf.reduce_mean(
            self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = -tf.reduce_mean(self.G_pro_logits)

        # gradient penalty from WGAN-GP
        self.differences = self.fake_images - self.images
        self.alpha = tf.random_uniform(
            shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.images + (self.alpha * self.differences)
        _, discri_logits = self.discriminate(
            interpolates, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]

        # 2 norm
        slopes = tf.sqrt(tf.reduce_sum(
            tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        self.D_origin_loss = self.D_loss

        self.D_loss += 10 * self.gradient_penalty
        self.D_loss += 0.001 * tf.reduce_mean(tf.square(self.D_pro_logits - 0.0))

        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            print(variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print("The total para of D", total_para)

        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            print(variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        print("The total para of G", total_para2)

        # save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(
            self.output_size) not in var.name]
        self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(
            self.output_size) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [
            var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [
            var for var in self.g_vars if 'gen_y_rgb_conv' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(
            self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(
            self.output_size) not in var.name]

        print("d_vars", len(self.d_vars))
        print("g_vars", len(self.g_vars))

        print("self.d_vars_n_read", len(self.d_vars_n_read))
        print("self.g_vars_n_read", len(self.g_vars_n_read))

        print("d_vars_n_2_rgb", len(self.d_vars_n_2_rgb))
        print("g_vars_n_2_rgb", len(self.g_vars_n_2_rgb))

        # for n in self.d_vars:
        #     print (n.name)

        self.g_d_w = [var for var in self.d_vars +
                      self.g_vars if 'bias' not in var.name]

        print("self.g_d_w", len(self.g_d_w))

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)

        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(
                self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)

        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):

        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            if self.pg != 1 and self.pg != 7:

                if self.trans:
                    self.r_saver.restore(sess, self.read_model_path)
                    self.rgb_saver.restore(sess, self.read_model_path)

                else:
                    self.saver.restore(sess, self.read_model_path)

            step = 0
            batch_num = 0
            while step <= self.max_iters:

                # optimization D
                n_critic = 1
                if self.pg == 5 and self.trans:
                    n_critic = 1

                for i in range(n_critic):

                    sample_z = np.random.normal(
                        size=[self.batch_size, self.sample_size])
                    train_list = self.data_In.getNextBatch(
                        batch_num, self.batch_size)
                    realbatch_array = self.data_In.getShapeForData(
                        train_list, resize_w=self.output_size)

                    if self.trans and self.pg != 0:

                        alpha = np.float(step) / self.max_iters

                        low_realbatch_array = scipy.ndimage.zoom(
                            realbatch_array, zoom=[1, 0.5, 0.5, 1])
                        low_realbatch_array = scipy.ndimage.zoom(
                            low_realbatch_array, zoom=[1, 2, 2, 1])
                        realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                    sess.run(opti_D, feed_dict={
                             self.images: realbatch_array, self.z: sample_z})
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.z: sample_z})

                summary_str = sess.run(summary_op, feed_dict={
                                       self.images: realbatch_array, self.z: sample_z})
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})

                if step % 400 == 0:

                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss, self.alpha_tra], feed_dict={
                                                                        self.images: realbatch_array, self.z: sample_z})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (
                        self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2],
                                '{}_{:02d}_real.png'.format(self.sample_path, step))

                    if self.trans and self.pg != 0:

                        low_realbatch_array = np.clip(
                            low_realbatch_array, -1, 1)

                        save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2],
                                    '{}_{:02d}_real_lower.png'.format(self.sample_path, step))

                    fake_image = sess.run(self.fake_images,
                                          feed_dict={self.images: realbatch_array, self.z: sample_z})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [
                                2, self.batch_size/2], '{}_{:02d}_train.png'.format(self.sample_path, step))

                if np.mod(step, 4000) == 0 and step != 0:
                    self.saver.save(sess, self.gan_model_path)
                step += 1

            save_path = self.saver.save(sess, self.gan_model_path)
            print("Model saved in file: %s" % save_path)

        tf.reset_default_graph()

    def discriminate(self, conv, reuse=False, pg=1, t=False, alpha_trans=0.01):

        #dis_as_v = []
        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = avgpool2d(conv)
                # from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim=self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1,
                                         name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(
                pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, name='dis_y_rgb_conv_{}'.format(conv.shape[1])))
            for i in range(pg - 1):

                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1,
                                    name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = avgpool2d(conv, 2)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [self.batch_size, -1])

            # for D
            output = fully_connect(conv, output_size=1, scope='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def generate(self, z_var, pg=1, t=False, alpha_trans=0.0):

        with tf.variable_scope('generator') as scope:

            de = tf.reshape(z_var, [self.batch_size, 1,
                                    1, tf.cast(self.get_nf(1), tf.int32)])
            de = conv2d(de, output_dim=self.get_nf(1), k_h=4, k_w=4,
                        d_w=1, d_h=1, padding='Other', name='gen_n_1_conv')
            de = Pixl_Norm(lrelu(de))
            de = tf.reshape(de, [self.batch_size, 4, 4,
                                 tf.cast(self.get_nf(1), tf.int32)])
            de = conv2d(de, output_dim=self.get_nf(
                1), d_w=1, d_h=1, name='gen_n_2_conv')
            de = Pixl_Norm(lrelu(de))

            for i in range(pg - 1):

                if i == pg - 2 and t:
                    # To RGB
                    de_iden = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1,
                                     name='gen_y_rgb_conv_{}'.format(de.shape[1]))
                    de_iden = upscale(de_iden, 2)

                de = upscale(de, 2)
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, name='gen_n_conv_1_{}'.format(de.shape[1]))))
                de = Pixl_Norm(lrelu(
                    conv2d(de, output_dim=self.get_nf(i + 1), d_w=1, d_h=1, name='gen_n_conv_2_{}'.format(de.shape[1]))))

            # To RGB
            de = conv2d(de, output_dim=3, k_w=1, k_h=1, d_w=1, d_h=1,
                        name='gen_y_rgb_conv_{}'.format(de.shape[1]))

            if pg == 1:
                return de

            if t:
                de = (1 - alpha_trans) * de_iden + alpha_trans*de

            else:
                de = de

            return de

    def get_nf(self, stage):
        return min(1024 / (2 ** (stage * 1)), 512)

    def get_fp(self, pg):
        return max(512 / (2 ** (pg - 1)), 16)

    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps
import os
import errno
fl = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
r_fl = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]


def read_image_list(category):

    filenames = []
    print("list file")
    list = os.listdir(category)
    list.sort()
    for file in list:
        if ('jpg' in file) or ('png' in file):
            filenames.append(category + "/" + file)
    print("list file ending!")

    length = len(filenames)
    perm = np.arange(length)
    np.random.shuffle(perm)
    filenames = np.array(filenames)
    filenames = filenames[perm]

    return filenames


class BasicDataSet(object):

    def __init__(self, image_path):

        self.dataname = "CelebA"
        self.dims = 64*64
        self.shape = [64, 64, 3]
        self.image_size = 64
        self.image_list = read_image_list(category=image_path)

    def load(self, image_path):

        # get the list of image path
        images_list = read_image_list(image_path)
        # get the data array of image

        return images_list

    @staticmethod
    def getShapeForData(filenames, resize_w=64):
        array = [get_image(batch_file, 128, is_crop=True, resize_w=resize_w,
                           is_grayscale=False) for batch_file in filenames]

        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64):

        ro_num = len(self.image_list) / batch_size - 1
        if batch_num % ro_num == 0:

            length = len(self.image_list)
            perm = np.arange(length)
            np.random.shuffle(perm)
            self.image_list = np.array(self.image_list)
            self.image_list = self.image_list[perm]
            print("images shuffle")
        t_slice = slice(int((batch_num % ro_num) * batch_size), int((batch_num % ro_num + 1) * batch_size))
        return self.image_list[t_slice]
    
def transform(image, npx= 64 , is_crop=False, resize_w= 64):

    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w = resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def get_image(image_path , image_size , is_crop=True, resize_w = 64 , is_grayscale = False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)
from skimage.util.montage import montage2d
def merge(images, size):
    if len(np.shape(images[0]))==3:
        return np.stack([montage2d(np.stack([c_img[:,:,i] for c_img in images],0)) for i in range(np.shape(images[0])[-1])],-1)
    elif len(np.shape(images[0]))==2:
        return montage2d(images)
    else:
        raise ValueError('Incompatible shape: {}'.format(np.shape(images[0])))

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


data_In = BasicDataSet('../input/train/Charlock')
root_log_dir = '.'
sample_size = 512
GAN_learn_rate = 0.0001
for i in range(11):
    t = False if (i % 2 == 0) else True
    pggan_checkpoint_dir_write = "./model_pggan/{}/".format(fl[i])
    sample_path = "./PGGanCeleba/sample_{}_{}".format(fl[i], t)
    mkdir_p(pggan_checkpoint_dir_write)
    mkdir_p(sample_path)
    pggan_checkpoint_dir_read = "./model_pggan/{}/".format(r_fl[i])
    pggan = PGGAN(batch_size=32,
                  max_iters=10,
                  model_path=pggan_checkpoint_dir_write,
                  read_model_path=pggan_checkpoint_dir_read,
                  data=data_In,
                  sample_size=sample_size,
                  sample_path=sample_path,
                  log_dir=root_log_dir,
                  learn_rate=GAN_learn_rate,
                  PG=fl[i],
                  t=t)

    pggan.build_model_PGGan()
    pggan.train()
data_In = BasicDataSet('../input/train/Charlock')
tf.reset_default_graph()
