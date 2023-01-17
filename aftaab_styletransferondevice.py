import numpy as np

import PIL.Image

import os

import scipy

import collections

import tensorflow as tf

import numpy as np

import scipy.io

import os

import shutil



def center_crop(img, new_width=None, new_height=None):

    img = np.float32(img)

    width = img.shape[1]

    height = img.shape[0]

    if new_width is None:

        new_width = min(width, height)

    if new_height is None:

        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))

    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))

    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:

        center_cropped_img = img[top:bottom, left:right]

    else:

        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img



def get_img(src, img_size=False):

    img = scipy.misc.imread(src, mode='RGB')

    if not (len(img.shape) == 3 and img.shape[2] == 3):

        img = np.dstack((img, img, img))

    if img_size != False:

        img = scipy.misc.imresize(img, img_size)

    return img





def get_files(img_dir):

    files = list_files(img_dir)

    return list(map(lambda x: os.path.join(img_dir, x), files))





def add_one_dim(image):

    shape = (1,) + image.shape

    return np.reshape(image, shape)





def list_files(in_path):

    files = []

    for (dirpath, dirnames, filenames) in os.walk(in_path):

        files.extend(filenames)

        break

    return files





def load_image(filename, shape=None, max_size=None, scale=1):

    image = PIL.Image.open(filename).convert('RGB')



    if max_size is not None:

        factor = float(max_size) / np.max(image.size)

        size = np.array(image.size) * factor



        size = size.astype(int)

        image = image.resize(size, PIL.Image.LANCZOS)  # PIL.Image.LANCZOS is one of resampling filter



    if shape is not None:

        new_shape = [scale*img_dim for img_dim in shape]

        image = image.resize(new_shape, PIL.Image.LANCZOS)  # PIL.Image.LANCZOS is one of resampling filter

        image = center_crop(image, shape[0], shape[1])

        

    return np.float32(image)





def save_image(image, filename):

    image = np.clip(image, 0.0, 255.0)



    # Convert to bytes.

    image = image.astype(np.uint8)



    # Write the image-file in jpg-format.

    with open(filename, 'wb') as file:

        PIL.Image.fromarray(image).save(file, 'jpeg')





def _conv_layer(input, weights, bias):

    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),

                        padding='SAME')

    return tf.nn.bias_add(conv, bias)





def _pool_layer(input):

    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),

                          padding='SAME')





def preprocess(image, mean_pixel):

    return image - mean_pixel





def undo_preprocess(image, mean_pixel):

    return image + mean_pixel





class VGG19:

    layers = (

        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',



        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',



        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',

        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',



        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',

        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',



        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',

        'relu5_3', 'conv5_4', 'relu5_4'

    )



    def __init__(self, data_path):

        data = scipy.io.loadmat(data_path)



        self.mean_pixel = np.array([123.68, 116.779, 103.939])



        self.weights = data['layers'][0]



    def preprocess(self, image):

        return image - self.mean_pixel



    def undo_preprocess(self, image):

        return image + self.mean_pixel



    def feed_forward(self, input_image, scope=None):

        net = {}

        current = input_image



        with tf.variable_scope(scope):

            for i, name in enumerate(self.layers):

                kind = name[:4]

                if kind == 'conv':

                    kernels = self.weights[i][0][0][2][0][0]

                    bias = self.weights[i][0][0][2][0][1]



                    # matconvnet: weights are [width, height, in_channels, out_channels]

                    # tensorflow: weights are [height, width, in_channels, out_channels]

                    kernels = np.transpose(kernels, (1, 0, 2, 3))

                    bias = bias.reshape(-1)



                    current = _conv_layer(current, kernels, bias)

                elif kind == 'relu':

                    current = tf.nn.relu(current)

                elif kind == 'pool':

                    current = _pool_layer(current)

                net[name] = current



        assert len(net) == len(self.layers)

        return net





class Transform:

    def __init__(self, n_styles, mode='train'):

        self.n_styles=n_styles

        if mode == 'train':

            self.reuse = None

        else:

            self.reuse = True



    def net(self, image, style_index):

#         image_p = self._reflection_padding(image)

        conv1 = self._conv_layer(image, 16, 9, 1, name='conv1')

        cinst1 = self._conditional_instance_norm(conv1, style_index, 'cinst1')

        

        conv2 = self._conv_layer(cinst1, 32, 3, 2, name='conv2')

        cinst2 = self._conditional_instance_norm(conv2, style_index, 'cinst2')

        

        conv3 = self._conv_layer(cinst2, 32, 3, 2, name='conv3')

        cinst3 = self._conditional_instance_norm(conv3, style_index, 'cinst3')

        

        resid1 = self._residual_block(cinst3, 3, name='resid1')

        resid2 = self._residual_block(resid1, 3, name='resid2')

        resid3 = self._residual_block(resid2, 3, name='resid3')

        resid4 = self._residual_block(resid3, 3, name='resid4')

        resid5 = self._residual_block(resid4, 3, name='resid5')

        

        conv_t1 = self._conv_tranpose_layer(resid5, 32, 3, 1, name='convt1')

        cinst4 = self._conditional_instance_norm(conv_t1, style_index, 'cinst4')

        

        conv_t2 = self._conv_tranpose_layer(cinst4, 16, 3, 1, name='convt2')

        cinst5 = self._conditional_instance_norm(conv_t2, style_index, 'cinst5')

        

        conv_t3 = self._conv_layer(conv_t2, 3, 9, 1, relu=False, name='convt3')

        return (tf.nn.tanh(conv_t3)+1)*127.5



    def _reflection_padding(self, net, padding):

        return tf.pad(net, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], "REFLECT")



    def _conv_layer(self, net, num_filters, filter_size, strides, padding='VALID', relu=True, name=None):

        weights_init = self._conv_init_vars(net, num_filters, filter_size, name=name)

        strides_shape = [1, strides, strides, 1]

        net = self._reflection_padding(net, (filter_size//2, filter_size//2))

        net = tf.nn.conv2d(net, weights_init, strides_shape, padding=padding)

        net = self._instance_norm(net, name=name)

        if relu:

            net = tf.nn.relu(net)

        return net



    def _conv_tranpose_layer(self, net, num_filters, filter_size, strides, name=None):



        batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]

        # Upsample

        net = tf.image.resize_images(net, (rows*2, cols*2), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return self._conv_layer(net, num_filters, filter_size, strides, name=name)



    def _residual_block(self, net, style_index, filter_size=3, name=None):

        batch, rows, cols, channels = [i.value for i in net.get_shape()]

        tmp = self._conv_layer(net, 32, filter_size, 1, padding='VALID', relu=True, name=name + '_1')

        return self._conv_layer(tmp, 32, filter_size, 1, padding='VALID', relu=False, name=name + '_2') + net     

    

    def _conditional_instance_norm(self, x, style_index, scope_bn):

        with tf.variable_scope(scope_bn, reuse=self.reuse):

            shift = tf.get_variable(name=scope_bn+'shift', shape=[self.n_styles, x.shape[-1]], initializer=tf.constant_initializer([0.]), trainable=True) # label_nums x C

            scale = tf.get_variable(name=scope_bn+'scale', shape=[self.n_styles, x.shape[-1]], initializer=tf.constant_initializer([1.]), trainable=True) # label_nums x C

        shift = tf.gather(shift, style_index)        

        scale = tf.gather(scale, style_index)

        x = self._instance_norm(x, name=scope_bn, shift=shift, scale=scale)

        return x

    

    def _instance_norm(self, net, name=None, shift=None, scale=None):

        batch, rows, cols, channels = [i.value for i in net.get_shape()]

        var_shape = [channels]

        mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

        if shift == None or scale == None:

            with tf.variable_scope(name, reuse=self.reuse):

                shift = tf.get_variable('shift', initializer=tf.zeros(var_shape), dtype=tf.float32)

                scale = tf.get_variable('scale', initializer=tf.ones(var_shape), dtype=tf.float32)

        epsilon = 1e-3

        normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)

        return scale * normalized + shift



    def _conv_init_vars(self, net, out_channels, filter_size, transpose=False, name=None):

        _, rows, cols, in_channels = [i.value for i in net.get_shape()]

        if not transpose:

            weights_shape = [filter_size, filter_size, in_channels, out_channels]

        else:

            weights_shape = [filter_size, filter_size, out_channels, in_channels]

        with tf.variable_scope(name, reuse=self.reuse):

            weights_init = tf.get_variable('weight', shape=weights_shape,

                                           initializer=tf.contrib.layers.variance_scaling_initializer(),

                                           dtype=tf.float32)

        return weights_init



    def _depthwise_conv_layer(self, net, num_filters, filter_size, strides, padding='SAME', relu=True, channel_mul=1,

                              name=None):

        depthwise_weights_init, pointwise_weights_init = self._depthwiseconv_init_vars(net, num_filters, channel_mul,

                                                                                       filter_size, name=name)

        strides_shape = [1, strides, strides, 1]

        net = tf.nn.separable_conv2d(net, depthwise_weights_init, pointwise_weights_init, strides_shape,

                                     padding=padding)

        net = self._conditional_instance_norm(net, name=name)

        if relu:

            net = tf.nn.relu(net)

        return net



    def _depthwiseconv_init_vars(self, net, out_channels, channel_multiplier, filter_size, name=None):

        _, rows, cols, in_channels = [i.value for i in net.get_shape()]

        depthwise_weights_shape = [filter_size, filter_size, in_channels, channel_multiplier]

        pointwise_weights_shape = [1, 1, in_channels * channel_multiplier, out_channels]



        with tf.variable_scope(name, reuse=self.reuse):

            depthwise_weights = tf.get_variable('depthwise_weight', shape=depthwise_weights_shape,

                                                initializer=tf.contrib.layers.variance_scaling_initializer(),

                                                dtype=tf.float32)

            pointwise_weights = tf.get_variable('pointwise_weight', shape=pointwise_weights_shape,

                                                initializer=tf.contrib.layers.variance_scaling_initializer(),

                                                dtype=tf.float32)

        return depthwise_weights, pointwise_weights





class StyleTransferTrainer:

    def __init__(self, content_layer_ids, style_layer_ids, content_images, style_images, session, net, num_epochs,

                 batch_size, content_weight, style_weight, tv_weight, learn_rate, save_path, check_period, test_image,

                 max_size, style_name):



        self.net = net

        self.sess = session

        self.style_name = style_name

        # sort layers info

        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))

        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))



        # input images

        self.x_list = content_images

        mod = len(content_images) % batch_size

        self.x_list = self.x_list[:-mod]

        self.y_list = style_images

        

        self.content_size = len(self.x_list)



        # parameters for optimization

        self.num_epochs = num_epochs

        self.content_weight = content_weight

        self.style_weight = style_weight

        self.tv_weight = tv_weight

        self.learn_rate = learn_rate

        self.batch_size = batch_size

        self.check_period = check_period



        # path for model to be saved

        self.save_path = save_path



        # image transform network

        self.transform = Transform(len(self.y_list))

        self.tester = Transform(len(self.y_list), 'test')



        # build graph for style transfer

        self._build_graph()



        # test during training

        if test_image is not None:

            self.TEST = True



            # load content image

            self.test_image = load_image(test_image, max_size=max_size)



            # build graph

            self.x_test = tf.placeholder(tf.float32, shape=self.test_image.shape, name='test_input')

            self.xi_test = tf.expand_dims(self.x_test, 0)  # add one dim for batch

            self.style_index_test = tf.placeholder(tf.int32, shape=(1,), name='test_style_index')

            self.style_index_test_batch = tf.expand_dims(self.style_index_test, 0)



            # result image from transform-net

            self.y_hat_test = self.tester.net(

                self.xi_test / 255.0, self.style_index_test_batch)  # please build graph for train first. tester.net reuses variables.



        else:

            self.TEST = False



    def _build_graph(self):



        """ prepare data """



        self.batch_shape = (self.batch_size, 256, 256, 3)

        print('Using shape={}'.format(self.y_list[0].shape))

        # graph input

        self.y_c = tf.placeholder(tf.float32, shape=self.batch_shape, name='content')

        self.y_s = tf.placeholder(tf.float32, shape=self.y_list[0].shape, name='style')

        self.style_index = tf.placeholder(tf.int32, shape=(1,), name='style_index')

        self.style_index_batch = tf.expand_dims(self.style_index, 0)



        # preprocess for VGG

        self.y_c_pre = self.net.preprocess(self.y_c)

        self.y_s_pre = self.net.preprocess(self.y_s)



        # get content-layer-feature for content loss

        content_layers = self.net.feed_forward(self.y_c_pre, scope='content')

        self.Ps = {}

        for id in self.CONTENT_LAYERS:

            self.Ps[id] = content_layers[id]



        # get style-layer-feature for style loss

        style_layers = self.net.feed_forward(self.y_s_pre, scope='style')

        self.As = {}

        for id in self.STYLE_LAYERS:

            self.As[id] = self._gram_matrix(style_layers[id])



        # result of image transform net

        self.x = self.y_c / 255.0

        self.y_hat = self.transform.net(self.x, self.style_index_batch)



        # get layer-values for x

        self.y_hat_pre = self.net.preprocess(self.y_hat)

        self.Fs = self.net.feed_forward(self.y_hat_pre, scope='mixed')



        """ compute loss """



        # style & content losses

        L_content = 0

        L_style = 0

        for id in self.Fs:

            if id in self.CONTENT_LAYERS:

                ## content loss ##



                F = self.Fs[id]  # content feature of x

                P = self.Ps[id]  # content feature of p



                b, h, w, d = F.get_shape()  # first return value is batch size (must be one)

                b = b.value  # batch size

                N = h.value * w.value  # product of width and height

                M = d.value  # number of filters



                w = self.CONTENT_LAYERS[id]  # weight for this layer



                L_content += w * 2 * tf.nn.l2_loss(F - P) / (b * N * M)



            elif id in self.STYLE_LAYERS:

                ## style loss ##



                F = self.Fs[id]



                b, h, w, d = F.get_shape()  # first return value is batch size (must be one)

                b = b.value  # batch size

                N = h.value * w.value  # product of width and height

                M = d.value  # number of filters



                w = self.STYLE_LAYERS[id]  # weight for this layer



                G = self._gram_matrix(F, (b, N, M))  # style feature of x

                A = self.As[id]  # style feature of a



                L_style += w * 2 * tf.nn.l2_loss(G - A) / (b * (M ** 2))



        # total variation loss

        L_tv = self._get_total_variation_loss(self.y_hat)



        """ compute total loss """



        # Loss of total variation regularization

        alpha = self.content_weight

        beta = self.style_weight

        gamma = self.tv_weight



        self.L_content = alpha * L_content

        self.L_style = beta * L_style

        self.L_tv = gamma * L_tv

        self.L_total = self.L_content + self.L_style + self.L_tv



    # borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/optimize.py

    def _get_total_variation_loss(self, img):

        b, h, w, d = img.get_shape()

        b = b.value

        h = h.value

        w = w.value

        d = d.value

        tv_y_size = (h - 1) * w * d

        tv_x_size = h * (w - 1) * d

        y_tv = tf.nn.l2_loss(img[:, 1:, :, :] - img[:, :self.batch_shape[1] - 1, :, :])

        x_tv = tf.nn.l2_loss(img[:, :, 1:, :] - img[:, :, :self.batch_shape[2] - 1, :])

        loss = 2. * (x_tv / tv_x_size + y_tv / tv_y_size) / b



        loss = tf.cast(loss, tf.float32)

        return loss



    def train(self):

        """ define optimizer Adam """

        global_step = tf.contrib.framework.get_or_create_global_step()



        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(self.L_total, trainable_variables)



        optimizer = tf.train.AdamOptimizer(self.learn_rate)

        train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=global_step,

                                             name='train_step')



        """ session run """

        self.sess.run(tf.global_variables_initializer())



        # saver to save model

        saver = tf.train.Saver()

        current_style_num = 1

        """ loop for train """

        num_examples = len(self.x_list)

        # get iteration info

        epoch = 0

        iterations = 0

        try:

            while epoch < self.num_epochs:

                while iterations * self.batch_size < num_examples:

                    if current_style_num>len(self.y_list):

                        current_style_num=1

                    curr = iterations * self.batch_size

                    step = curr + self.batch_size

                    x_batch = np.zeros(self.batch_shape, dtype=np.float32)

                    for j, img_p in enumerate(self.x_list[curr:step]):

                        x_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)



                    

                    iterations += 1



                    assert x_batch.shape[0] == self.batch_size

                   

                    _, L_total, L_content, L_style, L_tv, step = self.sess.run(

                        [train_op, self.L_total, self.L_content, self.L_style, self.L_tv, global_step],

                        feed_dict={self.y_c: x_batch, self.y_s: self.y_list[current_style_num-1], self.style_index: np.array([current_style_num-1])})



                    print('epoch : %d, iter : %4d, ' % (epoch, step),

                          'L_total : %g, L_content : %g, L_style : %g, L_tv : %f' % (L_total, L_content, L_style, L_tv))



                    if step % self.check_period == 0:

                        res = saver.save(self.sess, self.save_path + '/final.ckpt', step)



                        if self.TEST:

                            

                            output_image = self.sess.run([self.y_hat_test], feed_dict={self.x_test: self.test_image, self.style_index_test: np.array([current_style_num-1])})

                            output_image = np.squeeze(output_image[0])  # remove one dim for batch

                            output_image = np.clip(output_image, 0., 255.)



                            save_image(output_image, self.save_path + '/result_' + "%05d" % step + '.jpg')

                    current_style_num+=1

                epoch += 1

                iterations = 0

        except KeyboardInterrupt:

            pass

        finally:



            saver = tf.train.Saver()

            res = saver.save(self.sess, self.save_path + '/final.ckpt')

            self.sess.close()

            tf.reset_default_graph()



            for image_size in (384, 512):

                content_image = load_image('../input/examples/examples/content_img/content_2.png', max_size=image_size)



                # open session

                soft_config = tf.ConfigProto(allow_soft_placement=True)

                soft_config.gpu_options.allow_growth = True  # to deal with large image

                sess = tf.Session(config=soft_config)



                # build the graph

                transformer = StyleTransferTester(session=sess,

                                                  model_path=res,

                                                  content_image=content_image,

                                                  n_styles=len(self.y_list))

                transformer.save_as_tflite('{}_{}'.format(self.style_name, image_size))

                print('Saved as tflite!')

                sess.close()

                tf.reset_default_graph()



    def _gram_matrix(self, tensor, shape=None):

        if shape is not None:

            B = shape[0]  # batch size

            HW = shape[1]  # height x width

            C = shape[2]  # channels

            CHW = C * HW

        else:

            B, H, W, C = map(lambda i: i.value, tensor.get_shape())

            HW = H * W

            CHW = W * H * C



        # reshape the tensor so it is a (B, 2-dim) matrix

        # so that 'B'th gram matrix can be computed

        feats = tf.reshape(tensor, (B, HW, C))



        # leave dimension of batch as it is

        feats_T = tf.transpose(feats, perm=[0, 2, 1])



        # paper suggests to normalize gram matrix by its number of elements

        gram = tf.matmul(feats_T, feats) / CHW



        return gram





class StyleTransferTester:



    def __init__(self, session, content_image, model_path, n_styles):

        # session

        self.sess = session

        self.n_styles = n_styles

        # input images

        self.x0 = content_image

        self.style_index0 = np.array([5], dtype=np.int32)

        

        # input model

        self.model_path = model_path



        # image transform network

        self.transform = Transform(n_styles)



        # build graph for style transfer

        self._build_graph()



    def _build_graph(self):

        # graph input

        self.x = tf.placeholder(tf.float32, shape=self.x0.shape, name='input')

        self.style_index = tf.placeholder(tf.int32, shape=(1,), name='style_index')

        self.style_index_batch = tf.expand_dims(self.style_index, 0)



        self.xi = tf.expand_dims(self.x, 0)  # add one dim for batch

        # result image from transform-net

        self.y_hat = self.transform.net(self.xi / 255.0, self.style_index_batch)

        self.y_hat = tf.squeeze(self.y_hat)  # remove one dim for batch

        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)

        self.y_hat = tf.reshape(self.y_hat, np.array((-1,) + self.x0.shape, dtype=np.int32))



    def test(self):

        # initialize parameters

        self.sess.run(tf.global_variables_initializer())



        # load pre-trained model

        saver = tf.train.Saver()

        saver.restore(self.sess, self.model_path)

        

        # get transformed image

        output = self.sess.run(self.y_hat, feed_dict={self.x: self.x0, self.style_index: self.style_index0})



        return output



    def save_as_tflite(self, model_name):

        self.sess.run(tf.global_variables_initializer())



        # load pre-trained model

        saver = tf.train.Saver()

        saver.restore(self.sess, self.model_path)



        converter = tf.lite.TFLiteConverter.from_session(self.sess, [self.x, self.style_index], [self.y_hat])

        

        tflite_model = converter.convert()



        if not os.path.exists('tflite_models_final/'):

            os.mkdir('tflite_models_final')



        with open('tflite_models_final/{}.tflite'.format(model_name), 'wb') as f:

            f.write(tflite_model)



    def save_as_saved_model(self, model_name, max_size):

        self.sess.run(tf.global_variables_initializer())

        tf.saved_model.simple_save(self.sess, '{}/{}/'.format(max_size, model_name), {'input': self.xi},

                                   {'output': self.y_hat})





def main():

    # initiate VGG19 model

    if os.path.exists('models'):

        shutil.rmtree('models')

    os.mkdir('models')

    model_file_path = '../input/imagenetvggverydeep19mat/imagenet-vgg-verydeep-19.mat'

    vgg_net = VGG19(model_file_path)



    # get file list for training

    content_images = get_files('../input/mscoco/mscoco/mscoco_resized/train2014/')



    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

    style_layer_weights = [0.2, 0.2, 0.2, 0.2, 0.2]



    content_layers = ['relu4_2']

    content_layer_weights = [1.0]



    # create a map for content layers info

    CONTENT_LAYERS = {}

    for layer, weight in zip(content_layers, content_layer_weights):

        CONTENT_LAYERS[layer] = weight



    # create a map for style layers info

    STYLE_LAYERS = {}

    for layer, weight in zip(style_layers, style_layer_weights):

        STYLE_LAYERS[layer] = weight



    # open session

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    

    style_dir_name = 'texture'

    style_list = os.listdir('../input/examples/examples/style_img/{}'.format(style_dir_name))

    style_list.sort()

    style_list = ['../input/examples/examples/style_img/{}/'.format(style_dir_name)+ip for ip in style_list]

    

    style_list = [load_image(i, (256,256), scale=2) for i in style_list]

    # build the graph for train

    trainer = StyleTransferTrainer(session=sess,

                                   content_layer_ids=CONTENT_LAYERS,

                                   style_layer_ids=STYLE_LAYERS,

                                   content_images=content_images,

                                   style_images=[add_one_dim(style_image) for style_image in style_list],

                                   net=vgg_net,

                                   num_epochs=3,

                                   batch_size=4,

                                   content_weight=7.5e0,

                                   style_weight=5e1,

                                   tv_weight=2e2,

                                   learn_rate=10e-3,

                                   save_path='models',

                                   check_period=1000,

                                   test_image='../input/examples/examples/content_img/content_2.png',

                                   max_size=256,

                                   style_name=style_dir_name)

    # launch the graph in a session



    trainer.train()

    sess.close()





tf.reset_default_graph()



main()
