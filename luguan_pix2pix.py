#导入必要的库

import keras.backend as K



from keras.models import Model



# 导入Keras不同的层

from keras.layers import Conv2D, BatchNormalization, Input, Dropout, Add

from keras.layers import Conv2DTranspose, Reshape, Activation, Dense

from keras.layers import Concatenate, UpSampling2D, Flatten, GlobalAveragePooling2D



# 导入Adam优化器

from keras.optimizers import Adam



# 导入要用到的激活函数

from keras.layers.advanced_activations import LeakyReLU

from keras.activations import relu,tanh



# 导入图像处理库

from keras.preprocessing.image import load_img, img_to_array, array_to_img

import numpy as np



# glob用于处理文件

import glob



# 随机

import random



from keras.initializers import RandomNormal

conv_init = RandomNormal(0, 0.02)
# 处理数据

def load_image(fn, image_size):

    """

    加载一张图片

    fn:图像文件路径

    image_size:图像大小

    """

    

    # 和Image有关的文档

    # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html

    # 打开图片并转化成RGB格式

    im = load_img(fn)

    img_a = im.crop((0, 0, im.size[0] / 2, im.size[1]))

    img_b = im.crop((im.size[0] / 2, 0, im.size[0], im.size[1]))

    def clip(im,image_size):

        # 切割图像(截取图像中间的最大正方形)

        # crop 切割图像，接受一个四元组，分别表示左上右下

        if (im.size[0] >= im.size[1]):

            im = im.crop(((im.size[0] - im.size[1])//2, 0, (im.size[0] + im.size[1])//2, im.size[1]))

        else:

            im = im.crop((0, (im.size[1] - im.size[0])//2, im.size[0], (im.size[0] + im.size[1])//2))

        im = im.resize((image_size, image_size))

        return im

    img_a = clip(img_a, image_size)

    img_b = clip(img_b, image_size)

    #将0-255的RGB值转换到[-1,1]上的值

    arr_a = img_to_array(img_a) / 255 * 2 - 1

    arr_b = img_to_array(img_b) / 255 * 2 - 1

    return arr_a, arr_b



class DataSet(object):

    """

    用于管理数据的类

    """

    def __init__(self, data_path, image_size = 256):

        """

        构造函数

        """

        

        # 数据集路径

        self.data_path = data_path

        # 轮数

        self.epoch = 0

        # 初始化数据列表（调用自身方法）

        self.__init_list()

        # 图片尺寸

        self.image_size = image_size

        

    def __init_list(self):

        # glob.glob 输入pathname, 返回符合pathname的文件名的列表

        # 可以使用通配符

        # https://docs.python.org/3/library/glob.html

        self.data_list = glob.glob(self.data_path)[6:]

        

        # random.shuffle 打乱列表

        # https://docs.python.org/3/library/random.html#random.shuffle

        random.shuffle(self.data_list)

        

        # 初始化指针

        self.ptr = 0

        

    def get_batch(self, batchsize):

        """

        取出batchsize张图片

        """

        if (self.ptr + batchsize >= len(self.data_list)):

            # 列表中图片已经全部被取完

            # 先把列表里的加进来

            batch = [load_image(x, self.image_size) for x in self.data_list[self.ptr:]]

            rest = self.ptr + batchsize - len(self.data_list)

            

            # 重新初始化列表

            self.__init_list()

            

            # 再加剩下的

            batch.extend([load_image(x, self.image_size) for x in self.data_list[:rest]])

            self.ptr = rest

            self.epoch += 1

        else:

            # 已经够了

            batch = [load_image(x, self.image_size) for x in self.data_list[self.ptr:self.ptr + batchsize]]

            self.ptr += batchsize

        

        return self.epoch, batch

    

    def get_val(self):

        return np.array([load_image(x, self.image_size) for x in glob.glob(self.data_path)[:6]])

    

#     def get_pics(self, num):

#         """

#         取出num张图片，用于快照

#         不会影响队列

#         """

#         return np.array([load_image(x, self.image_size) for x in random.sample(self.data_list, num)])



def arr2image(X):

    """

    将RGB值从[-1,1]重新转回[0,255]的整数

    """

    int_X = (( X + 1) / 2 * 255).clip(0,255).astype('uint8')

    return array_to_img(int_X)



def generate(img, fn):

    """

    将一张图片img送入生成网络fn中

    """

    r = fn([np.array([img])])[0]

    return arr2image(np.array(r[0]))

def build_generator(inputs, image_size):

    x1 = Conv2D(64, kernel_size = 3, padding = "same", kernel_initializer = conv_init, strides = 2)(inputs)

#     x1 = BatchNormalization()(x1, training = 1)

    x2 = LeakyReLU(alpha = 0.2)(x1)

    x2 = Conv2D(128, kernel_size = 3, padding = "same", strides = 2, kernel_initializer = conv_init,use_bias = False)(x1)

    x2 = BatchNormalization()(x2, training = 1)

    x3 = LeakyReLU(alpha = 0.2)(x2)

    x3 = Conv2D(256, kernel_size = 3, padding = "same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x2)

    x3 = BatchNormalization()(x3, training = 1)

    x4 = LeakyReLU(alpha = 0.2)(x3)

    x4 = Conv2D(512, kernel_size = 3, padding = "same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x3)

    x4 = BatchNormalization()(x4, training = 1)

    x4_1 = LeakyReLU(alpha = 0.2)(x4)

    x4_1 = Conv2D(1024, kernel_size = 3, padding = "same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x4_1)

    x4_1 = BatchNormalization()(x4_1)

    x4_1 = LeakyReLU(alpha = 0.2)(x4_1)

    x4_1 = Conv2DTranspose(512, kernel_size = 3, padding="same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x4_1)

    x4_1 = BatchNormalization()(x4_1, training = 1)

    x4_1 = Dropout(0.5)(x4_1)

    x4_2 = Concatenate()([x4, x4_1]) # 跳层连接

    x4_2 = Activation("relu")(x4_2)

    x4_2 = Conv2DTranspose(256, kernel_size = 3, padding="same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x4_2)

    x4_2 = BatchNormalization()(x4_2, training = 1)

    x4_2 = Dropout(0.5)(x4_2)

    x5 = Concatenate()([x3, x4_2]) # 跳层连接

    x5 = Activation("relu")(x5)

    x5 = Conv2DTranspose(128, kernel_size = 3, padding="same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x5)

    x5 = BatchNormalization()(x5, training = 1)

    x5 = Dropout(0.5)(x5)

    x6 = Concatenate()([x2, x5]) # 跳层连接

    x6 = Activation("relu")(x6)

    x6 = Conv2DTranspose(64, kernel_size = 3, padding="same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x6)

    x6 = BatchNormalization()(x6, training = 1)

    x6 = Dropout(0.5)(x6)

    x7 = Concatenate()([x1, x6]) # 跳层连接

    x7 = Activation("relu")(x7)

    x7 = Conv2DTranspose(64, kernel_size = 3, padding="same", strides = 2, kernel_initializer = conv_init, use_bias = False)(x7)

    x7 = BatchNormalization()(x7, training = 1)

    x7 = Activation("relu")(x7)

    x7 = Conv2D(3, kernel_size = 3, padding="same", kernel_initializer = conv_init)(x7)

    x7 = Activation("tanh")(x7)

    return Model(inputs=inputs, outputs=x7)
def build_discriminator(input_a, input_b):

    # 构建判别网络

    kernel_size = 4

    layer_filters = [64, 128, 128, 256, 256,]



    x = Concatenate()([input_a, input_b])

    x = Conv2D(filters=64, kernel_initializer = conv_init,

                   kernel_size=kernel_size,

                   strides=2,

                   padding='same')(x)

    x = LeakyReLU(alpha = 0.2)(x)

    for filters in layer_filters:

        x = Conv2D(filters=filters, kernel_initializer = conv_init,

                   kernel_size=kernel_size,

                   strides=2,

                   padding='same',

                   use_bias = False

                  )(x)

        x = LeakyReLU(alpha = 0.2)(x)

        x = BatchNormalization()(x, training = 1)

    x = Conv2D(filters=1, kernel_initializer = conv_init,

               kernel_size=kernel_size,

               strides=2,

               padding='same')(x)    

    x = Flatten()(x)

    x = Dense(1)(x)

    x = Activation('sigmoid')(x)

    discriminator = Model([input_a, input_b], x)

    return discriminator
LATENT_SIZE= 100

IMAGE_SIZE = 128

TRAIN_STEPS = 100000

EPOCH = 50



# 构建整体网络

# 创建判别网络

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input_a = Input(shape=input_shape)

input_b = Input(shape=input_shape)

discriminator = build_discriminator(input_a, input_b)

optimizer = Adam(lr=2e-4, beta_1=0.5)

discriminator.compile(loss='binary_crossentropy',

                      optimizer=optimizer,

                      metrics=['accuracy'])



# 创建生成网络，并且和判别网络相连

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

input_a = Input(shape=input_shape)

generator = build_generator(input_a, IMAGE_SIZE)

optimizer = Adam(lr=2e-4, beta_1=0.5)

discriminator.trainable = False

fake = generator(input_a)

adversarial = Model(inputs=input_a, outputs=[discriminator([input_a, fake]), fake])



adversarial.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 20],

                    optimizer=optimizer)
BATCH_SIZE = 1

from IPython.display import display



def train():

    dataset = DataSet("../input/contours2cats/contours2cats/contours2cats/*.png", IMAGE_SIZE)

    epoch = 0

    iteration = 0  

    while epoch < EPOCH:

        iteration += 1

        epoch, data = dataset.get_batch(BATCH_SIZE)

        data_a = np.array(data)[:, 0, :, :, :]

        real_images = np.array(data)[:, 1 ,:,:,:]

        fake_images = generator.predict(data_a)

        loss1, acc1 = discriminator.train_on_batch([data_a, real_images], np.ones([BATCH_SIZE, 1]))

        loss2, acc2 = discriminator.train_on_batch([data_a, fake_images], np.zeros([BATCH_SIZE, 1]))

        #训练判别器

        

        log = "%d %d: [discriminator loss: %f, acc: %f]" % (epoch, iteration, (loss1  + loss2)/2, (acc1+acc2)/2)

        

        y = np.ones([BATCH_SIZE, 1])

        loss = adversarial.train_on_batch(data_a, [y, real_images])

        # 训练生成器

        log = "%s [adversarial loss: %s" % (log, loss)

        

        if (iteration % 20 == 0):

            print(log)

        

        if (iteration % 200 == 0):

            data_val = dataset.get_val()

            data_a_val = np.array(data_val)[:, 0, :, :, :]

            real_images_val = np.array(data_val)[:, 1 ,:,:,:]

            out_val = np.array(np.concatenate([generator.predict(np.array([x])) for x in data_a_val]))

            d1 = np.concatenate(out_val, axis = 1)

            d2 = np.concatenate(data_a_val, axis = 1)

            d3 = np.concatenate(real_images_val, axis = 1)

            d = np.concatenate([d1, d2, d3])

            display(arr2image(d))

        

        if (epoch % 500 == 0):

            generator.save("g_model.h5")

train()