#导入必要的库

import keras.backend as K



from keras.models import Model



# 导入Keras不同的层

from keras.layers import Conv2D, BatchNormalization, Input, Dropout, Add

from keras.layers import Conv2DTranspose, Reshape, Activation, Dense

from keras.layers import Concatenate, UpSampling2D, Flatten



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

        self.data_list = glob.glob(self.data_path)

        

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

        

    def get_pics(self, num):

        """

        取出num张图片，用于快照

        不会影响队列

        """

        return np.array([load_image(x, self.image_size) for x in random.sample(self.data_list, num)])



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

from IPython.display import display

# 新建一个dataset类

dataset = DataSet("../input/edges2shoes/edges2shoes/train/*.jpg")



# 获取一个batch的数据

# 其中epoch是当前训练轮数

epoch, data = dataset.get_batch(4)



# 这是一个batch的图片

# 其中每一项是一对（两张）图片



# print(data[0]) # 第一对

# print(data[0][0]) # 第一对的A类图



display(arr2image(data[0][0])) # 打印

display(arr2image(data[0][1])) # 打印第一对的B类图 



# display(arr2image(data[1][1]))

# 第二对的B类图





# 以上方法都会影响列表，也就是说这几张图片被读取之后，在这一轮中就不会再次被读取

# 不影响列表地从数据集中获取数据（用于显示中间结果）

data = dataset.get_pics(4)
def build_generator(inputs, image_size):

    # 构建生成网络

    image_resize = image_size // 4

    kernel_size = 5

    layer_filters = [128, 64, 64, 32, 32]



    # 任务二提示：如果输入已经是一张图片了，我们还需要下面这两层吗？

    #           用卷积层替代这两层

    # 目前这两层的作用：将随机噪声转化为边长为image_size四分之一的特征图

    x = Dense(image_resize * image_resize * layer_filters[0])(inputs)

    x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

    

    # 提醒：请时刻注意各层输出的大小，调整卷积层和反卷积层的步长

    # 以下是用循环搭建的，反卷积层->标准化层->激活层，各层的输入定义在layer_filters中

    # 前两层的步长为2，会让图片的边长扩大四倍

    for i in range(len(layer_filters)):

        filters = layer_filters[i]

        if i < 2:

            strides = 2

        else:

            strides = 1

        x = Conv2DTranspose(filters=filters,

                            kernel_size=kernel_size,

                            strides=strides,

                            padding='same')(x)

        x = BatchNormalization()(x, training = 1)

        x = LeakyReLU(alpha = 0.2)(x)



    # 最后一层

    x = Conv2D(3, kernel_size=kernel_size, padding="same")(x)

    x = Activation('tanh')(x)

    generator = Model(inputs, x)

    return generator

def build_discriminator(inputs):

    # 任务一提示： 判别器要接收两个输入，所以这个函数的参数要改成两个，比方说叫input_a和input_b

    # 再使用Concatenate层把这两个输入拼起来（本来每张图片三个通道，现在变成六个通道）（注意axis）

    # Concatenate()([input_a, input_b])

    

    # 构建判别网络

    kernel_size = 5

    layer_filters = [32, 64, 128, 256]



    x = inputs

    for filters in layer_filters:

        if filters == layer_filters[-1]:

            strides = 1

        else:

            strides = 2

        x = Conv2D(filters=filters,

                   kernel_size=kernel_size,

                   strides=strides,

                   padding='same')(x)

        x = LeakyReLU(alpha = 0.2)(x)

    

    # 任务五提示：不需要拉平再过全连接层，只需要用卷积+sigmoid激活

    x = Flatten()(x)

    x = Dense(1)(x)

    x = Activation('sigmoid')(x)

    

    # 这行也要改！inputs改成[input_a, input_b]

    discriminator = Model(inputs, x)

    return discriminator
LATENT_SIZE= 100

IMAGE_SIZE = 256

TRAIN_STEPS = 100000

EPOCH = 10 # 训练轮数



# 构建判别器部分

input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

inputs = Input(shape=input_shape)

# 任务三提示： 判别器需要接收两个输入了，所以需要创建两个输入层



# input_a = Input(shape = input_shape)

# input_b = Input(shape = input_shape)



discriminator = build_discriminator(inputs)

# 完成任务一后，这一句肯定跑不起来了，要改

# discriminator = build_discriminator(input_a, input_b)



optimizer = Adam(lr=1e-5)

discriminator.compile(loss='binary_crossentropy',

                      optimizer=optimizer,

                      metrics=['accuracy'])



# 构建生成器



# LATENT_SIZE是生成随机噪声的维度

# 我们不需要了

# 这个输入的尺寸应该改为图片的尺寸

input_shape = (LATENT_SIZE, )

inputs = Input(shape=input_shape)

generator = build_generator(inputs, IMAGE_SIZE)

optimizer = Adam(lr=1e-5)

discriminator.trainable = False



# 此处要修改，discriminator接受什么参数？

adversarial = Model(inputs, 

                    discriminator(generator(inputs)))

adversarial.compile(loss='binary_crossentropy',

                    optimizer=optimizer,

                    metrics=['accuracy'])
BATCH_SIZE = 16 # 批次大小

from IPython.display import display



def train():

    # 创建一个数据集

    dataset = DataSet("../input/edges2shoes/edges2shoes/train/*.jpg", IMAGE_SIZE)

    epoch = 0

    iteration = 0  

    while epoch < EPOCH:

        iteration += 1

        

        epoch, data = dataset.get_batch(BATCH_SIZE)

        # 获取真实的B类图片

        

        real_images = np.array(data)[:,1,:,:,:]

        # 要获取真实的A类图片可以使用

        # np.array(data)[:,0,:,:,:]

        

        # 以下是训练判别器的部分

        # 产生随机噪声，在我们这里没有用了

        noise = np.random.randn(BATCH_SIZE, LATENT_SIZE)

        

        # 产生的虚假图片（注意修改输入）

        fake_images = generator.predict(noise)

        

        # x是等待被判别的图片，y是（我们期待）的判别结果

        # x = [真图, 真图, 真图, ……, 假图, 假图, 假图]

        # y = [1  , 1  , 1,    ……, 0  , 0  ,  0  ]

        x = np.concatenate((real_images, fake_images))

        y = np.ones([2 * BATCH_SIZE, 1])

        y[BATCH_SIZE:, :] = 0.0

        

        # 任务三提示： 由于判别器接受两个输入，所以单有一个x是不够的

        # 我们要把它改写成：

        # x1 = [A类图片, A类图片, ……, A类图片, A类图片]

        # x2 = [真B,    真B,     ……, 假B,    假B   ]

        # 请注意x1和x2之间的对应关系

        # y不用改

        

        # 把x和y丢进判别器训练

        loss, acc = discriminator.train_on_batch(x, y)

        # 任务三提示

        # loss, acc = discriminator.train_on_batch([x1, x2], y)

        log = "%d %d: [discriminator loss: %f, acc: %f]" % (epoch, iteration, loss, acc)

        

        # 以下是训练生成器的部分

        

        # 产生随机数，也不需要了

        # 应该是以图片为输入

        noise = np.random.randn(BATCH_SIZE, LATENT_SIZE)

        

        # y不需要改

        y = np.ones([BATCH_SIZE, 1])

        

        # 这里也有个noise，要改掉

        loss, acc = adversarial.train_on_batch(noise, y)

        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)

        

        if (iteration % 50 == 0):

            print(log)

        

        if (iteration % 200 == 0):

            # 展示中间结果

            # 不应该用noise来做predict

            # 应该用dataset.get_pics获取图片进行测试

            out = generator.predict(noise)

            d = np.concatenate(out, axis = 1)

            display(arr2image(d))



# 调用训练函数进行训练

train()