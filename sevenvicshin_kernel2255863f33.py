from keras.layers.merge import concatenate

from keras.utils import to_categorical

from keras.layers import LeakyReLU #可以有负数值，有利于网络训练时趋于稳定 

from keras.layers import BatchNormalization #批量标准化

from keras.optimizers import RMSprop #优化器

import math

def build_cgan_discriminator(inputs, y_labels, image_size):

  '''

  识别图片，并将图片与输入的one-hot-vector关联起来

  它使用三层卷积网络从图片中抽取信息，最后使用sigmoid函数输出

  图片是否为真的概率

  y_labels： 设定的独热编码

  inputs ：此处为图像数据 ，generator的输出就是discriminator的输入



  '''

  kernel_size = 5

  layer_filters = [32, 64, 128, 256]# 卷积层过滤器的大小

  x = inputs

  y = Dense(image_size * image_size)(y_labels)

  y = Reshape((image_size, image_size, 1))(y)

  #把图片数据与one-hot-vector拼接起来,这里是唯一与前面代码不同之处

  x = concatenate([x, y])

 #设定卷积核的步伐

  for filters in layer_filters:

    if filters == layer_filters[-1]:

      strides = 1

    else:

      strides = 2

   #我们使用激活函数LeakyReLu而不是以前的Relu，输入值小于0，不返回0.根据经验有利于网络训练时趋于稳定 

    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters = filters,

              kernel_size = kernel_size,

              strides = strides,

              padding = 'same')(x)

    

  x = Flatten()(x)

  x = Dense(1)(x)

      #输出真假值，范围在0~1 #辅助分类器，输出图片分类

  x = Activation('sigmoid')(x)

  discriminator = Model([inputs, y_labels], x ,

                       name = 'discriminator')

  return discriminator
def  build_cgan_generator(inputs, y_labels, image_size):

  '''

  生成者网络在构造图片时，需要将输入向量与对应的one-hot-vector结合在一起考虑

  生成者网络输入给它的一维随机向量相当于输入解码器网络的编码向量，

  解码器网络将一维向量反向构造成图片所对应的二维向量，这也是生成者要做的工作

  参数说明：

  image_size 图片大小

  inputs 输入向量 一般是随机生成的一维随机向量

  '''

  image_resize = image_size // 4 #取整

  kernel_size = 5 # 卷积核大小

  layer_filters = [128, 64, 32, 1]  #反卷积层过滤器大小

  #将输入向量与One-hot-vector结合在一起

  x = concatenate([inputs, y_labels], axis = 1)

  x = Dense(image_resize * image_resize * layer_filters[0])(x) # 随机生成的一维随机向量 相乘为神经元节点

  x = Reshape((image_resize, image_resize, layer_filters[0]))(x)

   #构造三层反卷积网络 ，大于32的时候取2，否则取1 ，进入循环处理

  for filters in layer_filters:

    if filters > layer_filters[-2]:

      strides = 2

    else:

      strides = 1

    '''

    生成器的输入有两个,一个是高斯噪声noise,一个是由我希望生成的图片的label信息,

    通过embedding的方法把label调整到和噪声相同的维度,

    乘起来这样便使得noise的输入是建立在label作为条件的基础上

    '''

    #使用batch normalization将输入反卷积网络的向量做预处理，没有这一步GAN的训练就会失败

    x = BatchNormalization()(x)#批量标准化 

    x = Activation('relu')(x)

    x = Conv2DTranspose(filters = filters,

                       kernel_size = kernel_size,

                       strides = strides,

                       padding = 'same')(x)

  x = Activation('sigmoid')(x)

    # 构造生成器

  generator = Model([inputs, y_labels], x, name='generator')

  return generator
def  train_cgan(models, data, params):

  '''

  训练时需要遵守的步骤是，先冻结生成者网络，把真实图片输入到识别者网络，训练识别者网络识别真实图片。

  然后冻结识别者网络，让生成者网络构造图片输入给识别者网络识别，根据识别结果来改进生成者网络

  参数说明：

  '''

  generator,discriminator,adversarial = models

 #获取图片数据以及图片对应数字的one-hot-vector

  x_train, y_train = data

  batch_size, latent_size, train_steps, num_labels, model_name = params

  save_interval = 500

      #构造给生成者网络的一维随机向量 生成16行数据一维100分量 

  noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])

  '''

  np.eye产生对角矩阵,例如np.eye(3) = [[1,0,0], [0,1,0], [0,0,1]],

  于是np.eye(3)[2, 3, 1] = [[0,1,0], [0,0,1], [1,0,0]]

  '''

  noise_class = np.eye(num_labels)[np.arange(0, 16) % num_labels]

  train_size = x_train.shape[0]

  print(model_name, "Labels for generated images: ", np.argmax(noise_class, 1))

  for i in range(train_steps):

     #先训练识别者网络,将真实图片和伪造图片同时输入识别者，让识别者学会区分真假图片

    # 随机获取batch_size =64 张图片

    rand_indexes = np.random.randint(0, train_size, size = batch_size)

    real_images = x_train[rand_indexes] #真实图片

    #增加图片对应的one-hot-vector

    real_labels = y_train[rand_indexes]

    # 生成64张虚假图片噪声向量

    noise = np.random.uniform(-1.0, 1.0, size = [batch_size, latent_size])

    #增加构造图片对应的one-hot-vector

    fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]

    fake_images = generator.predict([noise, fake_labels])

    #把真实图片和虚假图片连接起来

    x = np.concatenate((real_images, fake_images))

    #将真实图片对应的one-hot-vecotr和虚假图片对应的One-hot-vector连接起来

    y_labels = np.concatenate((real_labels, fake_labels))

    

    y = np.ones([2 * batch_size, 1])

    #上半部分图片为真，下半部分图片为假

    y[batch_size:, :] = 0.0

    #先训练识别者网络，这里需要将图片及对应的one-hot-vector输入  小批量分块训练 打印损失和精确度

    loss, acc = discriminator.train_on_batch([x, y_labels], y)

    log = "%d: [discriminator loss : %f, acc: %f]" % (i, loss, acc)

    '''

    冻结识别者网络，构造随机一维向量以及指定数字的one-hot-vector输入生成者

    网络进行训练

    '''

    noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])

    fake_labels = np.eye(num_labels)[np.random.choice(num_labels, batch_size)]

    y = np.ones([batch_size, 1])

     #训练生成者时需要使用到识别者返回的结果，因此我们从两者连接后的网络进行训练

    loss, acc = adversarial.train_on_batch([noise, fake_labels], y)

    log = "%s [adversarial loss :%f, acc: %f]" % (log, loss, acc)

    if (i + 1) % save_interval == 0:

      print(log)

      # 隔保存间隔save_interval 最终结束绘制图片，保存模型

      if (i+1) == train_steps:

        show = True

      else:

        show = False

     # 将生成者构造的图片绘制出来

      plot_images_cgan(generator, 

                 noise_input = noise_input,

                 noise_class = noise_class,

                 show = show,

                 step = (i+1),

                 model_name = model_name)

      generator.save(model_name + ".h5")  #将生成者当前的网络参数存储成文件
def  build_and_train_models_cgan():

  (x_train, y_train), (_,_) = mnist.load_data()

  image_size = x_train.shape[1]

  x_train = np.reshape(x_train, [-1, image_size, image_size, 1])

  x_train = x_train.astype('float32') / 255

  #获得要生成数字的最大值

  num_labels = np.amax(y_train) + 1

  #转换为one-hot-vector

  y_train = to_categorical(y_train)

  

  #model_name = "/content/gdrive/My Drive/cgan_mnist"

  model_name = "/"

  latent_size = 100

  batch_size = 64

  train_steps = 10000

  lr = 2e-4

  decay = 6e-8

  input_shape = (image_size, image_size, 1)

  label_shape = (num_labels, )

  inputs = Input(shape=input_shape, name='discriminator_input')

  labels = Input(shape=label_shape, name = 'class_labels')

  #构建识别者网络时要传入图片对应的One-hot-vector

  discriminator = build_cgan_discriminator(inputs, labels, image_size)

  optimizer = RMSprop(lr=lr, decay = decay)

  discriminator.compile(loss='binary_crossentropy', 

                         optimizer=optimizer,

                         metrics = ['accuracy'])

  

  input_shape = (latent_size,)

  inputs = Input(shape=input_shape, name='z_input')

  #构造生成者时也要传入one-hot-vector

  generator = build_cgan_generator(inputs, labels, image_size)

  optimizer = RMSprop(lr = lr*0.5, decay = decay * 0.5)

  #将生成者和识别者连接起来时要冻结识别者

  discriminator.trainable = False

  outputs = discriminator([generator([inputs, labels]), labels])

  adversarial = Model([inputs, labels], outputs, name = model_name)

  adversarial.compile(loss ='binary_crossentropy',

                      optimizer = optimizer,

                      metrics = ['accuracy'])

  models = (generator, discriminator, adversarial)

  data = (x_train, y_train)

  params = (batch_size, latent_size, train_steps, num_labels, model_name)

  train_cgan(models, data, params)
def  plot_images_cgan(generator, 

                 noise_input, 

                 noise_class,

                 show = False,

                step = 0,

                model_name = ''):

  os.makedirs(model_name, exist_ok = True)

  filename = os.path.join(model_name, "%05d.png" % step)

  images = generator.predict([noise_input, noise_class])

  print(model_name, "labels for generated images: ", np.argmax(noise_class, 

                                                               axis =1))

  plt.figure(figsize = (2.2, 2.2))

  num_images = images.shape[0]

  image_size = images.shape[1]

  rows = int(math.sqrt(noise_input.shape[0]))

  for i in range(num_images):

    plt.subplot(rows, rows, i + 1)

    image = np.reshape(images[i], [image_size, image_size])

    plt.imshow(image, cmap= 'gray')

    plt.axis('off')

    

  plt.savefig(filename)

  if show:

    plt.show()

  else:

    plt.close('all')
from keras.layers import LeakyReLU

from keras.layers import BatchNormalization

from keras.optimizers import RMSprop

import math

from keras.layers import Activation, Dense, Input

from keras.datasets import mnist

from keras.layers import LeakyReLU

from keras.layers import Activation

from keras.optimizers import RMSprop

import os

import numpy as np



from keras.layers import Dense, Input

from keras.layers import Conv2D, Flatten

from keras.layers import Reshape, Conv2DTranspose

from keras.models import Model

from keras.datasets import mnist

from keras import backend as K



import numpy as np

import matplotlib.pyplot as plt



build_and_train_models_cgan()