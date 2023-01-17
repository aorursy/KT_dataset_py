# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import numpy as np

import os

import glob

from matplotlib import pyplot as plt

import io
images_path = glob.glob('../input/anime-sketch-colorization-pair/data/train/*.png')

images_path[:3]
plt.imshow(tf.keras.preprocessing.image.load_img(images_path[0]))
def read_img(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_png(img,channels=3)

    return img
def normalize(mask,image):

    mask = tf.cast(mask,tf.float32) / 127.5 -1

    image = tf.cast(image,tf.float32) / 127.5 -1

    return mask,image
def load_image(image_path):

    image = read_img(image_path)

    w = tf.shape(image)[1]

    w = w // 2

    input_mask = image[:,w:,:]

    input_image = image[:,:w,:]

    input_mask = tf.image.resize(input_mask,(256,256))

    input_image = tf.image.resize(input_image,(256,256))

    if tf.random.uniform(()) > 0.5:

        input_mask = tf.image.random_flip_left_right(input_mask)

        input_image = tf.image.random_flip_left_right(input_image)

    

    input_mask,input_image = normalize(input_mask,input_image)

    

    return input_mask,input_image
dataset = tf.data.Dataset.from_tensor_slices(images_path)
dataset = dataset.map(load_image)
dataset
BATCH_SIZE = 5

BUFFE_SIZE = len(images_path)
BUFFE_SIZE/5
dataset = dataset.shuffle(1000).batch(BATCH_SIZE)

dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
for mask,image in dataset.take(1):

    plt.subplot(1,2,1)

    plt.imshow(tf.keras.preprocessing.image.array_to_img( mask[0]) )

    plt.subplot(1,2,2)

    plt.imshow( tf.keras.preprocessing.image.array_to_img( image[0]))
test_path = glob.glob('../input/anime-sketch-colorization-pair/data/val/*.png')

test_path[:3]
def load_image_test(image_path):

    image = read_img(image_path)

    w = tf.shape(image)[1]

    w = w // 2

    input_mask = image[:,w:,:]

    input_image = image[:,:w,:]

    input_mask = tf.image.resize(input_mask,(256,256))

    input_image = tf.image.resize(input_image,(256,256))

    

    input_mask,input_image = normalize(input_mask,input_image)

    

    return input_mask,input_image
dataset_test = tf.data.Dataset.from_tensor_slices(test_path)
dataset_test = dataset_test.map(load_image_test)
dataset_test = dataset_test.batch(BATCH_SIZE)
for mask,image in dataset_test.take(1):

    plt.subplot(1,2,1)

    plt.imshow(tf.keras.preprocessing.image.array_to_img( mask[0]) )

    plt.subplot(1,2,2)

    plt.imshow( tf.keras.preprocessing.image.array_to_img( image[0]))
def downsample(filters,size,apply_bn=True):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters,size,strides=2,padding='same',use_bias=False))

    if apply_bn:

        model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.LeakyReLU())

    return model
def upsample(filters,size,apply_drop=False):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2DTranspose(filters,size,strides=2,padding='same',use_bias=False))

    if apply_drop:

        model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.ReLU())

    return model
def Generator():

    inputs = tf.keras.layers.Input(shape=(256,256,3))

    down_stack = [

        downsample(64,4,apply_bn=False),   #128*128*64

        downsample(128,4),                 #64*64*128

        downsample(256,4),                #32*32*256

        downsample(512,4),                #16*16*512

        downsample(512,4),                #8*8*512

        downsample(512,4),                #4*4*512

        downsample(512,4),                #2*2*512

        downsample(512,4),                #1*1*512

        

    ]

    up_stack = [

        upsample(512,4,apply_drop=True),    #2*2*512

        upsample(512,4,apply_drop=True),    #4*4*512

        upsample(512,4,apply_drop=True),    #8*8*512

        upsample(512,4,apply_drop=True),    #16*16*512

        upsample(256,4),                  #32*32*256

        upsample(128,4),                  #64*64*28

        upsample(64,4),                   #128*126*64  

    ]

    x = inputs

    skips = []

    for down in down_stack:

        x = down(x)

        skips.append(x)

    skips = reversed(skips[:-1])

    

    for up,skip in zip(up_stack,skips):

         x = up(x) 

         x = tf.keras.layers.concatenate([x,skip])   #1*1*1024->......->128*126*128

            

    x = tf.keras.layers.Conv2DTranspose(3,4,strides=2,padding='same',activation='tanh')(x) #255*256*3

    

    return tf.keras.Model(inputs=inputs,outputs=x)
gen_model_path = '../input/anime/generator-v2.h5'

if os.path.exists(gen_model_path):

    generator = tf.keras.models.load_model(gen_model_path)

else:

    generator = Generator()

#tf.keras.utils.plot_model(generator,show_shapes=True)
def Discriminator():

    input = tf.keras.layers.Input(shape=(256,256,3))

    target = tf.keras.layers.Input(shape=(256,256,3))

    

    x = tf.keras.layers.concatenate([input,target])  #256*256*6

    

    x = downsample(64,4,apply_bn=False)(x)    #128*128*64

    

    x = downsample(128,4)(x)    #64*64*128

    

    x = downsample(256,4)(x)    #32*32*256

    

    x = tf.keras.layers.Conv2D(512,4,strides=1,padding='same',use_bias=False)(x) #32*32*512

    

    x = tf.keras.layers.BatchNormalization()(x)

    

    x = tf.keras.layers.LeakyReLU()(x)

    

    x = tf.keras.layers.Conv2D(1,3,strides=1)(x) # （32-3+1）/ 1 =  30*30*1

    

    return tf.keras.Model(inputs=[input,target],outputs=x)
dis_model_path = '../input/anime/discriminator-v2.h5'

if os.path.exists(dis_model_path):

    discriminator = tf.keras.models.load_model(dis_model_path)

else:

    discriminator = Discriminator()



#tf.keras.utils.plot_model(discriminator,show_shapes=True)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 10 #这是一个超参数 增加l1损失的权重
def generator_loss(d_gen_output,gen_output,target):

    gen_loss = loss_fn(tf.ones_like(d_gen_output),d_gen_output)

    

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    

    return gen_loss + LAMBDA*l1_loss
def discriminator_loss(d_real_output,d_gen_output):

    real_loss = loss_fn(tf.ones_like(d_real_output),d_real_output)

    fake_loss = loss_fn(tf.zeros_like(d_gen_output),d_gen_output)

    return real_loss + fake_loss
generator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)

discriminator_optimizer = tf.keras.optimizers.Adam(2e-4,beta_1=0.5)
def generate_image(model,test_input,test_target):

    prediction = model(test_input,training=True)

    display_list = [test_input[0],test_target[0],prediction[0]]

    title = ['Input image','Ground image','Predicted Image ']

    plt.figure(figsize=(15,15))

    

    for i in range(3):

        plt.subplot(1,3,i+1)

        plt.title(title[i])

        plt.imshow(display_list[i]*0.5 + 0.5)

        plt.axis('off')

    plt.show()
EPOCHES = 5
@tf.function

def train_steps(input_image,target,epoch):

    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:

        gen_output = generator(input_image,training=True)

        d_real_out = discriminator([input_image,target],training=True)

        d_gen_out = discriminator([input_image,gen_output],training=True)

        

        gen_loss = generator_loss(d_gen_out,gen_output,target)

        disc_loss = discriminator_loss(d_real_out,d_gen_out)

    #计算梯度

    gen_gradient = gen_tape.gradient(gen_loss,generator.trainable_variables)

    disc_gradient = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    #更新梯度

    generator_optimizer.apply_gradients(zip(gen_gradient,generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(disc_gradient,discriminator.trainable_variables))

#enumerate 将数据对象(如列表、元组或字符串)组合为一个索引序列

def fit(train_ds,epoches,test_ds):

    for epoch in range(epoches+1):

        #if epoch % 10 == 0:

        for example_input,example_target in test_ds.take(1):

            generate_image(generator,example_input,example_target)

        print('Epoche:',epoch)

        for n,(input_image,target) in train_ds.enumerate():

            print('.',end='')

            train_steps(input_image,target,epoch)

        print()
fit(dataset,EPOCHES,dataset_test)
generator.save('generator-v2.h5')

discriminator.save('discriminator-v2.h5') 
#new_model = tf.keras.models.load_model('../input/anime/generator-v2.h5')
#pridiction = generator.predict(dataset_test.take(1))
# print(pridiction.shape)



# plt.imshow(pridiction[4])

# plt.show()
for test_input,test_target in dataset_test.take(5):

    generate_image(generator,test_input,test_target)