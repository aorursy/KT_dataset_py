import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import numpy as np

import matplotlib.pyplot as plt

import glob

import io

%matplotlib inline



##数据准备，生成dataset

(images,labels),(_,_) = keras.datasets.mnist.load_data()

images = images/127.5-1

images = np.expand_dims(images,-1)

dataset = tf.data.Dataset.from_tensor_slices((images,labels))

BATCH_SIZE = 256

noise_dim=50

dataset = dataset.shuffle(60000).batch(BATCH_SIZE)





##生成器

def generate():

    seed = layers.Input(shape=((noise_dim,)))#创建输入图片

    label = layers.Input(shape=(()))#空元组（创建条件），代表单个的值

    #将label转换成向量,是的向量和seed容易合并



    x = layers.Embedding(10,50,input_length = 1)(label)#输入为0—9 10个数，将noise和label合并，完成映射

    x = layers.Flatten()(x)

    x = layers.concatenate([seed,x])

    x = layers.Dense(3*3*128,use_bias=False)(x)

    x = layers.Reshape((3,3,128))(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(64,(3,3),strides = (2,2),use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(32,(3,3),strides = (2,2),use_bias=False,padding='same')(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(1,(3,3),strides = (2,2),use_bias=False,padding='same')(x)

    x = layers.Activation('tanh')(x)

    

    model = keras.Model(inputs = [seed,label],outputs = x)

    return model



##判别器

def discriminator():

    image = layers.Input(shape=(28,28,1))

    

    x = layers.Conv2D(32,(3,3),strides=(2,2),padding='same',use_bias=False)(image)

    x = layers.BatchNormalization()(x)

    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.5)(x)

    

    

    x = layers.Conv2D(32*2,(3,3),strides=(2,2),padding='same',use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.5)(x)

    

    x = layers.Conv2D(32*4,(3,3),strides=(2,2),padding='same',use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = layers.LeakyReLU()(x)

    x = layers.Dropout(0.5)(x)

    

    x = layers.Flatten()(x)

    

    x1 = layers.Dense(1)(x)#真假输出

    x2 = layers.Dense(10)(x)#

    model = keras.Model(inputs = image,outputs = [x1,x2])

    return model



gen = generate()

dis = discriminator()

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)#表明输入的量是未激活的

cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def generate_loss(fake_out,fake_cls_out,label):

    fake_loss = bce(tf.ones_like(fake_out),fake_out)

    cat_loss = cce(label,fake_cls_out)

    return fake_loss+cat_loss

def discriminator_loss(real_out,real_cls_out,fake_out,label):

    real_loss = bce(tf.ones_like(real_out),real_out)

    fake_loss = bce(tf.zeros_like(fake_out),fake_out)

    cat_loss = cce(label,real_cls_out)

    return real_loss+fake_loss+cat_loss



gen_opt = tf.keras.optimizers.Adam(1e-5)

dis_opt = tf.keras.optimizers.Adam(1e-5)



@tf.function#编译成图运算，加速运算

def train_step(image,label):

    size = label.shape[0]

    noise = tf.random.normal([size,noise_dim])

    

    with tf.GradientTape() as gen_tape,tf.GradientTape() as dis_tape:

        gen_imgs = gen((noise,label),training = True)

        fake_out,fake_cls_out = dis(gen_imgs,training = True)

        

        real_out,real_cls_out = dis(image,training = True)

        

        gen_loss = generate_loss(fake_out,fake_cls_out,label)

        dis_loss = discriminator_loss(real_out,real_cls_out,fake_out,label)

    

    gen_grad = gen_tape.gradient(gen_loss,gen.trainable_variables)

    dis_grad = dis_tape.gradient(dis_loss,dis.trainable_variables)

    

    gen_opt.apply_gradients(zip(gen_grad,gen.trainable_variables))

    dis_opt.apply_gradients(zip(dis_grad,dis.trainable_variables))

    

    

def plot_gen_img(model,noise,label,epoch_num):

    print('Epoch',epoch_num)

    gen_img = model((noise,label),training = False)

    gen_img = tf.squeeze(gen_img)#把（28，28，1）---------->(28,28)

    fig = plt.figure(figsize=(10,1))

    for i in range(gen_img.shape[0]):

        plt.subplot(1,10,i+1)

        plt.imshow((gen_img[i,:,:]+1)/2)

        plt.axis('off')

    plt.show()



    

num = 10   

noise_seed = tf.random.normal([num,noise_dim])

label_seed = np.random.randint(0,10,size=(num,1))

print(label_seed.T)



def train(dataset,epochs):

    for epoch in range(epochs):

        for img_batch,label in dataset:

            train_step(img_batch,label)

        if epoch%10==0:

            plot_gen_img(gen,noise_seed,label_seed,epoch)

    plot_gen_img(gen,noise_seed,label_seed,epoch)

    

EPOCHS = 200

train(dataset,EPOCHS)