import tensorflow as tf

from tensorflow.keras import layers

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import glob

import os



import pickle #供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。
import warnings

warnings.filterwarnings("ignore")
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

# tf.config.experimental.set_virtual_device_configuration(gpus[0],

#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

# tf.config.experimental.set_virtual_device_configuration(gpus[0],

#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])


for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_image_filename = glob.glob('/kaggle/input/apple2orange/apple2orange/trainA/*.jpg')

train_cartoon_filename = glob.glob('/kaggle/input/apple2orange/apple2orange/trainB/*.jpg')
train_image_path = [str(path) for path in train_image_filename]#+[str(path) for path in test_image_filename[5:]]

train_cartoon_path = [str(path) for path in train_cartoon_filename]#+[str(path) for path in test_cartoon_filename[5:]]
image_count = len(train_image_path)

image_count
def load_preprosess_image(img_path):

    img_raw = tf.io.read_file(img_path)

    img_tensor = tf.image.decode_jpeg(img_raw, channels=3) 

    img_tensor = tf.image.resize_with_crop_or_pad(img_tensor, 256, 256)

    #decode_image有个缺陷，不会返回shape，所以要用decode_图片类型；channel=3代表彩色图片

    img = tf.image.convert_image_dtype(img_tensor, tf.float32)

    img = img*2 -1

    return img
AUTOTUNE = tf.data.experimental.AUTOTUNE



path_ds1 = tf.data.Dataset.from_tensor_slices(train_image_path)

image_dataset = path_ds1.map(load_preprosess_image, num_parallel_calls=AUTOTUNE) #

path_ds2 = tf.data.Dataset.from_tensor_slices(train_cartoon_path)

cartoon_dataset = path_ds2.map(load_preprosess_image, num_parallel_calls=AUTOTUNE) #, num_parallel_calls=AUTOTUNE



dataset = tf.data.Dataset.zip((image_dataset, cartoon_dataset))

dataset
test_count = int(image_count*0.1)

train_count = image_count - test_count

train_count, test_count
train_dataset = dataset.skip(test_count)

test_dataset = dataset.take(test_count)
BATCH_SIZE = 16

BUFFER_SIZE = image_count
train_datasets = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

train_datasets = train_datasets.prefetch(AUTOTUNE)

test_datasets = test_dataset.batch(BATCH_SIZE)

train_datasets
# test_images = [load_preprosess_image(path).numpy() for path in test_image_path[:5]]

# test_cartoons = [load_preprosess_image(path).numpy() for path in test_cartoon_path[:5]]



# test_images[0].shape
# test_images[0]
# test_images = np.array(test_images)



# test_cartoons= np.array(test_cartoons)

# test_cartoons.shape
def generator_model(): #U-NET #

    #256*256*3

    inputs = layers.Input(shape=((256, 256, 3)))

    

    #128*128*64

    conv1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same')(inputs)

    #x = layers.BatchNormalization()(x)

    conv1 = layers.ReLU()(conv1)

    conv1 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv1)

    #x = layers.Dropout(0.5)(x)

    

    #64*64*128

    conv2 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same')(conv1)

    #x = layers.BatchNormalization()(x)

    conv2 = layers.ReLU()(conv2)

    conv2 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv2)

    #x = layers.Dropout(0.5)(x)

    

    #32*32*256

    conv3 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same')(conv2)

    #x = layers.BatchNormalization()(x)

    conv3 = layers.ReLU()(conv3)

    conv3 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv3)

    #x = layers.Dropout(0.5)(x)

    

    #16*16*512

    conv4 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(conv3)

    #x = layers.BatchNormalization()(x)

    conv4 = layers.ReLU()(conv4)

    conv4 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv4)

    #x = layers.Dropout(0.5)(x)

    

    #8*8512

    conv5 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(conv4)

    #x = layers.BatchNormalization()(x)

    conv5 = layers.ReLU()(conv5)

    conv5 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv5)

    #x = layers.Dropout(0.5)(x)

    

    #4*4*512

    conv6 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(conv5)

    #x = layers.BatchNormalization()(x)

    conv6 = layers.ReLU()(conv6)

    conv6 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv6)

    #x = layers.Dropout(0.5)(x)

    

    #2*2*512

    conv7 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(conv6)

    #x = layers.BatchNormalization()(x)

    conv7 = layers.ReLU()(conv7)

    conv7 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv7)

    #x = layers.Dropout(0.5)(x)

    

    #1*1*512

    conv8 = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same')(conv7)

    #x = layers.BatchNormalization()(x)

    conv8 = layers.ReLU()(conv8)

    conv8 = layers.MaxPooling2D(strides=(2, 2), padding='same')(conv8)

    #x = layers.Dropout(0.5)(x)

    

    #2*2*512

    conv9 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv8)

    conv9 = layers.BatchNormalization()(conv9)

    conv9 = layers.ReLU()(conv9)

    conv9 = layers.Dropout(0.5)(conv9)

    

    #4*4*512

    conv10 = tf.concat([conv9, conv7], 3)

    conv10 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv10)

    conv10 = layers.BatchNormalization()(conv10)

    conv10 = layers.ReLU()(conv10)

    conv10 = layers.Dropout(0.5)(conv10)

    

    #8*8*512

    conv11 = tf.concat([conv10, conv6], 3)

    conv11 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv11)

    conv11 = layers.BatchNormalization()(conv11)

    conv11 = layers.ReLU()(conv11)

    conv11 = layers.Dropout(0.5)(conv11)

    

    #16*16*512

    conv12 = tf.concat([conv11, conv5], 3)

    conv12 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(conv12)

    conv12 = layers.BatchNormalization()(conv12)

    conv12 = layers.ReLU()(conv12)

    

    #32*32*256

    conv13 = tf.concat([conv12, conv4], 3)

    conv13 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(conv13)

    conv13 = layers.BatchNormalization()(conv13)

    conv13 = layers.ReLU()(conv13)

    

    #64*64*128

    conv14 = tf.concat([conv13, conv3], 3)

    conv14 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv14)

    conv14 = layers.BatchNormalization()(conv14)

    conv14 = layers.ReLU()(conv14)

    

    #128*128*64

    conv15 = tf.concat([conv14, conv2], 3)

    conv15 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv15)

    conv15 = layers.BatchNormalization()(conv15)

    conv15 = layers.ReLU()(conv15)

    

    #256*256*3

    conv16 = tf.concat([conv15, conv1], 3)

    conv16 = layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding='same')(conv16)

    conv16 = layers.BatchNormalization()(conv16)

    

    #图片归一化

    outputs = tf.nn.tanh(conv16)

    

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    

    return model
generator1 = generator_model()

generator2 = generator_model()
def discriminator_model():  #inputs_real, inputs_cartoon, reuse=False,alpha=0.01

    input1 = layers.Input(shape=((256, 256, 3)))

    

    #128*128*64

    layer1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input1)

    layer1 = layers.LeakyReLU(alpha=0.01)(layer1)

    

    #64*64*128

    layer2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(layer1)

    layer2 = layers.BatchNormalization()(layer2)

    layer2 = layers.LeakyReLU(alpha=0.01)(layer2)

    

    #32*32*256

    layer3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(layer2)

    layer3 = layers.BatchNormalization()(layer3)

    layer3 = layers.LeakyReLU(alpha=0.01)(layer3)

    

    #16*16*512

    layer4 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(layer3)

    layer4 = layers.BatchNormalization()(layer4)

    layer4 = layers.LeakyReLU(alpha=0.01)(layer4)

    

    flatten = tf.reshape(layer4, (-1, 16*16*512))

    outputs = layers.Dense(1, activation='sigmoid')(flatten)

    

    model = tf.keras.Model(inputs = input1, outputs = outputs)

    

    return model
discriminator1 = discriminator_model()

discriminator2 = discriminator_model()
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True) #真假损失
def generator_loss(fake_out, real_cartoon, fake_cartoon, ):

    fake_loss = bce(tf.ones_like(fake_out), fake_out)

    

    #L1损失

    real_cartoon = tf.reshape(real_cartoon, [-1, 256*256*3])

    fake_cartoon = tf.reshape(fake_cartoon, [-1, 256*256*3])

    L1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(fake_cartoon - real_cartoon)))

    

    return fake_loss + L1_loss
def discriminator_loss(real_out, fake_out):

    real_loss = bce(tf.ones_like(real_out), real_out)

    fake_loss = bce(tf.zeros_like(fake_out), fake_out)

    

    return real_loss + fake_loss
generator_opt = tf.keras.optimizers.Adam(1e-3)

discriminator_opt = tf.keras.optimizers.Adam(1e-3)
@tf.function

def train_step(real_images, real_cartoons):

    

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_cartoons = generator1(real_images, training=True)

        gen_images = generator2(gen_cartoons, training=True)     

        fake_images = generator2(real_cartoons, training=True) 

        fake_cartoons = generator1(fake_images, training=True) 

                

        gen_cartoons_out = discriminator1(gen_cartoons, training=True)

        fake_images_out = discriminator2(fake_images, training=True)

        

        gen_loss = (bce(tf.ones_like(gen_cartoons_out), gen_cartoons_out)+

                   bce(tf.ones_like(fake_images_out), fake_images_out)+

                   tf.reduce_mean(tf.reduce_sum(tf.abs(gen_images - real_images))) +

                   tf.reduce_mean(tf.reduce_sum(tf.abs(fake_cartoons - real_cartoons))))

                    

        real_cartoons_out = discriminator1(real_cartoons, training=True)

        real_images_out = discriminator2(real_images, training=True)

        

        disc_loss = (discriminator_loss(real_cartoons_out, gen_cartoons_out)

                  + discriminator_loss(real_images_out, fake_images_out))

        generator_trainable_variables = generator1.trainable_variables + generator2.trainable_variables

        discriminator_trainable_variables = discriminator1.trainable_variables + discriminator2.trainable_variables

    

    gradient_gen = gen_tape.gradient(gen_loss, generator_trainable_variables)

    gradient_disc = disc_tape.gradient(disc_loss, discriminator_trainable_variables)

    generator_opt.apply_gradients(zip(gradient_gen, generator_trainable_variables))

    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator_trainable_variables))
num = 5 #每个EPOCH生产15*2张图片查看
def generate_plot_image(real_img, real_cart, epoch_num):

    print('Epoch:', epoch_num+1)

    #pre_images = tf.squeeze(pre_images)  #(None, 28, 28, 1)——>(None, 28, 28) plt.imshow((pre_images[i, :, :, 0]+1)/2, cmap='gray')

    gen_cart = generator1(real_img, training=False)

    gen_img = generator2(real_cart, training=False)

    fig = plt.figure(figsize=(6*2, 12))

    for i in range(4):

        plt.subplot(4, 4, i+1)

        plt.imshow((real_img[i, :, : , :]+1)/2)

        plt.axis('off') 

        plt.subplot(4, 4, i+4+1)

        plt.imshow((real_cart[i, :, : , :]+1)/2)

        plt.axis('off') 

        plt.subplot(4, 4, i+4*2+1)

        plt.imshow((gen_cart[i, :, :, :]+1)/2)

        plt.axis('off') 

        plt.subplot(4, 4, i+4*3+1)

        plt.imshow((gen_img[i, :, :, :]+1)/2)

        plt.axis('off') 

    plt.show()
#generator(test_images, training=False)
for test_image_batch, test_cartoon_batch in test_datasets.take(1):

                generate_plot_image(test_image_batch, test_cartoon_batch, -1)
def train(dataset, epochs):

    for epoch in range(epochs):

        for image_batch, cartoon_batch in dataset:

            train_step(image_batch, cartoon_batch)

            print('.', end='')

        if epoch % 5 == 0:

            for test_image_batch, test_cartoon_batch in test_datasets.take(1):

                generate_plot_image(test_image_batch, test_cartoon_batch, epoch)

    for test_image_batch, test_cartoon_batch in test_datasets.take(1):

                generate_plot_image(test_image_batch, test_cartoon_batch, epoch)
EPOCHS = 50
train(train_datasets, EPOCHS)
for test_image_batch, test_cartoon_batch in test_datasets.take(1):

                generate_plot_image(test_image_batch, test_cartoon_batch, -1)