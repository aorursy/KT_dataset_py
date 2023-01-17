



import tensorflow as tf

import os

import glob

from matplotlib import pyplot as plt

%matplotlib inline

import time

from IPython import display

 







tf.__version__







imgs_path = glob.glob('../input/anime-sketch-colorization-pair/data/train/*.png')







imgs_path[:3]





len(imgs_path)



#%%使用tf.keras.preprocessing.image.load——image来绘制第一张图片，该图片左边为真实的图像分割出来作为groundtruth，右边语义分割图为生成图作为input



plt.imshow(tf.keras.preprocessing.image.load_img(imgs_path[1]))







#%%read这些图像，并且解码，并且，参数channels=3.因为 它是一张彩色图像，最后返回这些图像



def read_jpg(path):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img, channels=3)

    return img



#%%对于test和train数据是不同的 处理。所以要分开处理。，mask代表语义分割图，image



def normalize(input_image, input_mask):

    input_image = tf.cast(input_image, tf.float32)/127.5 - 1

    input_mask = tf.cast(input_mask, tf.float32)/127.5 - 1

    return input_image, input_mask



#%%



def load_image(image_path):

    image = read_jpg(image_path)

    w = tf.shape(image)[1]

    w = w // 2

    input_image = image[:, :w, :]

    input_mask = image[:, w:, :]

    input_image = tf.image.resize(input_image, (256, 256))

    input_mask = tf.image.resize(input_mask, (256, 256))

    

    if tf.random.uniform(()) > 0.5:

        input_image = tf.image.flip_left_right(input_image)

        input_mask = tf.image.flip_left_right(input_mask)



    input_image, input_mask = normalize(input_image, input_mask)



    return input_mask, input_image



#%%



dataset = tf.data.Dataset.from_tensor_slices(imgs_path)



#%%



train = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)







train







BATCH_SIZE = 8

BUFFER_SIZE = 100



#%%



train_dataset = train.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



#%%



plt.figure(figsize=(10, 10))

for img, musk in train_dataset.take(1):

    plt.subplot(1,2,1)

    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))

    plt.subplot(1,2,2)

    plt.imshow(tf.keras.preprocessing.image.array_to_img(musk[0]))



#%%



imgs_path_test = glob.glob('../input/anime-sketch-colorization-pair/data/val/*.png')



#%%



len(imgs_path_test)



#%%



dataset_test = tf.data.Dataset.from_tensor_slices(imgs_path_test)



#%%



def load_image_test(image_path):

    image = read_jpg(image_path)

    w = tf.shape(image)[1]

    w = w // 2

    input_image = image[:, :w, :]

    input_mask = image[:, w:, :]

    input_image = tf.image.resize(input_image, (256, 256))

    input_mask = tf.image.resize(input_mask, (256, 256))

    

    input_image, input_mask = normalize(input_image, input_mask)



    return input_mask, input_image



#%%



dataset_test = dataset_test.map(load_image_test)







dataset_test = dataset_test.batch(BATCH_SIZE)



#%%



plt.figure(figsize=(10,10))

for img, musk in dataset_test.take(1):

    plt.subplot(1,2,1)

    plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))

    plt.subplot(1,2,2)

    plt.imshow(tf.keras.preprocessing.image.array_to_img(musk[0]))

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):

#    initializer = tf.random_normal_initializer(0., 0.02)



    result = tf.keras.Sequential()

    result.add(

        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',

                               use_bias=False))



    if apply_batchnorm:

        result.add(tf.keras.layers.BatchNormalization())



        result.add(tf.keras.layers.LeakyReLU())



    return result



#%%



def upsample(filters, size, apply_dropout=False):

#    initializer = tf.random_normal_initializer(0., 0.02)



    result = tf.keras.Sequential()

    result.add(

        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,

                                        padding='same',

                                        use_bias=False))



    result.add(tf.keras.layers.BatchNormalization())



    if apply_dropout:

        result.add(tf.keras.layers.Dropout(0.5))



    result.add(tf.keras.layers.ReLU())



    return result

def Generator():

    inputs = tf.keras.layers.Input(shape=[256,256,3])



    down_stack = [

        downsample(32, 3, apply_batchnorm=False), # (bs, 32, 32, 32)

        downsample(64, 3), # (bs, 16, 16, 64)

        downsample(128, 3), # (bs, 8, 8, 128)

        downsample(256, 3), # (bs, 4, 4, 256)

        downsample(512, 3), # (bs, 2, 2, 512)

        downsample(512, 3), # (bs, 1, 1, 512)

    ]



    up_stack = [

        upsample(512, 3, apply_dropout=True), # (bs, 2, 2, 1024)

        upsample(256, 3, apply_dropout=True), # (bs, 4, 4, 512)

        upsample(128, 3, apply_dropout=True), # (bs, 8, 8, 256)

        upsample(64, 3), # (bs, 16, 16, 128)

#         upsample(64, 3，apply_dropout=True),# (bs, 16, 16, 128)

        upsample(32, 3), # (bs, 32, 32, 64)

    ]



#    initializer = tf.random_normal_initializer(0., 0.02)

    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 3,

                                         strides=2,

                                         padding='same',

                                         activation='tanh') # (bs, 64, 64, 3)



    x = inputs



    # Downsampling through the model

    skips = []

    for down in down_stack:

        x = down(x)

        skips.append(x)



    skips = reversed(skips[:-1])



    # Upsampling and establishing the skip connections

    for up, skip in zip(up_stack, skips):

        x = up(x)

        x = tf.keras.layers.Concatenate()([x, skip])



    x = last(x)



    return tf.keras.Model(inputs=inputs, outputs=x)



#%%



generator = Generator()

#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)



#%%



LAMBDA = 10

def generator_loss(disc_generated_output, gen_output, target):

    

    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)



    # mean absolute error

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))



    total_gen_loss = gan_loss + (LAMBDA * l1_loss)



    return total_gen_loss, gan_loss, l1_loss



#%%



def Discriminator():

#    initializer = tf.random_normal_initializer(0., 0.02)



    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')

    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')



    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 64, 64, channels*2)



    down1 = downsample(32, 3, False)(x) # (bs, 32, 32, 32)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Dropout(0.5))

    down2 = downsample(64, 3)(down1) # (bs, 16, 16, 64)

    down3 = downsample(128, 3)(down2) # (bs, 8, 8, 128)

#     model=tf.keras.x

   

    

  

#     Dropout= tf.keras.layers. Dropout()()



    conv = tf.keras.layers.Conv2D(256, 3, strides=1,

                                  padding='same',

                                  use_bias=False)(down3) # (bs, 8, 8, 256)

    result = tf.keras.Sequential()

    result.add(tf.keras.layers.Dropout(0.5))



    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)



    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)



    last = tf.keras.layers.Conv2D(1, 3, strides=1)(leaky_relu) # (bs, 8, 8, 1)



    return tf.keras.Model(inputs=[inp, tar], outputs=last)



#%%



discriminator = Discriminator()

#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)



#%%



loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)



    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)



    total_disc_loss = real_loss + generated_loss



    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

def generate_images(model, test_input, tar):

    prediction = model(test_input, training=True)

    plt.figure(figsize=(10, 10))



    display_list = [test_input[0], tar[0], prediction[0]]

    title = ['Input Image', 'Ground Truth', 'Predicted Image']



    for i in range(3):

        plt.subplot(1, 3, i+1)

        plt.title(title[i])

    # getting the pixel values between [0, 1] to plot it.

        plt.imshow(display_list[i] * 0.5 + 0.5)

        plt.axis('off')

    plt.show()


for example_input, example_target in dataset_test.take(2):

    generate_images(generator, example_input, example_target)

# checkpoint_dir = './training_checkpoints'

# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

#                                  discriminator_optimizer=discriminator_optimizer,

#                                  generator=generator,

#                                  discriminator=discriminator)


EPOCHS = 50

#%%



@tf.function

def train_step(input_image, target, epoch):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(input_image, training=True)

        gen_output = generator(input_image,training = True)

        gen_output = generator(input_image,training = True)



        disc_real_output = discriminator([input_image, target], training=True)

        disc_generated_output = discriminator([input_image, gen_output], training=True)



        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)

        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

#         print(" gen_total_loss" + str(  gen_total_loss)+" , gen_gan_loss"+str( gen_gan_loss)+" ,gen_l1_loss"+str(gen_l1_loss)+" , disc_loss"+str( disc_loss))

            



    generator_gradients = gen_tape.gradient(gen_total_loss,

                                          generator.trainable_variables)

    discriminator_gradients = disc_tape.gradient(disc_loss,

                                               discriminator.trainable_variables)



    generator_optimizer.apply_gradients(zip(generator_gradients,

                                          generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,

                                              discriminator.trainable_variables))

    return gen_total_loss,disc_loss,gen_l1_loss

@tf.function

def train_step_tpu(input_image,target,epoch):

    with strategy.scope():

        with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:

            gen_output = generator(input_image,training = True)

            gen_output = generator(input_image,training = True)

            gen_output = generator(input_image,training = True)

            disc_real_output = discriminator([input_image,target],training = True)

            disc_generate_output = discriminator([input_image,gen_output])

            gen_total_loss ,gen_gen_loss ,gen_l1_loss = generate_loss(disc_generate_output,gen_output,target)

            disc_loss = discriminator_loss(disc_real_output,disc_generate_output)

        

            generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)

            discriminator_gradients = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(generator_gradients,generator.trainable_variables))

            discriminator_optimizer.apply_gradients(zip(discriminator_gradients,discriminator.trainable_variables))
checkpoint_dir = './training_checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
def fit(train_ds,epochs,test_ds):

    for epoch in range(epochs+1):

        if epoch%3== 0:

            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,

                                             discriminator_optimizer=discriminator_optimizer,

                                             generator=generator,

                                             discriminator=discriminator

                                            )

            checkpoint.save(checkpoint_prefix)

           

#         print("Epoch: ",epoch)

        for example_input,example_target in dataset_test.take(1):

                generate_images(generator,example_input,example_target)

        for n,(input_image,target) in train_ds.enumerate():

            if n%8 == 0:

                print('.',end ='')

                gen_total_loss, disc_loss,gen_l1_loss = train_step(input_image,target,epoch)

                print("Epoch: "+str(epoch)+" ,gen_total_loss" + str(gen_total_loss.numpy())+",disc_loss"+str( disc_loss.numpy())+" ,gen_l1_loss"+str(gen_l1_loss.numpy()))

# print("Epoch: "+str(epoch)+" ,gen_total_loss" + str(gen_total_loss.numpy())+",disc_loss"+str( disc_loss.numpy())+" ,gen_l1_loss"+str(gen_l1_loss.numpy()))

#                 print(gen_total_loss.numpy())

            

#             print(gen_total_loss.numpy())

                

        print()

fit(train_dataset, EPOCHS, dataset_test)

    
#%%



AD_EPOCHS = 50



#%%

Tensorboard=Tensorboard(log_dir="./model",histogram_freq=1,write_grades=true,update_freq='epoch')



fit(train_dataset, AD_EPOCHS, dataset_test)



#%%



generator.save('pix2pix0.h5')



#%%



for input_image, ground_true in dataset_test:

    generate_images(generator, input_image, ground_true)



g = Generator()

 

checkpoint = tf.train.Checkpoint(  generator=g)   

checkpoint=checkpoint.restore(tf.train.latest_checkpoint('./save'))



def generate_imagee(model, test_input):

    prediction = model(test_input, training=False)

    plt.figure(figsize=(10, 10))



    display_list = [test_input[0], prediction[0]]

    title = ['Input Image', 'Predicted Image']



    for i in range(2):

        plt.subplot(1, 3, i+1)

        plt.title(title[i])

    # getting the pixel values between [0, 1] to plot it.

        plt.imshow(display_list[i] * 0.5 + 0.5)

        plt.axis('off')

        

    plt.show()

    

    

for example_input, example_target in dataset_test.take(3):

    generate_imagee(generator, example_input)


g = Generator()

 

checkpoint = tf.train.Checkpoint(  generator=g)   

checkpoint=checkpoint.restore(tf.train.latest_checkpoint('./save'))



def generate_imagee(model, test_input):

    prediction = model(test_input, training=False)

    plt.figure(figsize=(10, 10))



    display_list = [test_input[0], prediction[0]]

    title = ['Input Image', 'Predicted Image']



    for i in range(2):

        plt.subplot(1, 3, i+1)

        plt.title(title[i])

    # getting the pixel values between [0, 1] to plot it.

        plt.imshow(display_list[i] * 0.5 + 0.5)

        plt.axis('off')

        

    plt.show()

    

    

for example_input, example_target in dataset_test.take(3):

    generate_imagee(generator, example_input)




     

AD_EPOCHS = 50



#%%

Tensorboard=Tensorboard(log_dir="./model",histogram_freq=1,write_grades=true,update_freq='epoch')



fit(train_dataset, AD_EPOCHS, dataset_test)

%



generator.save('pix2pix0.h5')



#%%



for input_image, ground_true in dataset_test:

    generate_images(generator, input_image, ground_true)





AD_EPOCHS = 50



#%%

Tensorboard=Tensorboard(log_dir="./model",histogram_freq=1,write_grades=true,update_freq='epoch')



fit(train_dataset, AD_EPOCHS, dataset_test)


generator.save('pix2pix0.h5')



#%%



for input_image, ground_true in dataset_test:

    generate_images(generator, input_image, ground_true)


AD_EPOCHS = 50



#%%

Tensorboard=Tensorboard(log_dir="./model",histogram_freq=1,write_grades=true,update_freq='epoch')



fit(train_dataset, AD_EPOCHS, dataset_test)


generator.save('pix2pix0.h5')



#%%



for input_image, ground_true in dataset_test:

    generate_images(generator, input_image, ground_true)
