import os

import glob

import matplotlib.pyplot as plt

import numpy as np

import time

import tensorflow as tf

from tensorflow.keras import layers

import PIL

from IPython import display

from tensorflow.keras.datasets import mnist

from tensorflow import GradientTape
gpus = tf.config.experimental.list_physical_devices('GPU') 

for gpu in gpus: 

    tf.config.experimental.set_memory_growth(gpu, True)
len(os.listdir('../input/cat-and-dog/training_set/training_set/cats/'))
BUFFER_SIZE=4001

Batch_size=32

#Convert train_images to a tf.data.Dataset

path='../input/cat-and-dog/training_set/training_set/cats/cat.*.jpg'

print(path)

train_dataset=tf.data.Dataset.list_files(tf.io.gfile.glob(path)).shuffle(BUFFER_SIZE)
def decode_img(img):

    img = tf.image.decode_jpeg(img, channels=3) #color images

    img = tf.image.convert_image_dtype(img, tf.float32) 

    #convert unit8 tensor to floats in the [0,1]range

    return tf.image.resize(img, [128, 128])

#resize the image into 224*224 

def process_path(file_path):

    img = tf.io.read_file(file_path)

    img = decode_img(img)

    return img
images=[]

for i in train_dataset:

    images.append(process_path(i))
train_dataset=tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(Batch_size)
train_dataset #Shape
def make_Genrator_Model():

    model=tf.keras.Sequential()

    model.add(layers.Dense(8*8*1024,use_bias=False,input_shape=(256,)))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8,8,1024)))

    

    assert model.output_shape==(None,8,8,1024) #Debug

    

    model.add(layers.Conv2DTranspose(512,(16,16),strides=(2,2),padding='same',use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

        

    model.add(layers.Conv2DTranspose(128,(64,64),strides=(2,2),padding='same',use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(64,(128,128),strides=(2,2),padding='same',use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(3,(128,128),strides=(1,1),padding='same',use_bias=False))

    model.add(layers.Activation(tf.nn.tanh))

    

    return model
def make_decriminator_model():

    model=tf.keras.Sequential()

    

    model.add(layers.Conv2D(64,(64,64),strides=(2,2),padding='same',input_shape=[128,128,3]))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Conv2D(128,(32,32),strides=(2,2),padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

          

    model.add(layers.Conv2D(512,(16,16),strides=(2,2),padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Conv2D(1024,(8,8),strides=(2,2),padding='same'))

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))

    

    model.add(layers.Flatten())

    model.add(layers.Dense(1))

    

    return model
cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
def discriminator_loss(real_output,fake_output):

    real_loss=cross_entropy(tf.ones_like(real_output),real_output)

    fake_loss=cross_entropy(tf.zeros_like(fake_output),fake_output)

    total_loss=real_loss+fake_loss

    return total_loss
def generator_loss(fake_output):

       return cross_entropy(tf.ones_like(fake_output),fake_output)
generator_optimizer=tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer=tf.keras.optimizers.Adam(1e-4)
EPOCHS=20

noise_dim=256

num_ex_to_generate=16

test_random_vectors=tf.random.normal([num_ex_to_generate,noise_dim])

print(test_random_vectors.shape)
def train(dataset,epochs):

    for epoch in range(epochs):

        start=time.time()



        for image_batch in dataset:

            train_step(image_batch)

            

        display.clear_output(wait=True)

        generate_and_save_images(generator,epoch+1,test_random_vectors)

        

        if (epoch+1)%10==0:

            checkpoint.save(file_prefix=checkpoint_prefix)

            

        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start))

    display.clear_output(wait=True)

    generate_and_save_images(generator,epochs,test_random_vectors)
def generate_and_save_images(model,epoch,test_input):

    predictions=model(test_input,training=False)

    fig=plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):

        plt.subplot(4,4,i+1)

        plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')

        plt.axis('off')

    plt.savefig('image_at_epoch{:04d}.png'.format(epoch))

    plt.show()
@tf.function

def train_step(images):

        noise=tf.random.normal([Batch_size,noise_dim])

    

        with GradientTape(persistent=True) as gen_tape,GradientTape(persistent=True) as disc_tape:

            genrated_images=generator(noise,training=True)



            real_output=discriminator(images,training=True)

            fake_output=discriminator(genrated_images,training=True)



            gen_loss=generator_loss(fake_output)

            disc_loss=discriminator_loss(real_output,fake_output)



        gradient_of_generator=gen_tape.gradient(gen_loss,generator.trainable_variables)

        gradient_of_discriminator=gen_tape.gradient(disc_loss,discriminator.trainable_variables)



        generator_optimizer.apply_gradients(zip(gradient_of_generator,generator.trainable_variables))

        generator_optimizer.apply_gradients(zip(gradient_of_discriminator,discriminator.trainable_variables))
discriminator=make_decriminator_model()

generator=make_Genrator_Model()

checkpoint_dir='./training_checkpoints_Gan'

checkpoint_prefix=os.path.join(checkpoint_dir,'ckpt')

checkpoint=tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,

                              generator=generator,discriminator=discriminator)

train(train_dataset,EPOCHS)