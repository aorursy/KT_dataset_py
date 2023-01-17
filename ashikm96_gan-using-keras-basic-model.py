import tensorflow as tf
tf.enable_eager_execution()

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from scipy import misc
from IPython import display
from keras import backend as K
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction =1

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import tensorflow as tf
tf.enable_eager_execution()
tf.__version__
#AUTOTUNE = tf.data.experimental.AUTOTUNE
#tf.test.gpu_device_name()
tf.Session(config=tf.ConfigProto(log_device_placement=True))

import pathlib
#data_root = tf.keras.utils.get_file('shoes','full_images_dest', untar=True)
data_root = pathlib.Path('../full_images_dest')
print(data_root)
import random
all_image_paths = [str(item) for item in data_root.iterdir()]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_files = []
#y_train = []
i=0
for _file in all_image_paths:
    train_files.append(_file)
    #label_in_file = _file.find("_")
    #y_train.append(int(_file[0:label_in_file]))
    
print("Files in train_files: %d" % len(train_files))


channels = 3
#nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), 112, 112,channels),
                     dtype=np.float32)

i = 0
for _file in train_files:
    #img = load_img( _file)  # this is a PIL image
    #img.thumbnail((image_width, image_height))
    # Convert to Numpy Array 
    img = PIL.Image.open(_file)
    x = img.resize((112,112), PIL.Image.ANTIALIAS)
    x = img_to_array(x)  
    #x = x.reshape((x.shape[0],x.shape[1],3))
    # Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 5000 == 0:
        print("%d images to array" % i)
print("All images to array!")
BUFFER_SIZE = 50025
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7*7*512, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
      
    model.add(tf.keras.layers.Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512) # Note: None is the batch size
    
    model.add(tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 256)  
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 128)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 64)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    
    model.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 32)    
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    

    model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 112,112,3)
  
    return model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
     
    return model
generator = make_generator_model()
discriminator = make_discriminator_model()
def generator_loss(generated_output):
    return tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output))
train_vars = tf.trainable_variables()
def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output))

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output))

    total_loss = real_loss + generated_loss

    return total_loss
generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
EPOCHS = 1000
noise_dim = 100
num_examples_to_generate =16 

# We'll re-use this random vector used to seed the generator so
# it will be easier to see the improvement over time.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])

def train_step(images):
   # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        generated_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(generated_output)
        disc_loss = discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))
def train(dataset, epochs):  
    for epoch in range(epochs):
        start = time.time()
        for images in dataset:
            train_step(images) 
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                           epoch + 401,
                           random_vector_for_generation)

        if((epoch+1)%10==0):
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec'.format(epoch + 401,
                                                  time.time()-start))
    # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                       epochs,random_vector_for_generation)
def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)
    predictions=predictions.numpy()
    #print(predictions[0].shape)
    fig = plt.figure(figsize=(12,12))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :,:] * 127.5 + 127.5).astype(np.uint8))
        plt.axis('off')
    if((epoch)%10 ==0):
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
%%time
train(train_dataset, EPOCHS)
generate_and_save_images(generator,50,random_vector_for_generation)
