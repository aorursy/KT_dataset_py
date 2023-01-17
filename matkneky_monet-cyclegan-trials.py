!pip install -q git+https://github.com/tensorflow/docs
#IMPORTS

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

import tensorflow_addons as tfa

from tensorflow.keras import layers

from tensorflow.keras.callbacks import History





import os, time

from kaggle_datasets import KaggleDatasets

from IPython.display import clear_output



import tensorflow_docs.vis.embed as embed

import PIL

from IPython import display

import imageio



import shutil
# Taken from https://www.kaggle.com/forwet/unpaired-data-cyclicgan-awesome-monets

# Adapted to CUT

# Configuration

class Configuration:

    """Class containing most of the parameters or hyperparameters used

    throughout the notebook."""

    

    epochs = 30

    MONET_TFREC = "/monet_tfrec/*.tfrec"

    MONET_JPG = "/monet_jpg/*.jpg"

    PHOTO_TFREC = "/photo_tfrec/*.tfrec"

    PHOTO_JPG = "/photo_jpg/*.jpg"

    BATCH_SIZE = 8

    IMAGE_SIZE = [256, 256, 3]

    BUFFER = 10000

    steps_per_epoch = 0

    

    #In CUT

    lambda_cycle = 10

    lambda_id = 0.5

    

    #In CUT

    # Weights initializer for the layers.

    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    # Gamma initializer for instance normalization.

    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    

    #Original

    #loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    

    #In CUT

    loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)



cfg = Configuration()

monet_jpg = tf.io.gfile.glob("../input/gan-getting-started/monet_jpg/*.jpg")

cfg.steps_per_epoch = len(monet_jpg)
# TPU Setup

# Taken from Kaggle's Tutorial

# https://www.kaggle.com/amyjang/monet-cyclegan-tutorial



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Device:', tpu.master())

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except:

    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)



AUTOTUNE = tf.data.experimental.AUTOTUNE
# Taken from https://www.kaggle.com/forwet/unpaired-data-cyclicgan-awesome-monets

# To do - CUT processing from publication

class MonetDataset:

        def __init__(self,config):

            """Creates a data of TFRecord files."""

            self.cfg = config

            # Specifing 'gan-getting-started' is mendatory as we

            # load also the output datas from input folder

            gcs_path = KaggleDatasets().get_gcs_path('gan-getting-started')

            self.monet_files = tf.io.gfile.glob(gcs_path+self.cfg.MONET_TFREC)

            self.photo_files = tf.io.gfile.glob(gcs_path + self.cfg.PHOTO_TFREC)

            

        def decode_image(self, image):

            """Function to preprocess the image prior to training."""

            img = tf.image.decode_jpeg(image, channels=3)

            img = tf.cast(img, tf.float32)

            img = img/127.5 - 1

            img = tf.reshape(img, [*self.cfg.IMAGE_SIZE])

            return img

        

        def read_tfrecord(self, instance):

            """Function to extract data from TFRecordDataset Instance."""

            tfrecordformat = {

                    "image_name": tf.io.FixedLenFeature([], tf.string),

                    "image": tf.io.FixedLenFeature([], tf.string),

                    "target": tf.io.FixedLenFeature([], tf.string)

                   }

            example = tf.io.parse_single_example(instance, tfrecordformat)

            return self.decode_image(example["image"])

        

        def prepare_dataset(self, monet=True):

            """Main function to prepare the input pipeline.

            Args: 

            monet- bool value

            Determines if we wanna generate monet dataset or the photo dataset"""

            dataset = tf.data.TFRecordDataset(self.monet_files if monet else self.photo_files, num_parallel_reads=AUTOTUNE)

            dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTOTUNE)

            dataset = dataset.map(self.random_jitter, num_parallel_calls=AUTOTUNE)

            dataset = dataset.repeat()

            dataset = dataset.shuffle(self.cfg.BUFFER)

            dataset = dataset.batch(self.cfg.BATCH_SIZE)

            dataset = dataset.prefetch(AUTOTUNE)

            return dataset



        def random_crop(self, image):

            """Function to perform random cropping."""

            image = tf.image.random_crop(image, [*self.cfg.IMAGE_SIZE])

            return image

        

        def random_jitter(self, image):

            """Function to perform random jittering."""

            image = tf.image.resize(image, [286, 286])

            image = self.random_crop(image)

            

            if tf.random.uniform([], 0, 1) > 0.5:

                image = tf.image.random_flip_left_right(image)

            return image

        

        

        

        def visualize_data(self, data):

            """Utility function to visualize the samples in the dataset instance being provided."""

            fig, ax = plt.subplots(2, self.cfg.BATCH_SIZE//2, figsize=(8, 4)) # Figsize->W x H

            ax = ax.flatten()

            for i, im in zip(range(self.cfg.BATCH_SIZE), data):

                im = im*0.5 + 0.5

                ax[i].imshow(im)

                ax[i].axis("off")

            plt.show()
# Creating instance of dataset

dataset = MonetDataset(Configuration())



# Creating seperate monet and photo dataset

monet_dataset = dataset.prepare_dataset(monet=True)

photo_dataset = dataset.prepare_dataset(monet=False)



# Checking some Monet examples

dataset.visualize_data(next(iter(monet_dataset)))
# Checking some Photo examples

dataset.visualize_data(next(iter(photo_dataset)))
del dataset
#Issue between TPU and tf.pad - looking into it - MirrorPadGrad

#Error :

#  ...on XLA_TPU_JIT: MirrorPadGrad (No registered 'MirrorPadGrad' OpKernel for XLA_TPU_JIT...



# Definition of two Padding Layers

# [Taken and adapted from Keras tutorial on CycleGAN]

# https://keras.io/examples/generative/cyclegan/

class ReflectionPadding2D(layers.Layer):

    """Implements Reflection Padding as a layer.



    Args:

        padding(tuple): Amount of padding for the

        spatial dimensions.



    Returns:

        A padded tensor with the same type as the input tensor.

    """



    def __init__(self, padding=(1, 1), **kwargs):

        self.padding = tuple(padding)

        super(ReflectionPadding2D, self).__init__(**kwargs)

        

    def compute_output_shape(self, input_shape):

        return(input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])



    def call(self, input_tensor, mask=None):

        padding_width, padding_height = self.padding

        padding_tensor = [

            [0, 0],

            [padding_height, padding_height],

            [padding_width, padding_width],

            [0, 0],

        ]

        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")



class ReplicaPadding2D(layers.Layer):

    """Implements Reflection Padding as a layer.



    Args:

        padding(tuple): Amount of padding for the

        spatial dimensions.



    Returns:

        A padded tensor with the same type as the input tensor.

    """



    def __init__(self, padding=(1, 1), **kwargs):

        self.padding = tuple(padding)

        super(ReplicaPadding2D, self).__init__(**kwargs)

        

    def compute_output_shape(self, input_shape):

        return(input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])



    def call(self, input_tensor, mask=None):

        padding_width, padding_height = self.padding

        padding_tensor = [

            [0, 0],

            [padding_height, padding_height],

            [padding_width, padding_width],

            [0, 0],

        ]

        return tf.pad(input_tensor, padding_tensor, mode="SYMMETRIC")
# Definition of Residual Block

# For Resnet generator as in CUT

# Issue with the Padding impacts here

def residual_block(x,

                   activation,

                   kernel_initializer=cfg.kernel_init,

                   kernel_size=(3, 3),

                   strides=(1, 1),

                   padding="valid",

                   gamma_initializer=cfg.gamma_init,

                   use_bias=False):

    dim = x.shape[-1]

    input_tensor = x



    # x = ReflectionPadding2D()(input_tensor) #Issue with TPU and tf.pad... looking into it

    x = layers.ZeroPadding2D()(input_tensor)

    x = layers.Conv2D(

        dim,

        kernel_size,

        strides=strides,

        kernel_initializer=kernel_initializer,

        padding=padding,

        use_bias=use_bias,

    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    x = activation(x)



    x = layers.ZeroPadding2D()(x) #Should be ReflectionPadding2D

    x = layers.Conv2D(

        dim,

        kernel_size,

        strides=strides,

        kernel_initializer=kernel_initializer,

        padding=padding,

        use_bias=use_bias,

    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    x = layers.add([input_tensor, x])

    return x
# Downsampling step

# ReflectionPadding should also be included here

def downsample(

    x,

    filters,

    activation,

    kernel_initializer=cfg.kernel_init,

    kernel_size=(3, 3),

    strides=(2, 2),

    padding="same",

    gamma_initializer=cfg.gamma_init,

    use_bias=False,

):

    x = layers.Conv2D(

        filters,

        kernel_size,

        strides=strides,

        kernel_initializer=kernel_initializer,

        padding=padding,

        use_bias=use_bias,

    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    if activation:

        x = activation(x)

    return x
# Downsampling step

# ReplicationPadding should be included here

def upsample(

    x,

    filters,

    activation,

    kernel_size=(3, 3),

    strides=(2, 2),

    padding="same",

    kernel_initializer=cfg.kernel_init,

    gamma_initializer=cfg.gamma_init,

    use_bias=False,

):

    x = layers.Conv2DTranspose(

        filters,

        kernel_size,

        strides=strides,

        padding=padding,

        kernel_initializer=kernel_initializer,

        use_bias=use_bias,

    )(x)

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    if activation:

        x = activation(x)

    return x
# Creating the Res Net generator

# Combining Resnet Blocks, Downsampler, Upsampler

def get_resnet_generator(

    filters=64,

    num_downsampling_blocks=2,

    num_residual_blocks=9,

    num_upsample_blocks=2,

    gamma_initializer=cfg.gamma_init,

    name=None,

):

    img_input = layers.Input(shape=cfg.IMAGE_SIZE, name=name + "_img_input")

    #x = ReflectionPadding2D(padding=(3, 3))(img_input)

    x = layers.ZeroPadding2D(padding=(3, 3))(img_input)

    x = layers.Conv2D(filters, (7, 7), kernel_initializer=cfg.kernel_init, use_bias=False)(

        x

    )

    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    x = layers.Activation("relu")(x)



    # Downsampling

    for _ in range(num_downsampling_blocks):

        filters *= 2

        x = downsample(x, filters=filters, activation=layers.Activation("relu"))



    # Residual blocks

    for _ in range(num_residual_blocks):

        x = residual_block(x, activation=layers.Activation("relu"))



    # Upsampling

    for _ in range(num_upsample_blocks):

        filters //= 2

        x = upsample(x, filters, activation=layers.Activation("relu"))



    # Final block

    x = layers.ZeroPadding2D(padding=(3, 3))(x) #Should be ReflectionPadding2D

    x = layers.Conv2D(3, (7, 7), padding="valid")(x)

    x = layers.Activation("tanh")(x)



    model = keras.models.Model(img_input, x, name=name)

    return model

# Creating the Discriminator

def get_discriminator(

    filters=64, kernel_initializer=cfg.kernel_init, num_downsampling=3, name=None

):

    img_input = layers.Input(shape=cfg.IMAGE_SIZE, name=name + "_img_input")

    x = layers.Conv2D(

        filters,

        (4, 4),

        strides=(2, 2),

        padding="same",

        kernel_initializer=kernel_initializer,

    )(img_input)

    x = layers.LeakyReLU(0.2)(x)



    num_filters = filters

    for num_downsample_block in range(3):

        num_filters *= 2

        if num_downsample_block < 2:

            x = downsample(

                x,

                filters=num_filters,

                activation=layers.LeakyReLU(0.2),

                kernel_size=(4, 4),

                strides=(2, 2),

            )

        else:

            x = downsample(

                x,

                filters=num_filters,

                activation=layers.LeakyReLU(0.2),

                kernel_size=(4, 4),

                strides=(1, 1),

            )



    x = layers.Conv2D(

        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer

    )(x)



    model = keras.models.Model(inputs=img_input, outputs=x, name=name)

    return model
# Call this to use prediction directly

gen_monet = tf.keras.models.load_model('../input/monetcycleganoutputs/gen_monet.h5')

gen_photo = tf.keras.models.load_model('../input/monetcycleganoutputs/gen_photo.h5')

disc_monet = tf.keras.models.load_model('../input/monetcycleganoutputs/disc_monet.h5')

disc_photo = tf.keras.models.load_model('../input/monetcycleganoutputs/disc_photo.h5')
# Call this before training

# Generate the 4 networks

#with strategy.scope():

    # Get the generators

#    gen_monet = get_resnet_generator(name="generator_Monet")

#    gen_photo = get_resnet_generator(name="generator_Photo")



    # Get the discriminators

#    disc_monet = get_discriminator(name="discriminator_Monet")

#    disc_photo = get_discriminator(name="discriminator_Photo")
# Having a look to the generator graph

tf.keras.utils.plot_model(gen_monet, show_shapes=True, dpi=64)
# Having a look to the discriminator graph

tf.keras.utils.plot_model(disc_monet, show_shapes=True, dpi=64)
# CycleGAN Model with our defined models

class CycleGan(keras.Model):

    def __init__(

        self,

        generator_Monet,

        generator_Photo,

        discriminator_Monet,

        discriminator_Photo,

        lambda_cycle=cfg.lambda_cycle,

        lambda_identity=cfg.lambda_id,

    ):

        super(CycleGan, self).__init__()

        self.gen_Monet = generator_Monet

        self.gen_Photo = generator_Photo

        self.disc_Monet = discriminator_Monet

        self.disc_Photo = discriminator_Photo

        self.lambda_cycle = lambda_cycle

        self.lambda_identity = lambda_identity



    def compile(

        self,

        gen_Monet_optimizer,

        gen_Photo_optimizer,

        disc_Monet_optimizer,

        disc_Photo_optimizer,

        gen_loss_fn,

        disc_loss_fn,

        cycl_loss_fn,

        id_loss_fn

    ):

        super(CycleGan, self).compile()

        self.gen_Monet_optimizer = gen_Monet_optimizer

        self.gen_Photo_optimizer = gen_Photo_optimizer

        self.disc_Monet_optimizer = disc_Monet_optimizer

        self.disc_Photo_optimizer = disc_Photo_optimizer

        self.generator_loss_fn = gen_loss_fn

        self.discriminator_loss_fn = disc_loss_fn

        self.cycl_loss_fn = cycl_loss_fn

        self.id_loss_fn = id_loss_fn

        



    def train_step(self, batch_data):

        real_monet, real_photo = batch_data



        with tf.GradientTape(persistent=True) as tape:

            

            # Photo to fake Monet

            fake_monet = self.gen_Monet(real_photo, training=True)

            # Monet to fake Photo

            fake_photo = self.gen_Photo(real_monet, training=True)



            # Cycle (Monet to fake Photo to fake Monet): x -> y -> x

            cycled_monet = self.gen_Monet(fake_photo, training=True)

            # Cycle (Photo to fake Monet to fake Photo) y -> x -> y

            cycled_photo = self.gen_Photo(fake_monet, training=True)



            # Identity mapping

            same_monet = self.gen_Monet(real_monet, training=True)

            same_photo = self.gen_Photo(real_photo, training=True)



            # Discriminator output

            disc_real_monet = self.disc_Monet(real_monet, training=True)

            disc_fake_monet = self.disc_Monet(fake_monet, training=True)



            disc_real_photo = self.disc_Photo(real_photo, training=True)

            disc_fake_photo = self.disc_Photo(fake_photo, training=True)



            # Generator adverserial loss

            gen_Monet_loss = self.generator_loss_fn(disc_fake_photo)

            gen_Photo_loss = self.generator_loss_fn(disc_fake_monet)



            # Generator cycle loss

            cycle_loss_Monet = self.cycl_loss_fn(real_monet, cycled_monet) * self.lambda_cycle

            cycle_loss_Photo = self.cycl_loss_fn(real_photo, cycled_photo) * self.lambda_cycle



            # Generator identity loss

            id_loss_Monet = (

                self.id_loss_fn(real_monet, same_monet)

                * self.lambda_cycle

                * self.lambda_identity

            )

            id_loss_Photo = (

                self.id_loss_fn(real_photo, same_photo)

                * self.lambda_cycle

                * self.lambda_identity

            )



            # Total generator loss

            total_loss_Monet = gen_Monet_loss + cycle_loss_Monet + id_loss_Monet

            total_loss_Photo = gen_Photo_loss + cycle_loss_Photo + id_loss_Photo



            # Discriminator loss

            disc_Monet_loss = self.discriminator_loss_fn(disc_real_monet, disc_fake_monet)

            disc_Photo_loss = self.discriminator_loss_fn(disc_real_photo, disc_fake_photo)



        # Get the gradients for the generators

        grads_Monet = tape.gradient(total_loss_Monet, self.gen_Monet.trainable_variables)

        grads_Photo = tape.gradient(total_loss_Photo, self.gen_Photo.trainable_variables)



        # Get the gradients for the discriminators

        disc_Monet_grads = tape.gradient(disc_Monet_loss, self.disc_Monet.trainable_variables)

        disc_Photo_grads = tape.gradient(disc_Photo_loss, self.disc_Photo.trainable_variables)



        # Update the weights of the generators

        self.gen_Monet_optimizer.apply_gradients(

            zip(grads_Monet, self.gen_Monet.trainable_variables)

        )

        self.gen_Photo_optimizer.apply_gradients(

            zip(grads_Photo, self.gen_Photo.trainable_variables)

        )



        # Update the weights of the discriminators

        self.disc_Monet_optimizer.apply_gradients(

            zip(disc_Monet_grads, self.disc_Monet.trainable_variables)

        )

        self.disc_Photo_optimizer.apply_gradients(

            zip(disc_Photo_grads, self.disc_Photo.trainable_variables)

        )



        return {

            "Monet_generator_loss": total_loss_Monet,

            "Photo_generator_loss": total_loss_Photo,

            "Monet_discriminator_loss": disc_Monet_loss,

            "Photo_discriminator_loss": disc_Photo_loss,

        }

# Call these lines before training

# You might want to check the different photos

# in photo to select which one you want to monitor
#global im_to_gif

#im_to_gif = np.zeros((30,256,256,3))
#photo = next(iter(photo_dataset))
# Taking one Photo from which we will check evolution

num_photo = 0

#plt.imshow(photo[num_photo]*0.5 + 0.5)
# Generate a CallBack function to save

# the prediction, for each epoch, of the Photo above 

#class GANMonitor(keras.callbacks.Callback):

#    """A callback to generate and save images after each epoch"""



#    def on_epoch_end(self, epoch, logs=None):

#        prediction = gen_monet(photo, training=False)[num_photo].numpy()

#        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

#        im_to_gif[epoch] = prediction     
# Call this for before training

# Defining losses, model and compiling

#with strategy.scope():

#    # Define the loss function for the generators

#    def generator_loss_fn(fake):

#        fake_loss = cfg.loss(tf.ones_like(fake), fake)

#        return fake_loss





    # Define the loss function for the discriminators

#    def discriminator_loss_fn(real, fake):

#        real_loss = cfg.loss(tf.ones_like(real), real)

#        fake_loss = cfg.loss(tf.zeros_like(fake), fake)

#        return (real_loss + fake_loss) * 0.5

    

#    def cyclic_loss_fn(real, cycled):

#        return tf.reduce_mean(tf.abs(real - cycled))

    

#    def id_loss_fn(real, same):

#        return tf.reduce_mean(tf.abs(real - same))



    # Create cycle gan model

#    cycle_gan_model = CycleGan(

#        generator_Monet=gen_monet,

#        generator_Photo=gen_photo,

#        discriminator_Monet=disc_monet,

#        discriminator_Photo=disc_photo

#    )



    # Compile the model

#    cycle_gan_model.compile(

#        gen_Monet_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

#        gen_Photo_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

#        disc_Monet_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

#        disc_Photo_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

#        gen_loss_fn=generator_loss_fn,

#        disc_loss_fn=discriminator_loss_fn,

#        cycl_loss_fn=cyclic_loss_fn,

#        id_loss_fn=id_loss_fn

#    )

    # Callbacks

#    plotter = GANMonitor()
# Training

#with strategy.scope():

#    history = cycle_gan_model.fit(tf.data.Dataset.zip((monet_dataset, photo_dataset)),

#                        epochs=cfg.epochs,

#                        steps_per_epoch=cfg.steps_per_epoch,

#                        callbacks=[History(),

#                                   plotter])
# Saving outputs

# Download them afterwards - refresh folder

# You'll have to reupload them through 'Add data'

# to use your own

#gen_monet.save('gen_monet.h5')

#gen_photo.save('gen_photo.h5')

#disc_monet.save('disc_monet.h5')

#disc_photo.save('disc_photo.h5')

#import pickle

#with open('history.pkl','wb') as f:

#    pickle.dump(history.history, f)
def smooth_curve(points, factor=0.8):

    smoothed_points = []

    for point in points:

        if smoothed_points:

            previous = smoothed_points[-1]

            smoothed_points.append(previous * factor + point * (1 - factor))

        else:

            smoothed_points.append(point)

    return smoothed_points





def plot_smoothed_acc_and_loss(history, factor=0.8, load=False):

    monet_g = []

    photo_g = []

    monet_d = []

    photo_d = []

    if load==True:

        for i in range(np.array(history["Monet_generator_loss"]).shape[0]):

            monet_g.append(np.array(history["Monet_generator_loss"][i]).squeeze().mean())

            photo_g.append(np.array(history["Photo_generator_loss"][i]).squeeze().mean())

            monet_d.append(np.array(history["Monet_discriminator_loss"][i]).squeeze().mean())

            photo_d.append(np.array(history["Photo_discriminator_loss"][i]).squeeze().mean())

    else:

        for i in range(np.array(history.history["Monet_generator_loss"]).shape[0]):

            monet_g.append(np.array(history.history["Monet_generator_loss"][i]).squeeze().mean())

            photo_g.append(np.array(history.history["Photo_generator_loss"][i]).squeeze().mean())

            monet_d.append(np.array(history.history["Monet_discriminator_loss"][i]).squeeze().mean())

            photo_d.append(np.array(history.history["Photo_discriminator_loss"][i]).squeeze().mean())

    

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(np.arange(1,31,1),

        smooth_curve(monet_g, factor=factor),

        label="Monet")

    axs[0].plot(np.arange(1,31,1),

        smooth_curve(photo_g, factor=factor),

        label="Photo")

    axs[0].set_title("Smoothed generator loss")

    axs[0].legend()



    axs[1].plot(np.arange(1,31,1),

        smooth_curve(monet_d, factor=factor),

        label="Monet")

    axs[1].plot(np.arange(1,31,1),

        smooth_curve(photo_d, factor=factor),

        label="Photo")

    axs[1].set_title("Smoothed discriminator loss")

    axs[1].legend()



    plt.show()
# Call this if loading outputs

import pickle

with (open("../input/monetcycleganoutputs/history.pkl", "rb")) as openfile:

    history = pickle.load(openfile)
# Switch 'load' to false if you have trained the model

plot_smoothed_acc_and_loss(history, 0.8, load=True)
def create_gif(num_photo=0, load=False):

    if load == True:

        anim_file = '../input/monetcycleganoutputs/CycleGAN.gif'

    else:

        # Creating a gif from each predictions

        anim_file = 'CycleGAN.gif'

        init_pic = np.array(photo[num_photo]*0.5 + 0.5)

        with imageio.get_writer(anim_file, mode='I') as writer:

            # Three first frames are the converted picture

            writer.append_data(init_pic)

            writer.append_data(init_pic)

            writer.append_data(init_pic)

            for i in range(im_to_gif.shape[0]):

                writer.append_data(im_to_gif[i])

                writer.append_data(im_to_gif[i])

                writer.append_data(im_to_gif[i])

            for i in range(int(im_to_gif.shape[0])):

                writer.append_data(im_to_gif[-1])

    return anim_file
# Switch 'load' to false if you have trained the model and put the correct photo number

anim_file = create_gif(num_photo=num_photo,

                       load=True)
def gen_input_img(num_photo=0, load=False):

    fig, ax = plt.subplots(figsize=(5,5))

    

    if load == True:

        img = np.array(PIL.Image.open('../input/monetcycleganoutputs/input_image.png'))

        plt.imshow(img)

        ax.axis("off")

        

    else:

        img = photo[3]*0.5 + 0.5

        plt.imshow(img)

        ax.axis("off")

        plt.title('Input photo')    
# Switch 'load' to false if you have trained the model and put the correct photo number

gen_input_img(num_photo=num_photo,

              load=True)
# Prediction evolution according to epoch

embed.embed_file(anim_file)
photo = next(iter(photo_dataset))

predict_img = gen_monet.predict(tf.expand_dims(photo[0], axis=0))

fig, ax = plt.subplots(1, 2, figsize=(8,8))

ax = ax.flatten()

ax[0].imshow(photo[0]*0.5 + 0.5)

ax[0].set_title('Input photo')

ax[0].axis("off")

out  = (predict_img[0]*127.5 + 127.5).astype(np.uint8)

ax[1].imshow(out)

ax[1].set_title('Output fake Monet')

ax[1].axis("off")
# !mkdir ../images



# photo_jpg = tf.io.gfile.glob("../input/gan-getting-started/photo_jpg/*.jpg")



# for i, image in zip(range(1, len(photo_jpg)+1), photo_dataset):

#     prediction = gen_monet(image, training=False)[0].numpy()

#     prediction = (prediction*127.5 + 127.5).astype(np.uint8)

#     im = PIL.Image.fromarray(prediction)

#     im.save(f"../images/{i}.jpg")

#     if(i%100==0):

#         print(f"Processed {i} images")

        

# shutil.make_archive("/kaggle/working/images", "zip", "/kaggle/images")