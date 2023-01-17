import tensorflow as tf

from   tensorflow import keras

from   tensorflow.keras import layers

import tensorflow_addons as tfa

import tensorflow_datasets as tfds



from   kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import numpy as np





try:

    # Initialize a cluster resolver

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()   

    

    # Show the connection string (device) when creating a session

    print('Device:', tpu.master())   

    

    # Make devices on the cluster available to use

    tf.config.experimental_connect_to_cluster(tpu)     

    # Initialize the tpu device

    tf.tpu.experimental.initialize_tpu_system(tpu)              

    

    # Synchronous training on TPUs and TPU Pods.

    #    While using distribution strategies, the variables created 

    #    within the strategy's scope will be replicated across all 

    #    the replicas and can be kept in sync using all-reduce algorithms

    strategy = tf.distribute.experimental.TPUStrategy(tpu)  

except:

    strategy = tf.distribute.get_strategy()



# Show number of replicas

print('Number of replicas:', strategy.num_replicas_in_sync)     



#show the version of Tensorflow

print(tf.__version__)                                          
# Get the Google Cloud Storage path URI (GCS path) for Kaggle Datasets

GCS_PATH = KaggleDatasets().get_gcs_path()  



# Obtain two lists of files that match the given patterns specified in str() 

MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))

print('# of Monet TFRecord Files:', len(MONET_FILENAMES))

print('# of Photo TFRecord Files:', len(PHOTO_FILENAMES))
# Define the Image height and width (i,e 256 X 256)

IMG_HEIGHT = 256

IMG_WIDTH  = 256 



def decode_image(image):

    # Decode a JPEG-encoded image to a uint8 tensor.

    image = tf.image.decode_jpeg(image, channels=3)

    

    # Normalize the image to the range of the tanh activation function [-1, 1] for 

    # inputs to the generator and discriminator in GAN model 

    # (i.e. the pixel values are divided by (255/2) to form a value of in a range of [0, 2] and then subtract by 1

    # to result into a range of [-1, 1])

    image = (tf.cast(image, tf.float32) / 127.5) - 1        

    

    # Reshape the tensor using (256, 256, 3) where 3 is number of channels: Red, Green, and Blue 

    image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDTH, 3])             

    return image



def read_tfrecord(example):

    # Define TFRecord format 

    tfrecord_format = {

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "image":      tf.io.FixedLenFeature([], tf.string),

        "target":     tf.io.FixedLenFeature([], tf.string)

    }

    # Parse a single example

    example = tf.io.parse_single_example(example, tfrecord_format)  

    # Decode a JPEG image to a uint8 tensor by calling decode_image()

    image = decode_image(example['image'])                          

    return image                                                    # Return an image tensor

# Set it to tf.data.experimental.AUTOTUNE which will prompt 

# the tf.data runtime to tune the value dynamically at runtime.

AUTOTUNE = tf.data.experimental.AUTOTUNE  



def load_dataset(filenames, labeled=True, ordered=False):

    dataset = tf.data.TFRecordDataset(filenames)

    # map a dataset with a mapping function read_tfrecord and 

    # Number of parallel calls is set to AUTOTUNE constant previously defined

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset
def view_image(ds, nrows=1, ncols=5):

    ds_iter = iter(ds)

    # image = next(iter(ds)) # extract 1 from the dataset

    # image = image.numpy()  # convert the image tensor to NumPy ndarrays.



    fig = plt.figure(figsize=(25, nrows * 5.05 )) # figsize with Width, Height

    

    # loop thru all the images (number of rows * number of columns)

    for i in range(ncols * nrows):

        image = next(ds_iter)

        image = image.numpy()

        ax = fig.add_subplot(nrows, ncols, i+1, xticks=[], yticks=[])

        ax.imshow(image[0] * 0.5 + .5) # rescale the data in [0, 1] for display
BATCHSIZE= 1

monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(BATCHSIZE, drop_remainder=True)

photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(BATCHSIZE, drop_remainder=True)
view_image(monet_ds,2, 5)
view_image(photo_ds,2,5)
OUTPUT_CHANNELS = 3



# Define downsample function



def downsample(filters, kernel_size, apply_norm=True):

    # Define a random Gaussian weight initializer with a mean of 0 and a standard deviation of 0.02 for the kernel

    initializer = tf.random_normal_initializer(0., 0.02)

    # Define gamma initializer for Instance Normalization layer (i.e. gamma_initializer in InstanceNormalization)

    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)



    # create a sequential model

    result = keras.Sequential() 

    # add a Conv2D layer: 

    result.add(layers.Conv2D(filters, kernel_size, strides=2, padding='same',

                             kernel_initializer=initializer,

                             use_bias=False)) # no bias vector



    # Apply normization, if True

    if apply_norm:

        # CycleGAN uses instance normalization instead of batch normalization.

        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

        # result.add(tfa.layers.BatchNormalization(gamma_initializer=gamma_init))

    

    # The best practice for GANs (both Generator/Discrimator) is to use Leaky ReLU that allows some values less than zero and 

    # learns where the cut-off should be in each node. 

    # Use the default Negative slope coefficient = 0.3 

    result.add(layers.LeakyReLU(0.3))



    return result
# Define upsampling function



def upsample(filters, kernel_size, apply_dropout=False, dropout=0.5):

    # Define a random Gaussian weight initializer with a mean of 0 and a standard deviation of 0.02 for the kernel

    initializer = tf.random_normal_initializer(0., 0.02)

    # Degome gamma initializer for Instance Normalization layer (i.e. gamma_initializer in InstanceNormalization)

    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)



    # Create a sequential mode

    result = keras.Sequential() 

    # add a Conv2DTranspose layer: 

    result.add(layers.Conv2DTranspose(filters, kernel_size, strides=2,

                                      padding='same',

                                      kernel_initializer=initializer,

                                      use_bias=False))



    # CycleGAN uses instance normalization instead of batch normalization.

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))



    

    # Add a Dropout layer randomly sets input units to 0 with a frequency of rate 0.5 at each step

    if apply_dropout:

        result.add(layers.Dropout(dropout))



    # The best practice for GANs (both Generator/Discrimator) is to use Leaky ReLU that allows some values less than zero and 

    # learns where the cut-off should be in each node. 

    # Use the default Negative slope coefficient = 0.3 

    result.add(layers.LeakyReLU(0.3))



    return result
def Generator():

    inputs = layers.Input(shape=[256,256,3])



    # bs = batch size

    down_stack = [

        downsample(64, 4, apply_norm=False), # (bs, 128, 128, 64)

        downsample(128, 4), # (bs, 64, 64, 128)

        downsample(256, 4), # (bs, 32, 32, 256)

        downsample(512, 4), # (bs, 16, 16, 512)

        downsample(512, 4), # (bs, 8, 8, 512)

        downsample(512, 4), # (bs, 4, 4, 512)

        downsample(512, 4), # (bs, 2, 2, 512)

        downsample(512, 4), # (bs, 1, 1, 512)

    ]



    up_stack = [

        upsample(512, 4, apply_dropout=True, dropout=0.5), # (bs, 2, 2, 1024)

        upsample(512, 4, apply_dropout=True, dropout=0.5), # (bs, 4, 4, 1024)

        upsample(512, 4, apply_dropout=True, dropout=0.5), # (bs, 8, 8, 1024)

        upsample(512, 4), # (bs, 16, 16, 1024)

        upsample(256, 4), # (bs, 32, 32, 512)

        upsample(128, 4), # (bs, 64, 64, 256)

        upsample(64, 4),  # (bs, 128, 128, 128)

    ]



    # Define a random Gaussian weight initializer with a mean of 0 and a standard deviation of 0.02 for the kernel

    initializer = tf.random_normal_initializer(0., 0.02)

    # The Generator uses the hyperbolic tangent (tanh) activation function in the last (outupt) layer 

    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,

                                  strides=2,

                                  padding='same',

                                  kernel_initializer=initializer,

                                  activation='tanh') # (bs, 256, 256, 3)

    

    # Initialize x with the input layer

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

        x = layers.Concatenate()([x, skip])



    x = last(x)



    return keras.Model(inputs=inputs, outputs=x)
def Discriminator(notarget=True):

    # Define a random Gaussian weight initializer with a mean of 0 and a standard deviation of 0.02

    initializer = tf.random_normal_initializer(0., 0.02)

    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)



    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    if notarget:

        x   = inp  # (bs, 256, 256, 3)

    else:

        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x   = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, 3*2)



    down1 = downsample(64,  4, False)(x) # (bs, 128, 128, 64)

    down2 = downsample(128, 4)(down1)   # (bs, 64, 64, 128)

    down3 = downsample(256, 4)(down2)   # (bs, 32, 32, 256)



    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256) 

    

    # Strides is set to 1 and padding is default (i.e. valid) in the Discriminator()

    conv = layers.Conv2D(512, 4, 

                         strides=1,

                         kernel_initializer=initializer,

                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)



    instancenorm = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)



    leaky_relu = layers.LeakyReLU(0.3)(instancenorm)



    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)



    # Strides is set to 1 and padding is default (i.e. valid) in the Discriminator()

    # last is 30 X 30 array

    last = layers.Conv2D(1, 4, 

                         strides=1, 

                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)



    return tf.keras.Model(inputs=inp, outputs=last)
# Open a strategy scope 

# Define the Generators and Discrimators for CycleGAN

with strategy.scope():

    monet_generator     = Generator() # transforms photos to Monet-esque paintings

    photo_generator     = Generator() # transforms Monet paintings to be more like photos

    

    monet_discriminator = Discriminator() # differentiates real Monet paintings and generated Monet paintings

    photo_discriminator = Discriminator() # differentiates real photos and generated photos

class CycleGan(keras.Model):

    def __init__(

        # Attributes

        self,

        monet_generator,

        photo_generator,

        monet_discriminator,

        photo_discriminator,

        lambda_cycle=10,

    ):

        super(CycleGan, self).__init__()

        self.m_gen        = monet_generator

        self.p_gen        = photo_generator

        self.m_disc       = monet_discriminator

        self.p_disc       = photo_discriminator

        self.lambda_cycle = lambda_cycle

        

    def compile(

        self,

        m_gen_optimizer,

        p_gen_optimizer,

        m_disc_optimizer,

        p_disc_optimizer,

        gen_loss_fn,

        disc_loss_fn,

        cycle_loss_fn,

        identity_loss_fn

    ):

        super(CycleGan, self).compile()

        self.m_gen_optimizer  = m_gen_optimizer

        self.p_gen_optimizer  = p_gen_optimizer

        self.m_disc_optimizer = m_disc_optimizer

        self.p_disc_optimizer = p_disc_optimizer

        self.gen_loss_fn      = gen_loss_fn

        self.disc_loss_fn     = disc_loss_fn

        self.cycle_loss_fn    = cycle_loss_fn

        self.identity_loss_fn = identity_loss_fn

        

    # Defining the training procedure

        

    def train_step(self, batch_data):

        real_monet, real_photo = batch_data

        

        with tf.GradientTape(persistent=True) as tape:

            # Calculate losses for Monet Processing 

            fake_monet        = self.m_gen(real_photo, training=True)  

            disc_fake_monet   = self.m_disc(fake_monet, training=True) 

            disc_real_monet   = self.m_disc(real_monet, training=True)

            same_monet        = self.m_gen(real_monet, training=True)

            monet_gen_loss    = self.gen_loss_fn(disc_fake_monet)

            monet_disc_loss   = self.disc_loss_fn(disc_real_monet, disc_fake_monet)

            monet_identity_loss = self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)



            # Calculate losses for Photo Processing

            fake_photo        = self.p_gen(real_monet, training=True)

            disc_fake_photo   = self.p_disc(fake_photo, training=True)

            disc_real_photo   = self.p_disc(real_photo, training=True)

            same_photo        = self.p_gen(real_photo, training=True)

            photo_gen_loss    = self.gen_loss_fn(disc_fake_photo)

            photo_disc_loss   = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

            photo_identity_loss = self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            

            # Calculate total cycled losses

            cycled_photo      = self.p_gen(fake_monet, training=True) 

            cycled_monet      = self.m_gen(fake_photo, training=True)

            photo_cycled_loss = self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle) 

            monet_cycled_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle)

            total_cycled_loss  = photo_cycled_loss + monet_cycled_loss



            # evaluates total generator loss

            total_monet_gen_loss = monet_gen_loss + monet_identity_loss + total_cycled_loss

            total_photo_gen_loss = photo_gen_loss + photo_identity_loss + total_cycled_loss

 

            

        # Calculate the gradients for generator and discriminator

        monet_generator_gradients     = tape.gradient(total_monet_gen_loss, self.m_gen.trainable_variables)

        photo_generator_gradients     = tape.gradient(total_photo_gen_loss, self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss, self.m_disc.trainable_variables)

        photo_discriminator_gradients = tape.gradient(photo_disc_loss, self.p_disc.trainable_variables)



        # Apply the gradients to the optimizer

        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients, self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients, self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients, self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients, self.p_disc.trainable_variables))

        

        return {

            "monet_gen_loss": total_monet_gen_loss,

            "photo_gen_loss": total_photo_gen_loss,

            "monet_disc_loss": monet_disc_loss,

            "photo_disc_loss": photo_disc_loss

        }
# The Adam optimizer with tuned hyperparameters is used for training. 

# The learning rate is using 0.0002.

# The momentum term β1 is specified to 0.5 helped stabilize training.

with strategy.scope():

    monet_generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    photo_generator_optimizer     = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)



    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
with strategy.scope():

    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True,

                                                  reduction=tf.keras.losses.Reduction.NONE)



    # The discriminator loss function:

    # It compares real images to a matrix of 1s and fake images to a matrix of 0s. 

    # The perfect discriminator will output all 1s for real images and all 0s for fake images. 

    # The discriminator loss outputs the average of the real and generated loss.

    def discriminator_loss(real, generated):

        real_loss       = loss_obj(tf.ones_like(real), real)

        generated_loss  = loss_obj(tf.zeros_like(generated), generated)

        avg_disc_loss   = (real_loss + generated_loss) * 0.5

        return avg_disc_loss



    # The generator loss function:

    # The generator wants to fool the discriminator into thinking the generated image is real.

    # The perfect generator will have the discriminator output only 1s. 

    # Thus, it compares the generated image to a matrix of 1s to find the loss.

    def generator_loss(generated):

        return loss_obj(tf.ones_like(generated), generated)

 

    # The cycle consistency loss is calculatd by finding the average of their difference.

    def calc_cycle_loss(real_image, cycled_image, LAMBDA):

        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1



    # The identity loss compares the image with its generator (i.e. photo with photo generator). 

    # If given a photo as input, we want it to generate the same image as the image was originally a photo. 

    # The identity loss compares the input with the output of the generator.

    def identity_loss(real_image, same_image, LAMBDA):

        loss = tf.reduce_mean(tf.abs(real_image - same_image))

        return LAMBDA * 0.5 * loss
with strategy.scope():

    

    # Create a cycle_gan_model

    cycle_gan_model = CycleGan(monet_generator, 

                               photo_generator, 

                               monet_discriminator, 

                               photo_discriminator

    )



    # Configure the cycle_gan_model with optimizers and loss function 

    cycle_gan_model.compile(

        m_gen_optimizer  = monet_generator_optimizer,

        p_gen_optimizer  = photo_generator_optimizer,

        m_disc_optimizer = monet_discriminator_optimizer,

        p_disc_optimizer = photo_discriminator_optimizer,

        gen_loss_fn      = generator_loss,

        disc_loss_fn     = discriminator_loss,

        cycle_loss_fn    = calc_cycle_loss,

        identity_loss_fn = identity_loss

    )

# Train the model

EPOCHS=25

cycle_gan_model.fit(tf.data.Dataset.zip((monet_ds, photo_ds)), epochs=EPOCHS)
_, ax = plt.subplots(5, 2, figsize=(25, 25))

for i, img in enumerate(photo_ds.take(5)):

    prediction = monet_generator(img, training=False)[0].numpy()

    

    # Convert the data from the range [-1,1] to [0, 255] by multiplying 127.5 and plus 127.5.

    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

    img        = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)



    ax[i, 0].imshow(img)

    ax[i, 1].imshow(prediction)

    ax[i, 0].set_title("Input Photo")

    ax[i, 1].set_title("Monet-esque")

    ax[i, 0].axis("off")

    ax[i, 1].axis("off")

plt.show()
import shutil

import PIL

! mkdir ../images



i = 1

for img in photo_ds:

    prediction = monet_generator(img, training=False)[0].numpy()

    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

    im = PIL.Image.fromarray(prediction)

    im.save("../images/" + str(i) + ".jpg")

    i += 1



shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")