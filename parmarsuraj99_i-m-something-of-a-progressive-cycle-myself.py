!pip install -q git+https://github.com/tensorflow/examples.git
import os

from tqdm.auto import tqdm



import numpy as np

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow.keras import layers as L

from tensorflow_examples.models.pix2pix import pix2pix



from kaggle_datasets import KaggleDatasets
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

    

print(tf.__version__)
GCS_PATH = KaggleDatasets().get_gcs_path()



MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))



BUFFER_SIZE = 1000

BATCH_SIZE = 1

IMG_WIDTH = 128

IMG_HEIGHT = 128



IMAGE_SIZE = [IMG_WIDTH, IMG_HEIGHT]



def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = (tf.cast(image, tf.float32) / 127.5) - 1

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image



def read_tfrecord(example):

    tfrecord_format = {

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    image = decode_image(example['image'])

    return image



def load_dataset(filenames, labeled=True, ordered=False):

    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset



monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(BATCH_SIZE)

photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(BATCH_SIZE)



example_monet = next(iter(monet_ds))

example_photo = next(iter(photo_ds))



plt.subplot(121)

plt.title('Photo')

plt.imshow(example_photo[0] * 0.5 + 0.5)



plt.subplot(122)

plt.title('Monet')

plt.imshow(example_monet[0] * 0.5 + 0.5)


OUTPUT_CHANNELS = 3



generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')



discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)

discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


class CycleGan(tf.keras.Model):

    def __init__(

        self,

        generator_G,

        generator_F,

        discriminator_X,

        discriminator_Y,

        lambda_cycle=10.0,

        lambda_identity=0.5,

    ):

        super(CycleGan, self).__init__()

        self.gen_G = generator_G

        self.gen_F = generator_F

        self.disc_X = discriminator_X

        self.disc_Y = discriminator_Y

        self.lambda_cycle = lambda_cycle

        self.lambda_identity = lambda_identity



    def compile(

        self,

        gen_G_optimizer,

        gen_F_optimizer,

        disc_X_optimizer,

        disc_Y_optimizer,

        gen_loss_fn,

        disc_loss_fn,

        cycle_loss_fn=tf.keras.losses.MeanAbsoluteError(),

        identity_loss_fn=tf.keras.losses.MeanAbsoluteError()

        

    ):

        super(CycleGan, self).compile()

        self.gen_G_optimizer = gen_G_optimizer

        self.gen_F_optimizer = gen_F_optimizer

        self.disc_X_optimizer = disc_X_optimizer

        self.disc_Y_optimizer = disc_Y_optimizer

        self.generator_loss_fn = gen_loss_fn

        self.discriminator_loss_fn = disc_loss_fn

        self.cycle_loss_fn = cycle_loss_fn

        self.identity_loss_fn = identity_loss_fn



    def train_step(self, batch_data):

        # x is Horse and y is zebra

        real_x, real_y = batch_data



        # For CycleGAN, we need to calculate different

        # kinds of losses for the generators and discriminators.

        # We will perform the following steps here:

        #

        # 1. Pass real images through the generators and get the generated images

        # 2. Pass the generated images back to the generators to check if we

        #    we can predict the original image from the generated image.

        # 3. Do an identity mapping of the real images using the generators.

        # 4. Pass the generated images in 1) to the corresponding discriminators.

        # 5. Calculate the generators total loss (adverserial + cycle + identity)

        # 6. Calculate the discriminators loss

        # 7. Update the weights of the generators

        # 8. Update the weights of the discriminators

        # 9. Return the losses in a dictionary



        with tf.GradientTape(persistent=True) as tape:

            # Horse to fake zebra

            fake_y = self.gen_G(real_x, training=True)

            # Zebra to fake horse -> y2x

            fake_x = self.gen_F(real_y, training=True)



            # Cycle (Horse to fake zebra to fake horse): x -> y -> x

            cycled_x = self.gen_F(fake_y, training=True)

            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y

            cycled_y = self.gen_G(fake_x, training=True)



            # Identity mapping

            same_x = self.gen_F(real_x, training=True)

            same_y = self.gen_G(real_y, training=True)



            # Discriminator output

            disc_real_x = self.disc_X(real_x, training=True)

            disc_fake_x = self.disc_X(fake_x, training=True)



            disc_real_y = self.disc_Y(real_y, training=True)

            disc_fake_y = self.disc_Y(fake_y, training=True)



            # Generator adverserial loss

            gen_G_loss = self.generator_loss_fn(disc_fake_y)

            gen_F_loss = self.generator_loss_fn(disc_fake_x)



            # Generator cycle loss

            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y, self.lambda_cycle) * self.lambda_cycle

            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x, self.lambda_cycle) * self.lambda_cycle



            # Generator identity loss

            id_loss_G = (

                self.identity_loss_fn(real_y, same_y, self.lambda_cycle)

                * self.lambda_cycle

                * self.lambda_identity

            )

            id_loss_F = (

                self.identity_loss_fn(real_x, same_x, self.lambda_cycle)

                * self.lambda_cycle

                * self.lambda_identity

            )



            # Total generator loss

            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G

            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F



            # Discriminator loss

            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)

            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)



        # Get the gradients for the generators

        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)

        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)



        # Get the gradients for the discriminators

        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)

        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)



        # Update the weights of the generators

        self.gen_G_optimizer.apply_gradients(

            zip(grads_G, self.gen_G.trainable_variables)

        )

        self.gen_F_optimizer.apply_gradients(

            zip(grads_F, self.gen_F.trainable_variables)

        )



        # Update the weights of the discriminators

        self.disc_X_optimizer.apply_gradients(

            zip(disc_X_grads, self.disc_X.trainable_variables)

        )

        self.disc_Y_optimizer.apply_gradients(

            zip(disc_Y_grads, self.disc_Y.trainable_variables)

        )



        return {

            "G_loss": total_loss_G,

            "F_loss": total_loss_F,

            "D_X_loss": disc_X_loss,

            "D_Y_loss": disc_Y_loss,

        }



class GANMonitor(tf.keras.callbacks.Callback):

    """A callback to generate and save images after each epoch"""



    def __init__(self, num_img=4):

        self.num_img = num_img



    def on_epoch_end(self, epoch, logs=None):

        _, ax = plt.subplots(4, 2, figsize=(12, 12))

        for i, img in enumerate(photo_ds.take(4)):

            prediction = self.model.gen_G(img)[0].numpy()

            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)



            ax[i, 0].imshow(img)

            ax[i, 1].imshow(prediction)

            ax[i, 0].set_title("Input image")

            ax[i, 1].set_title("Translated image")

            ax[i, 0].axis("off")

            ax[i, 1].axis("off")



            prediction = tf.keras.preprocessing.image.array_to_img(prediction)

            prediction.save(

                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)

            )

        plt.show()

        plt.close()

def discriminator_loss(real, generated):

        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)



        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)



        total_disc_loss = real_loss + generated_loss



        return total_disc_loss * 0.5

    

def generator_loss(generated):

        return tf.keras.losses.BinaryCrossentropy(

            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)

    

def calc_cycle_loss(real_image, cycled_image, LAMBDA):

        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))



        return LAMBDA * loss1

    

def identity_loss(real_image, same_image, LAMBDA):

        loss = tf.reduce_mean(tf.abs(real_image - same_image))

        return LAMBDA * 0.5 * loss
# Loss function for evaluating adversarial loss

adv_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)



# Define the loss function for the generators

def generator_loss_fn(fake):

    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)

    return fake_loss





# Define the loss function for the discriminators

def discriminator_loss_fn(real, fake):

    real_loss = adv_loss_fn(tf.ones_like(real), real)

    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)

    return (real_loss + fake_loss) * 0.5





# Create cycle gan model

cycle_gan_model = CycleGan(

    generator_G=generator_g, generator_F=generator_f, 

    discriminator_X=discriminator_x, discriminator_Y=discriminator_y

)



# Compile the model

cycle_gan_model.compile(

    gen_G_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

    gen_F_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

    disc_X_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

    disc_Y_optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),

    gen_loss_fn=generator_loss,

    disc_loss_fn=discriminator_loss,

    cycle_loss_fn = calc_cycle_loss,

    identity_loss_fn = identity_loss

)

# Callbacks

plotter = GANMonitor()
cycle_gan_model.fit(

    tf.data.Dataset.zip((photo_ds, monet_ds)),

    epochs=15,

    callbacks=[plotter],

)
GCS_PATH = KaggleDatasets().get_gcs_path()



MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))



BUFFER_SIZE = 1000

BATCH_SIZE = 1

IMG_WIDTH = 256

IMG_HEIGHT = 256



IMAGE_SIZE = [IMG_WIDTH, IMG_HEIGHT]



def decode_image(image):

    image = tf.image.decode_jpeg(image, channels=3)

    image = (tf.cast(image, tf.float32) / 127.5) - 1

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image



def read_tfrecord(example):

    tfrecord_format = {

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "image": tf.io.FixedLenFeature([], tf.string),

        "target": tf.io.FixedLenFeature([], tf.string)

    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    image = decode_image(example['image'])

    return image



def load_dataset(filenames, labeled=True, ordered=False):

    dataset = tf.data.TFRecordDataset(filenames)

    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset



monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(BATCH_SIZE)

photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(BATCH_SIZE)



example_monet = next(iter(monet_ds))

example_photo = next(iter(photo_ds))



plt.subplot(121)

plt.title('Photo')

plt.imshow(example_photo[0] * 0.5 + 0.5)



plt.subplot(122)

plt.title('Monet')

plt.imshow(example_monet[0] * 0.5 + 0.5)

gc.collect()
cycle_gan_model.fit(

    tf.data.Dataset.zip((photo_ds, monet_ds)),

    epochs=15,

    callbacks=[plotter],

)
import PIL

! mkdir ../images
i = 1

for img in tqdm(photo_ds, total=len(os.listdir("/kaggle/input/gan-getting-started/photo_jpg/"))):

    prediction = generator_g(img, training=False)[0].numpy()

    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)

    im = PIL.Image.fromarray(prediction)

    im.save("../images/" + str(i) + ".jpg")

    i += 1
import shutil

shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")