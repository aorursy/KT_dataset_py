import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)


# !/opt/bin/nvidia-smi
def load_data():
    """thank for Amy Jang for this piece of code : https://www.kaggle.com/amyjang/monet-cyclegan-tutorial#Load-in-the-data 
    """
    from kaggle_datasets import KaggleDatasets
    GCS_PATH = KaggleDatasets().get_gcs_path()
    MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))

    PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))


    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMAGE_SIZE = [256, 256]

    def decode_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
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


    monet_ds = load_dataset(MONET_FILENAMES, labeled=True)
    photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True)
    return monet_ds, photo_ds
# Here padding='same' would eliminate the affect of kernel on the output size, 
# by changing strides from 1 to 2, we can see the size halved: 4=>2.
x = tf.random.normal((2, 4, 4, 1))
layers.Conv2D(64, (3, 3),  padding='same', strides=(1,1))(x).shape[1:3]
x = tf.random.normal((2, 256, 256, 1))
layers.Conv2D(64, (3, 3),  padding='same', strides=(4,4))(x).shape[1:3]
class Generator(keras.Model):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        # z: [b, 100] => [b, 64, 64, 256] => [b, 256, 256, 3]
        self.fc = layers.Dense(64*64*256, use_bias=False)
        #layers.BatchNormalization()
        

        self.conv1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        #assert model.output_shape == (None, 7, 7, 128)
        self.bn1 = layers.BatchNormalization()


        self.conv2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        #assert model.output_shape == (None, 14, 14, 64)
        self.bn2 = layers.BatchNormalization()


        self.conv3 = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        #assert model.output_shape == (None, 28, 28, 1)


        
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = tf.reshape(x, (-1, 64, 64, 256))
        x = tf.nn.leaky_relu(x)
        
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.tanh(self.conv3(x))
        
        return x
        
        
class Discriminator(keras.Model):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # input: [b, 256, 256, 3] => output: [b, 1]
        
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(3, 3), padding='valid')
        #layers.Dropout(0.3)
        
        self.conv2 = layers.Conv2D(128, (5, 5), strides= (3, 3), padding='valid')
        self.bn2 = layers.BatchNormalization()
        #layers.Dropout(0.3)
        
        self.conv3 = layers.Conv2D(256, (5, 5), strides= (3, 3), padding='valid')
        self.bn3 = layers.BatchNormalization()
        
        # [b, h, w, 3] => [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)


        
    def call(self, inputs, training=None):
        
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        
        x = self.flatten(x)
        logits = self.fc(x)
        
        return logits

with strategy.scope():
    def celoss_ones(logits):
        # logits shape: [b, 1]
        # labels shape: [b] = [1] * num_pics
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.ones_like(logits))

        return tf.reduce_mean(loss)
with strategy.scope():
    def celoss_zeros(logits):
        # logits shape: [b, 1]
        # labels shape: [b] = [0] * num_pics
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                       labels=tf.zeros_like(logits))

        return tf.reduce_mean(loss)


    

class DeepConvGan(keras.Model):
    
    def __init__(self, generator, discriminator):
        
        super(DeepConvGan, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        
        
    def compile(self, gen_optimizer, 
                disc_optimizer,
                celoss_ones, 
                celoss_zeros):
        
        super(DeepConvGan, self).compile()
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.celoss_ones = celoss_ones
        self.celoss_zeros = celoss_zeros
        
        
    def train_step(self, batch_data):
        
        batch_x, batch_z = batch_data
        
        # train disc
        with tf.GradientTape() as tape:
            # random noise to fake monet 
            fake_image = self.generator(batch_z, training=True)
            
            # disc the fake and real image
            d_fake_logits = self.discriminator(fake_image, training=True)
            d_real_logits = self.discriminator(batch_x, training=True)
            
            # get discriminator loss
            d_loss_real = self.celoss_ones(d_real_logits)
            d_loss_fake = self.celoss_zeros(d_fake_logits)
            monet_disc_loss = d_loss_fake + d_loss_real
            
        grads = tape.gradient(monet_disc_loss, self.discriminator.trainable_variables)
        
        self.disc_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        
        # Train generator
        with tf.GradientTape() as tape:
            # get generator loss
            fake_image = self.generator(batch_z, training=True)
            d_fake_logits = self.discriminator(fake_image, training=True)
            photo_gen_loss = self.celoss_ones(d_fake_logits)
             
        grads = tape.gradient(photo_gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        return {
            "photo_gen_loss": photo_gen_loss,
            "monet_disc_loss": monet_disc_loss
        }

        
# Main Code cell 
tf.random.set_seed(22)
np.random.seed(22)

# set up hyperparameters
z_dim = 100
epochs = 300
batch_size = 128
learning_rate = 2e-4

with strategy.scope():
    gen_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    disc_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

# load data and train
monet_ds, photo_ds = load_data()
monet_ds = monet_ds.batch(batch_size).repeat()
photo_ds = photo_ds.batch(batch_size).repeat()
monet_iter = iter(monet_ds)
photo_iter = iter(photo_ds)

batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
gen_ds = tf.data.Dataset.from_tensor_slices(batch_z).batch(batch_size).repeat()
    
with strategy.scope():
    
    
    # initialize instances
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 256, 256, 3))
    generator = Generator()
    generator.build(input_shape=(None, z_dim))

    
    dc_gan = DeepConvGan(generator, discriminator)
    
    dc_gan.compile(gen_optimizer=gen_optimizer, 
                disc_optimizer=disc_optimizer,
                celoss_ones=celoss_ones, 
                celoss_zeros=celoss_zeros)
    
    
#                 gen_loss_fn=gen_loss_fn, 
#                 disc_loss_fn=gen_loss_fn
    
dc_gan.fit(tf.data.Dataset.zip((monet_ds, gen_ds)),epochs=300, steps_per_epoch=600 // batch_size)


img = dc_gan.generator(tf.random.uniform([1, z_dim], minval=-1., maxval=1.), training=False).numpy()[0,:,:,:]
plt.imshow(img*0.5+0.5)
plt.imshow(next(monet_iter).numpy()[0]*0.5+0.5)