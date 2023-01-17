import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
import numpy as np

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
print('Monet TFRecord Files:', len(MONET_FILENAMES))

PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))
print('Photo TFRecord Files:', len(PHOTO_FILENAMES))
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
monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(1)
photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)
#photo_ds = photo_ds.take(30)
#monet_ds = monet_ds.take(30)
example_monet = next(iter(monet_ds))
example_photo = next(iter(photo_ds))
plt.subplot(121)
plt.title('Photo')
plt.imshow(example_photo[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Monet')
plt.imshow(example_monet[0] * 0.5 + 0.5)
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    result.add(layers.LeakyReLU())

    return result
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result
# bs = batch size
down_stack = [
    downsample(64, 4, apply_instancenorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
]

up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
]

def reshape_out():
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh') # (bs, 256, 256, 3)
    return last
def Generator():
    inputs = layers.Input(shape=[256,256,3])

    last = reshape_out()

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
def RescaleTF(aBlock):
    x1 = tf.math.reduce_max(aBlock)
    x2 = tf.math.reduce_min(aBlock)
    s = 2/(x2-x1)
    p = -x1*s - 1
    return aBlock * s + p

def Enc():
    encoder_inputs = layers.Input(shape=[256,256,3])
    x = encoder_inputs
    skips = []
    #skips.append(x)

    # Downsampling through the model
    for i,down in enumerate(down_stack):
        #print(i,"E ", x.shape,"\t",type(x))
        x = down(x)
        #print(i,"skips ", x.shape,"\t", type(x))
        
        # x is zeros for i = 7
        #if i != 7 and i > 1:
        #    x = RescaleTF(x)
        skips.append(x)

    return keras.Model(encoder_inputs, skips, name="encoder")

def Dec1(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[0](inp)
    return keras.Model(inp, x, name="dec1")
                      
def Dec2(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[1](inp)
    return keras.Model(inp, x, name="dec2")
        
def Dec4(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[2](inp)
    return keras.Model(inp, x, name="dec4")
         
def Dec8(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[3](inp)
    return keras.Model(inp, x, name="dec8")
         
def Dec16(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[4](inp)
    return keras.Model(inp, x, name="dec16")
         
def Dec32(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[5](inp)
    return keras.Model(inp, x, name="dec32")
         
def Dec64(shapeIn):
    inp = layers.Input(shape=shapeIn)
    x = up_stack[6](inp)
    return keras.Model(inp, x, name="dec64")

def Dec128(shapeIn):
    inp = layers.Input(shape=shapeIn)
    last = reshape_out()
    x = last(inp)
    return keras.Model(inp, x, name="dec128")
def RescaleNP(aBlock):
    x1 = np.amin(aBlock)
    x2 = np.amax(aBlock)
    #print(x1,x2)
    s = 2/(x2-x1)
    p = -x1*s - 1
    print("sp",s,p)
    return aBlock * s + p

def Decode(pskips): 
    x = pskips[-1]
    skip = pskips[-2]
    #skip = np.zeros((1,2,2,512))
    x = dec1(x)
    x = layers.Concatenate()([x, skip])

    skip = pskips[-3]
    #skip = np.zeros((1,4,4,512))
    x = dec2(x)
    x = layers.Concatenate()([x, skip])
    #x = np.zeros((1,4,4,1024))
    #RescaleNP(x)

    skip = pskips[-4]
    #Uncommenting the following line may remove some artifacts
    #depending on the final network.
    #skip = np.zeros((1,8,8,512))
    x = dec4(x)
    x = layers.Concatenate()([x, skip])

    skip = pskips[-5]
    #skip = np.zeros((1,16,16,512))
    x = dec8(x)
    x = layers.Concatenate()([x, skip])

    skip = pskips[-6]
    #skip = np.zeros((1,32,32,256))
    x = dec16(x)
    x = layers.Concatenate()([x, skip])

    skip = pskips[-7]
    #skip = np.zeros((1,64,64,128))
    x = dec32(x)
    x = layers.Concatenate()([x, skip])

    skip = pskips[-8]
    #skip = np.zeros((1,128,128,64))
    x = dec64(x)
    x = layers.Concatenate()([x, skip])

    x = dec128(x)
    return x
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = layers.Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = layers.LeakyReLU()(norm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)
with strategy.scope():
    monet_generator = Generator() # transforms photos to Monet-esque paintings
    photo_generator = Generator() # transforms Monet paintings to be more like photos

    monet_discriminator = Discriminator() # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = Discriminator() # differentiates real photos and generated photos
    
    monet_enc = Enc()
    #monet_enc.summary()
    #stop
    dec1 = Dec1((1,1,512))
    #dec1.summary()
    dec2 = Dec2((2,2,1024))
    dec4 = Dec4((4,4,1024))
    dec8 = Dec8((8,8,1024))
    dec16 = Dec16((16,16,1024))
    dec32 = Dec32((32,32,512))
    dec64 = Dec64((64,64,256))
    dec128 = Dec128((128,128,128))
plt.subplot(1, 2, 1)
plt.title("Original Photo")
plt.imshow(example_photo[0] * 0.5 + 0.5)

#to_monet = monet_generator(example_photo)
pskips = monet_enc.predict(example_photo)
to_monet = Decode(pskips)

plt.subplot(1, 2, 2)
plt.title("Monet-esque Photo")
plt.imshow(to_monet[0] * 0.5 + 0.5)
plt.show()
class CycleGan(keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        monet_encoder,
        dec1,
        dec2,
        dec4,
        dec8,
        dec16,
        dec32,
        dec64,
        dec128,
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.m_enc = monet_encoder
        self.p_dec1 = dec1
        self.p_dec2 = dec2
        self.p_dec4 = dec4
        self.p_dec8 = dec8
        self.p_dec16 = dec16
        self.p_dec32 = dec32
        self.p_dec64 = dec64
        self.p_dec128 = dec128#'''
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
        photo_dec1_opt,
        photo_dec2_opt,
        photo_dec4_opt,
        photo_dec8_opt,
        photo_dec16_opt,
        photo_dec32_opt,
        photo_dec64_opt,
        photo_dec128_opt,
        use_enc_dec,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.p_dec1_opt = photo_dec1_opt
        self.p_dec2_opt = photo_dec2_opt
        self.p_dec4_opt = photo_dec4_opt
        self.p_dec8_opt = photo_dec8_opt
        self.p_dec16_opt = photo_dec16_opt
        self.p_dec32_opt = photo_dec32_opt
        self.p_dec64_opt = photo_dec64_opt
        self.p_dec128_opt = photo_dec128_opt
        self.identity_loss_fn = identity_loss_fn
        self.use_enc_dec = use_enc_dec
        
        def Dec(skips):                
            #skips[-2] = tf.zeros((1,2,2,512))
            #skips[-3] = tf.zeros((1,4,4,512))            
            #skips[-4] = tf.zeros((1,8,8,512))
            #skips[-5] = tf.zeros((1,16,16,512))
            #skips[-6] = tf.zeros((1,32,32,256))
            #skips[-7] = tf.zeros((1,64,64,128))
            #skips[-8] = tf.zeros((1,128,128,64))

            x = skips[-1]
            #x = np.random.rand(1,1,1,512)
            skip = skips[-2]
            x = self.p_dec1(x, training=True)
            #L2 = self.identity_loss_fn(x, skip, self.lambda_cycle).numpy()
            x = layers.Concatenate()([x, skip])

            skip = skips[-3]
            x = self.p_dec2(x, training=True)
            x = layers.Concatenate()([x, skip])

            skip = skips[-4]
            x = self.p_dec4(x, training=True)
            x = layers.Concatenate()([x, skip])

            skip = skips[-5]
            x = self.p_dec8(x, training=True)
            x = layers.Concatenate()([x, skip])

            skip = skips[-6]
            x = self.p_dec16(x, training=True)
            x = layers.Concatenate()([x, skip])

            skip = skips[-7]
            x = self.p_dec32(x, training=True)
            x = layers.Concatenate()([x, skip])

            skip = skips[-8]
            x = self.p_dec64(x, training=True)
            x = layers.Concatenate()([x, skip])

            return self.p_dec128(x, training=True)
        
        self.skip_dec = Dec
    
    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        
        with tf.GradientTape(persistent=True) as tape:            
            # photo to monet back to photo
            if use_enc_dec:
                skips = self.m_enc(real_photo, training=True)
                fake_monet = self.skip_dec(skips)
            else:
                fake_monet = self.m_gen(real_photo, training=True) 
                
            cycled_photo = self.p_gen(fake_monet, training=True)
           
            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            if use_enc_dec:
                skips = self.m_enc(fake_photo, training=True)
                cycled_monet = self.skip_dec(skips)
            else:
                cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            if use_enc_dec:
                skips = self.m_enc(real_monet, training=True)
                same_monet = self.skip_dec(skips)
            else:
                same_monet = self.m_gen(real_monet, training=True)
                
            same_photo = self.p_gen(real_photo, training=True)
            
            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet, training=True)
            disc_real_photo = self.p_disc(real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet, training=True)
            disc_fake_photo = self.p_disc(fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))
        if use_enc_dec:
            photo_grad_dec1 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec1.trainable_variables)
            photo_grad_dec2 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec2.trainable_variables)
            photo_grad_dec4 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec4.trainable_variables)
            photo_grad_dec8 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec8.trainable_variables)
            photo_grad_dec16 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec16.trainable_variables)
            photo_grad_dec32 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec32.trainable_variables)
            photo_grad_dec64 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec64.trainable_variables)
            photo_grad_dec128 = tape.gradient(total_monet_gen_loss,
                                                      self.p_dec128.trainable_variables)

            self.p_dec1_opt.apply_gradients(zip(photo_grad_dec1,
                                                     self.p_dec1.trainable_variables))
            self.p_dec2_opt.apply_gradients(zip(photo_grad_dec2,
                                                     self.p_dec2.trainable_variables))
            self.p_dec4_opt.apply_gradients(zip(photo_grad_dec4,
                                                     self.p_dec4.trainable_variables))
            self.p_dec8_opt.apply_gradients(zip(photo_grad_dec8,
                                                     self.p_dec8.trainable_variables))
            self.p_dec16_opt.apply_gradients(zip(photo_grad_dec16,
                                                     self.p_dec16.trainable_variables))
            self.p_dec32_opt.apply_gradients(zip(photo_grad_dec32,
                                                     self.p_dec32.trainable_variables))
            self.p_dec64_opt.apply_gradients(zip(photo_grad_dec64,
                                                     self.p_dec64.trainable_variables))
            self.p_dec128_opt.apply_gradients(zip(photo_grad_dec128,
                                                 self.p_dec128.trainable_variables))
        
        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }
with strategy.scope():
    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5
with strategy.scope():
    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)
with strategy.scope():
    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1
with strategy.scope():
    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss
with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
    photo_dec1_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec2_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec4_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec8_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec16_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec32_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec64_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_dec128_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
use_enc_dec = True

with strategy.scope():
    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, monet_discriminator, photo_discriminator,
        monet_enc,
        dec1,
        dec2,
        dec4,
        dec8,
        dec16,
        dec32,
        dec64,
        dec128  
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        photo_dec1_opt = photo_dec1_opt,
        photo_dec2_opt = photo_dec2_opt,
        photo_dec4_opt = photo_dec4_opt,
        photo_dec8_opt = photo_dec8_opt,
        photo_dec16_opt = photo_dec16_opt,
        photo_dec32_opt = photo_dec32_opt,
        photo_dec64_opt = photo_dec64_opt,
        photo_dec128_opt = photo_dec128_opt,
        use_enc_dec = use_enc_dec,
        identity_loss_fn = identity_loss
    )
cycle_gan_model.fit(
    tf.data.Dataset.zip((monet_ds, photo_ds)),
    epochs=25
)
if use_enc_dec:
    print("Min and max skip values")
    
_, ax = plt.subplots(5, 2, figsize=(25, 25))
for i, img in enumerate(photo_ds.take(5)):
    
    if use_enc_dec:
        pskips = monet_enc.predict(img)
        prediction = Decode(pskips)
        if i<=1:
            for j,sk in enumerate(pskips):
                print("[",i,",",j,"]\t", np.amin(sk),":",np.amax(sk))
            print()
    else:
        prediction = monet_generator(img, training=False)
    
    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction[0]*0.5 + 0.5)
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-esque")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.show()
