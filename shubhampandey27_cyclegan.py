import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Sequential, Model
from IPython.display import clear_output
from tqdm import tqdm
import time
AUTOTUNE = tf.data.experimental.AUTOTUNE
# HYPER PARAMETERS

WIDTH, HEIGHT = 256, 256
EPOCHS = 20
BUFFER_SIZE = 400
BATCH_SIZE = 1
OUTPUT_CHANNEL = 3

# LOSSES
LAYER_COUNT = 0
LAMBDA = 100
PATH_X = "/kaggle/input/gan-getting-started/photo_jpg"
PATH_Y = "/kaggle/input/gan-getting-started/monet_jpg"

def crop(img):
    crop_img = tf.image.random_crop(img, size=[HEIGHT, WIDTH, 3])
    return crop_img


def train_preprocess(img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)    
    img = tf.image.resize(img, [HEIGHT+20, WIDTH+20], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = crop(img)
    
    # normalising
    img = tf.cast(img, tf.float32)
    img = img/127.5 -1
    
    # random flipping    
    img = tf.image.random_flip_left_right(img)

    return img

train_x = tf.data.Dataset.list_files(PATH_X + '/*.jpg').map(train_preprocess).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_y = tf.data.Dataset.list_files(PATH_Y + '/*.jpg').map(train_preprocess).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
sample_x = next(iter(train_x))[0]
sample_x = tf.reshape(sample_x, [-1, 256, 256, 3])
sample_y = next(iter(train_y))[0]
sample_y = tf.reshape(sample_y, [-1, 256, 256, 3])
plt.subplot(121)
plt.imshow((sample_x[0]+1.)/2.)
plt.subplot(122)
plt.imshow((sample_y[0]+1.)/2.)
def add_layer(filters, kernel_size,batchnorm = True):
    init = tf.random_normal_initializer(0., 0.02)
    blocks = Sequential()
    blocks.add(layers.Conv2D(filters, kernel_size=kernel_size, strides = 4, kernel_initializer=init,padding='same', use_bias=False))
    if batchnorm:
        blocks.add(layers.BatchNormalization())
    blocks.add(layers.LeakyReLU())
    return blocks

def add_trans_layer(filters, kernel_size, dropout=True):
    init = tf.random_normal_initializer(0., 0.02)
    blocks = Sequential()
    blocks.add(layers.Conv2DTranspose(filters, kernel_size, strides=4, kernel_initializer=init, use_bias=False, padding='same'))
    blocks.add(layers.BatchNormalization())
    if dropout:
        blocks.add(layers.Dropout(0.4))
    blocks.add(layers.LeakyReLU())
    return blocks

def make_gen():
    inputs = layers.Input(shape = [256, 256, 3])

    down = [
            add_layer(64, 5, False),
            add_layer(128, 5),
            add_layer(256, 5),
            add_layer(512, 5)
    ]

    up = [
            add_trans_layer(256, 5),
            add_trans_layer(128, 5),
            add_trans_layer(64, 5, dropout=False)
    ]

    init = tf.random_normal_initializer(0., 0.02)

    output = layers.Conv2DTranspose(OUTPUT_CHANNEL, kernel_size=5, strides=(4,4), padding='same', kernel_initializer=init, activation='tanh')

    x = inputs

    stack = []

    for d in down:
        x = d(x)
        stack.append(x)
    
    stack.pop(-1)

    for a in up:
        x = a(x)
        x = layers.Concatenate()([x, stack[-1]])
        stack.pop(-1)

    x = output(x)

    return Model(inputs = inputs, outputs=x)

Gx = make_gen()
Gy = make_gen()
tf.keras.utils.plot_model(Gx, show_shapes=True, dpi=64)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def gen_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

Gx_opt = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
Gy_opt = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
def make_disc():
    init = tf.random_normal_initializer(0., 0.05)

    input_shape = [256, 256, 3]

    down = [
            add_layer(64, (5,5), batchnorm=False),
            add_layer(256, (5,5)),
            layers.ZeroPadding2D(),
            layers.Conv2D(512, kernel_size=(5,5), kernel_initializer=init, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.ZeroPadding2D(),
            layers.Conv2D(1, kernel_size=(5,5), kernel_initializer=init),
            layers.Flatten(),
            layers.Dense(1, activation='tanh')
    ]

    inp = layers.Input(shape=input_shape)

    x = inp
    for layer in down:
        x = layer(x)
  
    return Model(inputs=inp, outputs = x)
Dx = make_disc()
Dy = make_disc()

tf.keras.utils.plot_model(Dx, show_shapes=True, dpi=64)
def disc_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return fake_loss + real_loss

Dx_opt = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
Dy_opt = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
def cyclic_loss(real_img, cycled_img):
    return LAMBDA*tf.reduce_mean(abs(real_img-cycled_img))

def identity_loss(real_img, same_img):
    return 0.5*LAMBDA*tf.reduce_mean(abs(real_img-same_img))
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = Gx(real_x, training=True)
        cycled_x = Gy(fake_y, training=True)
        fake_x = Gy(real_y, training=True)
        cycled_y = Gx(fake_x, training=True)
        same_x = Gx(real_x, training=True)
        same_y = Gy(real_y, training=True)

        Dx_real = Dx(real_x, training=True)
        Dx_fake = Dx(fake_x, training=True)

        Dy_real = Dy(real_y, training=True)
        Dy_fake = Dy(fake_y, training=True)

        Dx_loss = disc_loss(Dx_real, Dx_fake)
        Dy_loss = disc_loss(Dy_real, Dy_fake)
        
        cycled_loss = cyclic_loss(real_x, cycled_x) + cyclic_loss(real_y, cycled_y)
        
        Gx_loss = gen_loss(Dy_fake) + identity_loss(real_x, same_x) + cycled_loss 
        Gy_loss = gen_loss(Dx_fake) + identity_loss(real_y, same_y) + cycled_loss


    Gx_grad = tape.gradient(Gx_loss,Gx.trainable_variables)
    Gx_opt.apply_gradients(zip(Gx_grad, Gx.trainable_variables))

    Gy_grad = tape.gradient(Gy_loss, Gy.trainable_variables)
    Gy_opt.apply_gradients(zip(Gy_grad, Gy.trainable_variables))

    Dx_grad = tape.gradient(Dx_loss, Dx.trainable_variables)
    Dx_opt.apply_gradients(zip(Dx_grad, Dx.trainable_variables))

    Dy_grad = tape.gradient(Dy_loss, Dy.trainable_variables)
    Dy_opt.apply_gradients(zip(Dy_grad, Dy.trainable_variables))

def train(train_x, train_y, EPOCHS):
    init_time = time.time()
    for epoch in range(EPOCHS):
        start = time.time()
        for x_batch, y_batch in zip(train_x, train_y):
            train_step(x_batch, y_batch)
        clear_output(wait=True)
        print('Training...')
        print(f"|{(epoch+1)*'#' + (EPOCHS-epoch-1)*'-'}|")
        print(f'Time taken on Epoch {epoch+1} : {time.time()-start}')
        print(f'Total time ellapsed so far : {time.time()-init_time}')
    clear_output(wait=True)
    print("Model Trained.")
    print(f'Total time ellapsed : {time.time()-init_time}')
train(train_x, train_y, EPOCHS)
plt.figure()
plt.subplot(121)
plt.imshow(sample_x[0])
plt.subplot(122)
plt.imshow(Gx(sample_x)[0])