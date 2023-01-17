import numpy as np

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from utils import tile_raster_images

import matplotlib.pyplot as plt

import math



%matplotlib inline

plt.rcParams['image.cmap'] = 'jet'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

n_samples = mnist.train.num_examples



# training parameters

batch_size = 100

lr = 0.0002

n_epochs = 20



def lrelu(x, th=0.2):

    return tf.maximum(th * x, x)
# D(x)

def discriminator(x, isTrain=True, reuse=False):

    with tf.variable_scope('discriminator', reuse=reuse):

        # 1st hidden layer

        conv = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')

        bn = tf.layers.batch_normalization(conv)

        lrelu_ = lrelu(bn, 0.2)

        

        

        # 2nd hidden layer

        conv1 = tf.layers.conv2d(lrelu_, 128, [4, 4], strides=(2, 2), padding='same')

        bn1 = tf.layers.batch_normalization(conv1)

        lrelu_1 = lrelu(bn1, 0.2)

        

        # output layer

        flat = tf.reshape(lrelu_1, [-1, 8*8*128])

        logits = tf.layers.dense(inputs=flat, units=1)

        out = tf.nn.sigmoid(logits)

        

        return out, logits



# G(z)

def generator(z, isTrain=True):

    with tf.variable_scope('generator'):

        # 1st hidden layer

        conv = tf.layers.conv2d_transpose(z, 512, [4, 4], strides=(1, 1), padding='valid')

        bn = tf.layers.batch_normalization(conv)

        lrelu_ = lrelu(tf.layers.batch_normalization(bn, training=isTrain))

        

        # 2nd hidden layer

        

        conv1 = tf.layers.conv2d_transpose(lrelu_, 256, [4, 4], strides=(2, 2), padding='same')

        bn1 = tf.layers.batch_normalization(conv1)

        lrelu_1 = lrelu(tf.layers.batch_normalization(bn1, training=isTrain))



        # 3rd hidden layer

        conv2 = tf.layers.conv2d_transpose(lrelu_1, 128, [4, 4], strides=(2, 2), padding='same')

        bn2 = tf.layers.batch_normalization(conv2)

        lrelu_2 = lrelu(tf.layers.batch_normalization(bn2, training=isTrain))

        # output layer

        

        conv3 = tf.layers.conv2d_transpose(lrelu_2, 1, [4, 4], strides=(2, 2), padding='same')

        out = tf.nn.sigmoid(conv3)

        

        return out



def show_generated(G, N, shape=(32,32), stat_shape=(10,10), interpolation="bilinear"):

    """Visualization of generated samples

     G - generated samples

     N - number of samples

     shape - dimensions of samples eg (32,32)

     stat_shape - dimension for 2D sample display (eg for 100 samples (10,10)

    """

    

    image = (tile_raster_images(

        X=G,

        img_shape=shape,

        tile_shape=(int(math.ceil(N/stat_shape[0])), stat_shape[0]),

        tile_spacing=(1, 1)))

    plt.figure(figsize=(10, 14))

    plt.imshow(image, interpolation=interpolation)

    plt.axis('off')

    plt.show()

    



def gen_z(N, batch_size):

    z = np.random.normal(0, 1, (batch_size, 1, 1, N))

    return z



cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#generator = make_generator_model()

tf.reset_default_graph() 

# input variables

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))

z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))

isTrain = tf.placeholder(dtype=tf.bool)

    

# generator

G_z = generator(z, isTrain)

d_x, plop = discriminator(x, isTrain = False)



noise = np.random.normal(1,size = [1, 1,1,100])





config = tf.ConfigProto()

config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)

tf.global_variables_initializer().run()



generated_image =        sess.run(G_z, {z:noise, isTrain:False})         #generator(noise, training=False)



plt.imshow(generated_image[0, :, :, 0], cmap='gray')



outp = sess.run(d_x, {x:noise2} )

print(outp)
noise2 = np.random.normal(1, size = [1,32,32,1])

outp = sess.run([d_x], {x:noise2} )
tf.reset_default_graph()

sess.close()
# input variables

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))

z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))

isTrain = tf.placeholder(dtype=tf.bool)

    

# generator

G_z = generator(z, isTrain)

    

# discriminator

# real

D_real, D_real_logits = discriminator(x, isTrain)

# fake

D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)





# labels for learning



#true_labels = tf.ones([batch_size, 1, 1, 1]

#fake_labels = tf.zeros([batch_size, 1, 1, 1]



# loss for each network                       

D_loss_real = cross_entropy(tf.ones_like(D_real_logits), D_real_logits)

D_loss_fake = cross_entropy(tf.zeros_like(D_fake_logits), D_fake_logits)

D_loss = D_loss_real + D_loss_fake

G_loss = cross_entropy(tf.ones_like(D_fake_logits), D_fake_logits)



# trainable variables for each network

T_vars = tf.trainable_variables()

D_vars = [var for var in T_vars if var.name.startswith('discriminator')]

G_vars = [var for var in T_vars if var.name.startswith('generator')]



# optimizer for each network

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

    D_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(D_loss, var_list=D_vars)

    G_optim = tf.train.AdamOptimizer(lr, beta1=0.3).minimize(G_loss, var_list=G_vars)





# open session and initialize all variables

config = tf.ConfigProto()

config.gpu_options.allow_growth=True

sess = tf.InteractiveSession(config=config)

tf.global_variables_initializer().run()



# MNIST resize and normalization

train_set = tf.image.resize_images(mnist.train.images, [32, 32]).eval()

# input normalization

#... vec je normaliziran ?



#fixed_z_ = np.random.uniform(-1, 1, (100, 1, 1, 100))

fixed_z_ = gen_z(100, 100)

total_batch = int(n_samples / batch_size)

                       





for epoch in range(n_epochs):

    for iter in range(total_batch):

        # update discriminator

        x_ = train_set[iter*batch_size:(iter+1)*batch_size]

        

        # update discriminator

        

        z_ = gen_z(100, batch_size)

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True, })

                



        # update generator

        loss_g_, _ = sess.run([G_loss, G_optim], {x: x_, z: z_, isTrain: True})

            

    print('[%d/%d] loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), n_epochs, loss_d_, loss_g_))

    

    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    print(test_images.shape)

    show_generated(test_images, 100)