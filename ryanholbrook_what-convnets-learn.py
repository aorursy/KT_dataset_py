import math

import numpy as np

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt

from matplotlib import gridspec

from itertools import product
class OptVis(object):

    def __init__(self,

                 model, layer, filter, neuron=False,

                 size=[128, 128], fft=True, scale=0.01):

        """ Create a model for use with an optimization visualization of some part of the network. Currently supported are filters and neurons. """

        # Create activation model

        activations = model.get_layer(layer).output

        if len(activations.shape) == 4:

            activations = activations[:,:,:,filter]

        else:

            raise ValueError("Activation shapes other than 4 not implemented.")

        if neuron:

            _, y, x = activations.shape

            # find center

            # TODO: need to compute this from selected size, not activations

            yc = int(round(y/2))

            xc = int(round(x/2))

            activations = activations[:, yc, xc]

        self.activation_model = keras.Model(

            inputs=model.inputs,

            outputs=activations

        )



        # Create random initialization buffer

        self.shape = [1, *size, 3]

        self.fft = fft

        self.image = init_buffer(height=size[0], width=size[1], fft=fft, scale=scale)

        self.fft_scale = fft_scale(size[0], size[1], decay_power=1.0)



    def __call__(self):

        # Preprocessing

        # 



        image = self.activation_model(self.image)

        

        return image



    def compile(self, optimizer):

        self.optimizer = optimizer



    @tf.function

    def train_step(self):

        # Compute loss

        with tf.GradientTape() as tape:

            image = self.image

            if self.fft:

                image = fft_to_rgb(shape=self.shape,

                                   buffer=image,

                                   fft_scale=self.fft_scale)

            image = to_valid_rgb(image)

            image = random_transform(

                tf.squeeze(image),

                jitter=8, 

                scale=1.1,

                rotate=1.0,

                fill_method='reflect')

            image = tf.expand_dims(image, 0)

            loss = clip_gradients(score(self.activation_model(image)))

    

        # Apply gradient

        grads = tape.gradient(loss, self.image)

        self.optimizer.apply_gradients([(-grads, self.image)])

        

        return {'loss': loss}



    @tf.function

    def fit(self, epochs=1, log=False):

        for epoch in tf.range(epochs):

            loss = self.train_step()

            if log: print('Score: {}'.format(loss['loss']))

        

        image = self.image

        if self.fft:

            image = fft_to_rgb(shape=self.shape,

                               buffer=image,

                               fft_scale=self.fft_scale)

        return to_valid_rgb(image)
def score(x):

    s = tf.math.reduce_mean(x)

    return s



@tf.custom_gradient

def clip_gradients(y):

    def backward(dy):

        return tf.clip_by_norm(dy, 1.0)

    return y, backward



# unused

def normalize_gradients(grads, method='l2'):

    if method is 'l2':

        grads = tf.math.l2_normalize(grads)

    elif method is 'std':

        grads /= tf.math.reduce_std(grads) + 1e-8

    elif method is 'clip':

        grads = tf.clip_by_norm(grads, 1.0)

    return grads
# ImageNet statistics

color_correlation_svd_sqrt = np.asarray(

    [[0.26, 0.09, 0.02],

     [0.27, 0.00, -0.05],

     [0.27, -0.09, 0.03]]

).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = np.asarray([0.485, 0.456, 0.406])

color_std = np.asarray([0.229, 0.224, 0.225])



def correlate_color(image):

    image_flat = tf.reshape(image, [-1, 3])

    image_flat = tf.matmul(image_flat, color_correlation_normalized.T)

    image = tf.reshape(image_flat, tf.shape(image))

    return image



def normalize(image):

    return (image - color_mean) / color_std



def to_valid_rgb(image, crop=False):

    if crop:

        image = image[:, 25:-25, 25:-25, :]

    image = correlate_color(image)

    image = tf.nn.sigmoid(image)

    return image
@tf.function

def random_transform(image, jitter=0, rotate=0, scale=1, **kwargs):

    jx = tf.random.uniform([], -jitter, jitter)

    jy = tf.random.uniform([], -jitter, jitter)

    r = tf.random.uniform([], -rotate, rotate)

    s = tf.random.uniform([], 1.0, scale)

    image = apply_affine_transform(

        image,

        theta=r,

        tx=jx, ty=jy,

        zx=s, zy=s,

        **kwargs,

    )

    return image



@tf.function

def apply_affine_transform(x,

                           theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,

                           row_axis=0, col_axis=1, channel_axis=2,

                           fill_method='reflect', cval=0.,

                           interpolation_method='nearest'):

    """ Apply an affine transformation to an image x. """



    theta = tf.convert_to_tensor(theta, dtype=tf.float32)

    tx = tf.convert_to_tensor(tx, dtype=tf.float32)

    ty = tf.convert_to_tensor(ty, dtype=tf.float32)

    shear = tf.convert_to_tensor(shear, dtype=tf.float32)

    zx = tf.convert_to_tensor(zx, dtype=tf.float32)

    zy = tf.convert_to_tensor(zy, dtype=tf.float32)



    transform_matrix = _get_inverse_affine_transform(

        theta,

        tx, ty,

        shear,

        zx, zy,

    )



    x = _apply_inverse_affine_transform(

        x,

        transform_matrix,

        fill_method=fill_method,

        interpolation_method=interpolation_method,

    )



    return x





# adapted from https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/affine_transformations.py

# MIT License: https://github.com/keras-team/keras-preprocessing/blob/master/LICENSE

@tf.function

def _get_inverse_affine_transform(theta, tx, ty, shear, zx, zy):

    """ Construct the inverse of the affine transformation matrix with the given transformations. 

    

    The transformation is taken with respect to the usual right-handed coordinate system."""



    transform_matrix = tf.eye(3, dtype=tf.float32)



    if theta != 0:

        theta = theta * math.pi / 180 # convert degrees to radians

        # this is 

        rotation_matrix = tf.convert_to_tensor(

            [[tf.math.cos(theta), tf.math.sin(theta), 0],

             [-tf.math.sin(theta), tf.math.cos(theta), 0],

             [0, 0, 1]],

            dtype=tf.float32)

        transform_matrix = rotation_matrix



    if tx != 0 or ty != 0:

        shift_matrix = tf.convert_to_tensor(

            [[1, 0, -tx],

             [0, 1, -ty],

             [0, 0, 1]],

            dtype=tf.float32)

        if transform_matrix is None:

            transform_matrix = shift_matrix

        else:

            transform_matrix = tf.matmul(transform_matrix, shift_matrix)



    if shear != 0:

        shear = shear * math.pi / 180 # convert degrees to radians

        shear_matrix = tf.convert_to_tensor(

            [[1, tf.math.sin(shear), 0],

             [0, tf.math.cos(shear), 0],

             [0, 0, 1]],

            dtype=tf.float32)

        if transform_matrix is None:

            transform_matrix = shear_matrix

        else:

            transform_matrix = tf.matmul(transform_matrix, shear_matrix)



    if zx != 1 or zy != 1:

        # need to assert !=0

        zoom_matrix = tf.convert_to_tensor(

            [[1/zx, 0, 0],

             [0, 1/zy, 0],

             [0, 0, 1]],

            dtype=tf.float32)

        if transform_matrix is None:

            transform_matrix = zoom_matrix

        else:

            transform_matrix = tf.matmul(transform_matrix, zoom_matrix)

            

    return transform_matrix



@tf.function

def _apply_inverse_affine_transform(A, Ti, fill_method, interpolation_method):

    """Perform an affine transformation of the image A defined by a

transform whose inverse is Ti. The matrix Ti is assumed to be in

homogeneous coordinate form.



    Available fill methods are "replicate" and "reflect" (default).

    Available interpolation method is "nearest".



    """

    nrows, ncols, _ = A.shape



    # Create centered coordinate grid

    x = tf.range(ncols*nrows) % ncols

    x = tf.cast(x, dtype=tf.float32) - ((ncols-1)/2) # center

    y = tf.range(ncols*nrows) // ncols

    y = tf.cast(y, dtype=tf.float32) - ((nrows-1)/2) # center

    y = -y # left-handed to right-handed coordinates

    z = tf.ones([ncols*nrows], dtype=tf.float32)

    grid = tf.stack([x, y, z])



    # apply transformation

    # x, y, _ = tf.matmul(Ti, grid)

    xy = tf.matmul(Ti, grid)

    x = xy[0, :]

    y = xy[1, :]

    

    # convert coordinates to (approximate) indices

    i = -y + ((nrows-1)/2)

    j = x + ((ncols-1)/2)



    # replicate: 111|1234|444

    if fill_method is 'replicate':

        i = tf.clip_by_value(i, 0.0, nrows-1)

        j = tf.clip_by_value(j, 0.0, ncols-1)

    # reflect: 432|1234|321

    elif fill_method is 'reflect':

        i = _reflect_index(i, nrows-1)

        j = _reflect_index(j, ncols-1)

        

    # nearest neighbor interpolation

    grid = tf.stack([i, j])

    grid = tf.round(grid)

    grid = tf.cast(grid, dtype=tf.int32)

    B = tf.gather_nd(A, tf.transpose(grid))

    B = tf.reshape(B, A.shape)



    return B



@tf.function

def _reflect_index(i, n):

    """Reflect the index i across dimensions [0, n]."""

    i = tf.math.floormod(i-n, 2*n)

    i = tf.math.abs(i - n)

    return tf.math.floor(i)
# Adapted from https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py

# and https://github.com/elichen/Feature-visualization/blob/master/optvis.py

def rfft2d_freqs(h, w):

    """Computes 2D spectrum frequencies."""



    fy = np.fft.fftfreq(h)[:, np.newaxis]

    # when we have an odd input dimension we need to keep one additional

    # frequency and later cut off 1 pixel

    if w % 2 == 1:

        fx = np.fft.fftfreq(w)[: w // 2 + 2]

    else:

        fx = np.fft.fftfreq(w)[: w // 2 + 1]

        

    return np.sqrt(fx * fx + fy * fy)



def fft_scale(h, w, decay_power=1.0):

    freqs = rfft2d_freqs(h, w)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power

    scale *= np.sqrt(w * h)

    return tf.convert_to_tensor(scale, dtype=tf.complex64)



def fft_to_rgb(shape, buffer, fft_scale):

    """Convert FFT spectrum buffer to RGB image buffer."""

    

    batch, h, w, ch = shape



    spectrum = tf.complex(buffer[0], buffer[1]) * fft_scale

    image = tf.signal.irfft2d(spectrum)

    image = tf.transpose(image, (0, 2, 3, 1))

    

    # in case of odd spatial input dimensions we need to crop

    image = image[:batch, :h, :w, :ch]

    image = image / 4.0  # TODO: is that a magic constant?

    

    return image
def init_buffer(height, width=None, batches=1, channels=3, scale=0.01, fft=True):

    """Initialize an image buffer."""

    width = width or height

    shape = [batches, height, width, channels]

    fn = init_fft if fft else init_pixel

    

    buffer = fn(shape, scale)

    

    return tf.Variable(buffer, trainable=True)



def init_pixel(shape, scale=None):

    batches, h, w, ch = shape

#     initializer = tf.initializers.VarianceScaling(scale=scale)

    initializer = tf.random.uniform

    buffer = initializer(shape=[batches, h, w, ch],

                         dtype=tf.float32)

    return buffer





def init_fft(shape, scale=0.1):

    """Initialize FFT image buffer."""

    

    batch, h, w, ch = shape

    freqs = rfft2d_freqs(h, w)

    init_val_size = (2, batch, ch) + freqs.shape



    buffer = np.random.normal(size=init_val_size, scale=scale).astype(np.float32)

    return buffer
def visualize(model, layer, filter, neuron=False, size=[150, 150], fft=True, lr=0.05, epochs=500, log=False):

    optvis = OptVis(model, layer, filter, neuron=neuron, size=size, fft=fft)

    optvis.compile(

        optimizer=tf.optimizers.Adam(lr)

    )

    image = optvis.fit(epochs=epochs, log=log)

    

    plt.imshow(tf.squeeze(image).numpy())

    plt.axis('off')

    

def visualize_layer(model, layer, init_filter=0, neuron=False, size=[150, 150],

                    fft=True, lr=0.05, epochs=500, log=False,

                    rows=2, cols=4, width=16):

    gs = gridspec.GridSpec(rows, cols, wspace=0.01, hspace=0.01)

    plt.figure(figsize=(width, (width * rows) / cols))

    for f, (r, c) in enumerate(product(range(rows), range(cols))):

        optvis = OptVis(model, layer, f+init_filter, neuron=neuron, size=size, fft=fft)

        optvis.compile(optimizer=tf.optimizers.Adam(lr))

        image = optvis.fit(epochs=epochs)

        plt.subplot(gs[r, c])

        plt.imshow(tf.squeeze(image))

        plt.axis('off')
img = init_buffer(120, scale=1.0)

scale = fft_scale(120, 120)

img = fft_to_rgb([1, 120, 120, 3], img, scale)

img = to_valid_rgb(img)

plt.imshow(tf.squeeze(img))

plt.axis('off')

plt.show();
model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=[128, 128, 3])
visualize(model, layer="block2_conv2", filter=0, neuron=True)
visualize_layer(model, layer="block2_conv2", neuron=True, rows=3, cols=4, epochs=250)
visualize_layer(model, layer="block1_conv1", rows=3, cols=4, epochs=100)
visualize_layer(model, layer="block1_conv1", neuron=True, rows=3, cols=4, epochs=100)
visualize_layer(model, layer="block2_conv2", rows=3, cols=4, epochs=250)
visualize_layer(model, layer="block3_conv2", rows=3, cols=4, epochs=250)
visualize_layer(model, layer="block4_conv2", rows=3, cols=4, epochs=250)
visualize_layer(model, layer="block5_conv1", rows=6, cols=4, epochs=250)
visualize_layer(model, layer="block5_conv3", rows=3, cols=4, epochs=250)
resnet50 = tf.keras.applications.ResNet50V2(weights='imagenet', input_shape=[150, 150, 3], include_top=False)
visualize(resnet50, layer="conv4_block1_out", filter=0, neuron=True, epochs=1000)
layer = 'conv3_block1_out'

visualize_layer(resnet50, layer=layer, epochs=500, rows=3, cols=4)
layer = 'conv3_block4_out'

visualize_layer(resnet50, layer=layer, epochs=500, rows=3, cols=4)
layer = 'conv4_block3_out'

neuron = True

size=[224, 224]

visualize_layer(resnet50, layer=layer, epochs=500, rows=3, cols=4, size=size)
layer = 'conv5_block1_out'

size=[224, 224]

visualize_layer(resnet50, layer=layer, epochs=500, rows=3, cols=4, size=size)
layer = 'conv5_block3_out'

size=[224, 224]

visualize_layer(resnet50, layer=layer, epochs=500, rows=6, cols=4, size=size)
layer = 'conv5_block3_out'

size = [1024, 1024]

plt.figure(figsize=(24, 24))

visualize(resnet50, layer=layer, filter=0, epochs=200, size=size)
inceptionv3 = tf.keras.applications.InceptionV3(weights='imagenet')
size = [224, 224]

layer = 'mixed1'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed2'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed3'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed4'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed5'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed6'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed7'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed8'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed9'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'mixed10'

visualize_layer(inceptionv3, layer=layer, epochs=500, rows=6, cols=4, size=size)
xception = tf.keras.applications.Xception(weights='imagenet')
size = [224, 224]

layer = 'block1_conv1'

visualize_layer(xception, layer=layer, epochs=200, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'block1_conv2'

visualize_layer(xception, layer=layer, epochs=200, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'add_1'

visualize_layer(xception, layer=layer, epochs=250, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'add_3'

visualize_layer(xception, layer=layer, epochs=250, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'add_5'

visualize_layer(xception, layer=layer, epochs=400, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'add_7'

visualize_layer(xception, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'add_9'

visualize_layer(xception, layer=layer, epochs=500, rows=3, cols=4, size=size)
size = [224, 224]

layer = 'add_11'

visualize_layer(xception, layer=layer, epochs=500, rows=3, cols=4, size=size)