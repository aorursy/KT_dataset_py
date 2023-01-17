!ls ../input/anime-data-prep/trainData/
from glob import glob

import tensorflow as tf

from matplotlib import pyplot as plt
class DataLoader:

    

    def __init__(self, image_size=(256, 256)):

        self.image_size = image_size

    

    def load_image(self, image_path):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_png(image)

        image = tf.image.resize(image, size=self.image_size)

        return image

    

    def map_function(self, sketch_path, image_path):

        return self.load_image(sketch_path), self.load_image(image_path)

    

    def get_dataset(self, sketch_path, image_path, batch_size=4, buffer_size=1024):

        dataset = tf.data.Dataset.from_tensor_slices((sketch_path, image_path))

        dataset = dataset.map(

            self.map_function,

            num_parallel_calls=tf.data.experimental.AUTOTUNE

        ).shuffle(buffer_size).batch(batch_size)

        return dataset
dataloader = DataLoader()

dataset = dataloader.get_dataset(

    glob('../input/anime-data-prep/trainData/Sketches/*'),

    glob('../input/anime-data-prep/trainData/Images/*')

)

x, y = next(iter(dataset))

x.shape, y.shape
plt.imshow(tf.cast(x[0], dtype=tf.uint8))
plt.imshow(tf.cast(y[0], dtype=tf.uint8))
def DownsampleBlock(input_tensor, filters, kernel_size=4, stride=2, batch_norm=True):

    '''Downsample Block

    Reference: https://arxiv.org/pdf/1803.05400.pdf

    Params:

        input_tensor    -> Input tensor to the block

        filters         -> Number of filters in Conv Layer

        kernel_size     -> Size of Conv Kernel

        stride          -> Strides of the kernel

        batch_norm      -> Use batch norm or not (Flag)

    '''



    x = tf.keras.layers.Conv2D(

        filters=filters, kernel_size=kernel_size,

        strides=stride, padding='same', use_bias=False,

        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,

                                                                 mode='fan_in',

                                                                 distribution='truncated_normal')

    )(input_tensor)



    if batch_norm:

        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x





def UpsampleBlock(input_tensor, filters, kernel_size=4, stride=2, batch_norm=True):

    '''Upsample Block

    Reference: https://arxiv.org/pdf/1803.05400.pdf

    Params:

        input_tensor    -> Input tensor to the block

        filters         -> Number of filters in ConvTranspose Layer

        kernel_size     -> Size of Conv Kernel

        stride          -> Strides of the kernel

    '''



    x = tf.keras.layers.Conv2DTranspose(

        filters=filters, kernel_size=kernel_size,

        strides=stride, padding='same', use_bias=False,

        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,

                                                                 mode='fan_in',

                                                                 distribution='truncated_normal')

    )(input_tensor)



    if batch_norm:

        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    return x
def Generator(input_shape=(256, 256, 3)):



    '''Generator Notebook

    Reference: https://arxiv.org/pdf/1803.05400.pdf

    Params:

        input_shape -> Shape of Input Tensor

    '''



    input_tensor = tf.keras.Input(shape=input_shape)

    encoder_filters = [64, 64, 128, 256, 512, 512, 512, 512]

    encoder_blocks = []

    decoder_filters = [512, 512, 512, 256, 128, 64, 64]

    decoder_blocks = []



    for i, filter in enumerate(encoder_filters):

        x = DownsampleBlock(input_tensor if i == 0 else x,

                            filter,

                            stride=1 if i == 0 else 2,

                            batch_norm=False if i == 0 else True

                            )

        encoder_blocks.append(x)

    

    for i, filter in enumerate(decoder_filters):

        x = UpsampleBlock(x, filter, batch_norm=True if i != (len(decoder_filters) - 1) else False)

        x = tf.keras.layers.Concatenate()([encoder_blocks[- (i + 2)], x])

        decoder_blocks.append(x)



    output_tensor = tf.keras.layers.Conv2D(filters=3,

                                           kernel_size=1,

                                           strides=1,

                                           padding='same',

                                           activation='tanh')(x)



    return tf.keras.Model(input_tensor, output_tensor)
gen = Generator()
y_pred = gen(x * 2 - 1)

y_pred.shape
plt.imshow(y_pred[0])