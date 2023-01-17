"""

Я использовал стандартную капсульную архитектуру из опубликованного исследования.



"""



import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras import backend as K

from keras.layers import Layer

import keras.applications

from keras import activations

from keras import utils

from keras.models import *

from keras.layers import *

from skimage.transform import resize

from skimage import data, color



import keras.backend as K

import tensorflow as tf

from keras import initializers, layers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from keras.layers import *

from keras.preprocessing.image import *



# Мой собственный проект (github.com/l4morak/dsa)



def DSA(x1, inp, out, depth):

    attention_depth = depth

    attention_layer = Dense(inp, activation = 'softmax')

    attention_memory_layer = Dense(inp)

    

    attention = attention_layer(x1)

    never_recount = RepeatVector(attention_depth)(x1)

    attentions = Lambda(lambda y: y[:, 0, :])(never_recount)

    memory = attention_memory_layer(attention)

    for a in range(1, attention_depth):

        extracted_mem = Lambda(lambda x: x[:, a, :] - memory, name = 'Memory_extraction_n_' + str(a))(never_recount)

        attention = attention_layer(extracted_mem)

        attentions = concatenate([attentions, attention], name = 'concatenate_attentions_{0}_{1}'.format(a-1, a))

        memory = add([memory, attention_memory_layer(attention)])

    attentions = Reshape((attention_depth, inp), name = 'Reshape_to_sequence')(attentions)

    attented = multiply([x1, attentions], name = 'Multiplication')

    x = Bidirectional(CuDNNGRU(out, name = 'Bi-RNN'))(attented) 



    return x





# Any results you write to the current directory are saved as output.
path = '../input/dlschool-fashionmnist2/'



train_df = pd.read_csv(f'{path}fashion-mnist_train.csv')

test_df = pd.read_csv(f'{path}fashion-mnist_test.csv')
train_df.head(5)
y_train = train_df['label'].values.reshape(-1, 1)

x_train = train_df.drop(columns = ['label']).values.reshape(-1, 28, 28, 1).astype(np.float32)

x_test = test_df.values.reshape(-1, 28, 28, 1).astype(np.float32)

from keras.utils import to_categorical

y_train = to_categorical(y_train)
imgen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, horizontal_flip = True,

                           rotation_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1, validation_split = 0.1)

imgen.fit(x_train)

x_test = imgen.standardize(x_test)
# the squashing function.

# we use 0.5 in stead of 1 in hinton's paper.

# if 1, the norm of vector will be zoomed out.

# if 0.5, the norm will be zoomed in while original norm is less than 0.5

# and be zoomed out while original norm is greater than 0.5.

def squash(x, axis=-1):

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)

    return scale * x





# define our own softmax function instead of K.softmax

# because K.softmax can not specify axis.

def softmax(x, axis=-1):

    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))

    return ex / K.sum(ex, axis=axis, keepdims=True)





# define the margin loss like hinge loss

def margin_loss(y_true, y_pred):

    lamb, margin = 0.5, 0.1

    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (

        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)





class Capsule(Layer):

    """A Capsule Implement with Pure Keras

    There are two vesions of Capsule.

    One is like dense layer (for the fixed-shape input),

    and the other is like timedistributed dense (for various length input).



    The input shape of Capsule must be (batch_size,

                                        input_num_capsule,

                                        input_dim_capsule

                                       )

    and the output shape is (batch_size,

                             num_capsule,

                             dim_capsule

                            )



    Capsule Implement is from https://github.com/bojone/Capsule/

    Capsule Paper: https://arxiv.org/abs/1710.09829

    """



    def __init__(self,

                 num_capsule,

                 dim_capsule,

                 routings=3,

                 share_weights=True,

                 activation='squash',

                 **kwargs):

        super(Capsule, self).__init__(**kwargs)

        self.num_capsule = num_capsule

        self.dim_capsule = dim_capsule

        self.routings = routings

        self.share_weights = share_weights

        if activation == 'squash':

            self.activation = squash

        else:

            self.activation = activations.get(activation)



    def build(self, input_shape):

        input_dim_capsule = input_shape[-1]

        if self.share_weights:

            self.kernel = self.add_weight(

                name='capsule_kernel',

                shape=(1, input_dim_capsule,

                       self.num_capsule * self.dim_capsule),

                initializer='glorot_uniform',

                trainable=True)

        else:

            input_num_capsule = input_shape[-2]

            self.kernel = self.add_weight(

                name='capsule_kernel',

                shape=(input_num_capsule, input_dim_capsule,

                       self.num_capsule * self.dim_capsule),

                initializer='glorot_uniform',

                trainable=True)



    def call(self, inputs):

        """Following the routing algorithm from Hinton's paper,

        but replace b = b + <u,v> with b = <u,v>.



        This change can improve the feature representation of Capsule.



        However, you can replace

            b = K.batch_dot(outputs, hat_inputs, [2, 3])

        with

            b += K.batch_dot(outputs, hat_inputs, [2, 3])

        to realize a standard routing.

        """



        if self.share_weights:

            hat_inputs = K.conv1d(inputs, self.kernel)

        else:

            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])



        batch_size = K.shape(inputs)[0]

        input_num_capsule = K.shape(inputs)[1]

        hat_inputs = K.reshape(hat_inputs,

                               (batch_size, input_num_capsule,

                                self.num_capsule, self.dim_capsule))

        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))



        b = K.zeros_like(hat_inputs[:, :, :, 0])

        for i in range(self.routings):

            c = softmax(b, 1)

            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))

            if i < self.routings - 1:

                b = K.batch_dot(o, hat_inputs, [2, 3])

                if K.backend() == 'theano':

                    o = K.sum(o, axis=1)



        return o



    def compute_output_shape(self, input_shape):

        return (None, self.num_capsule, self.dim_capsule)



callbacks = [keras.callbacks.ModelCheckpoint('model_capsule.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1), 

            keras.callbacks.LearningRateScheduler(lambda x: 1/(10**(3+x/10)), verbose=0)]



input_image = Input(shape=(28, 28, 1))

x = Conv2D(64, (3, 3), activation='relu')(input_image)

x = Conv2D(64, (3, 3), activation='relu')(x)

x = AveragePooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu')(x)

x = Conv2D(128, (3, 3), activation='relu')(x)



"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)

then connect a Capsule layer.



the output of final model is the lengths of 10 Capsule, whose dim=16.



the length of Capsule is the proba,

so the problem becomes a 10 two-classification problem.

"""



x = Reshape((-1, 128))(x)

capsule = Capsule(10, 30, 3, True)(x)

output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

model = Model(inputs=input_image, outputs=output)



# we use a margin loss

model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit_generator(imgen.flow(x_train, y_train, subset = 'training', batch_size = 128), steps_per_epoch = 1000, epochs = 25, verbose = 1, callbacks = callbacks,

                    validation_data = imgen.flow(x_train, y_train, subset = 'validation', batch_size = 128), validation_steps = 300)
model.load_weights('model_capsule.hdf5')
from keras.models import Model

from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense

from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras import backend as K



weight_decay = 0.0005



def initial_conv(input):

    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(input)



    channel_axis = 1 if K.image_data_format() == "channels_first" else -1



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    return x





def expand_conv(init, base, k, strides=(1, 1)):

    x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(init)



    channel_axis = 1 if K.image_data_format() == "channels_first" else -1



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)



    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(init)



    m = Add()([x, skip])



    return m





def conv1_block(input, k=1, dropout=0.0):

    init = input



    channel_axis = 1 if K.image_data_format() == "channels_first" else -1



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)

    x = Activation('relu')(x)

    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    if dropout > 0.0: x = Dropout(dropout)(x)



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    m = Add()([init, x])

    return m



def conv2_block(input, k=1, dropout=0.0):

    init = input



    channel_axis = 1 if K.image_dim_ordering() == "th" else -1



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)

    x = Activation('relu')(x)

    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    if dropout > 0.0: x = Dropout(dropout)(x)



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    m = Add()([init, x])

    return m



def conv3_block(input, k=1, dropout=0.0):

    init = input



    channel_axis = 1 if K.image_dim_ordering() == "th" else -1



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)

    x = Activation('relu')(x)

    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    if dropout > 0.0: x = Dropout(dropout)(x)



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)

    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',

                      W_regularizer=l2(weight_decay),

                      use_bias=False)(x)



    m = Add()([init, x])

    return m



def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):

    """

    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object

    :param nb_classes: Number of output classes

    :param N: Depth of the network. Compute N = (n - 4) / 6.

              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2

              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4

              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6

    :param k: Width of the network.

    :param dropout: Adds dropout if value is greater than 0.0

    :param verbose: Debug info to describe created WRN

    :return:

    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1



    ip = Input(shape=input_dim)



    x = initial_conv(ip)

    nb_conv = 4



    x = expand_conv(x, 16, k)

    nb_conv += 2



    for i in range(N - 1):

        x = conv1_block(x, k, dropout)

        nb_conv += 2



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)



    x = expand_conv(x, 32, k, strides=(2, 2))

    nb_conv += 2



    for i in range(N - 1):

        x = conv2_block(x, k, dropout)

        nb_conv += 2



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)



    x = expand_conv(x, 64, k, strides=(2, 2))

    nb_conv += 2



    for i in range(N - 1):

        x = conv3_block(x, k, dropout)

        nb_conv += 2



    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)

    x = Activation('relu')(x)



    x = AveragePooling2D((4, 4))(x)

    x = Flatten()(x)

    

    x = Dense(nb_classes, W_regularizer=l2(weight_decay), activation='softmax')(x)



    model = Model(ip, x)



    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))

    return model
callbacks = [keras.callbacks.ModelCheckpoint('model_wrn.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1), 

            keras.callbacks.LearningRateScheduler(lambda x: 1/(10**(3+x/10)), verbose=0)]



wrn_28_10 = create_wide_residual_network((28, 28, 1), nb_classes=10, N=2, k=2, dropout=0.0)

wrn_28_10.summary()

wrn_28_10.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['categorical_accuracy'])

wrn_28_10.fit_generator(imgen.flow(x_train, y_train, subset = 'training', batch_size = 128), steps_per_epoch = 1000, epochs = 25, verbose = 1, callbacks = callbacks,

                    validation_data = imgen.flow(x_train, y_train, subset = 'validation', batch_size = 128), validation_steps = 300)
wrn_28_10.load_weights('model_wrn.hdf5')
sample_submission = pd.read_csv(f'{path}sample_submission.csv')

ensemble = (wrn_28_10.predict(x_test) + model.predict(x_test)) / 2

sample_submission['Category'] = ensemble.argmax(axis=1)

sample_submission.to_csv('ensemble.csv', index = False)



sample_submission['Category'] = wrn_28_10.predict(x_test).argmax(axis=1)

sample_submission.to_csv('wrn.csv', index = False)



sample_submission['Category'] = model.predict(x_test).argmax(axis=1)

sample_submission.to_csv('capsule.csv', index = False)