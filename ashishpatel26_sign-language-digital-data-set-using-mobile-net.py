#https://github.com/rcmalli/keras-mobilenet/blob/master/keras_mobilenet/depthwise_conv2d.py
from keras import backend as K, initializers, regularizers, constraints
from keras.backend import image_data_format
from keras.backend.tensorflow_backend import _preprocess_conv2d_input, _preprocess_padding
from keras.engine.topology import InputSpec
import tensorflow as tf
from keras.layers import Conv2D
from keras.legacy.interfaces import conv2d_args_preprocessor, generate_legacy_interface
from keras.utils import conv_utils
# This code mostly is taken form Keras: Separable Convolution Layer source code and changed according to needs.


def depthwise_conv2d_args_preprocessor(args, kwargs):
    converted = []
    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['depthwise_initializer'] = init
        converted.append(('init', 'depthwise_initializer'))
    args, kwargs, _converted = conv2d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

legacy_depthwise_conv2d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=depthwise_conv2d_args_preprocessor)


class DepthwiseConv2D(Conv2D):

    @legacy_depthwise_conv2d_support
    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 depth_multiplier=1,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)

        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`SeparableConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):

        if self.data_format is None:
            data_format = image_data_format()
        if self.data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))

        x = _preprocess_conv2d_input(inputs, self.data_format)
        padding = _preprocess_padding(self.padding)
        strides = (1,) + self.strides + (1,)

        outputs = tf.nn.depthwise_conv2d(inputs, self.depthwise_kernel,
                                         strides=strides,
                                         padding=padding,
                                         rate=self.dilation_rate)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], self.filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, self.filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

DepthwiseConvolution2D = DepthwiseConv2D
import numpy as np
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import metrics
from keras import losses
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Maximum
from keras.models import Model
from keras.engine.topology import get_source_inputs
def MobileNet(alpha=1.0, shape=[64,64,1]):
    
    img_input = Input(shape)
    
    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    
    out = Dense(10, activation='softmax')(x)

    model = Model(img_input, out, name='mobilenet')
    return model
   
def get_callbacks(filepath, patience=4):
    mcp = ModelCheckpoint(filepath,
                          monitor='val_loss',
                          verbose=0,
                          save_best_only=True,
                          save_weights_only=False,
                          mode='min',
                          period=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=patience, min_lr=1e-16, verbose=1)
    return [mcp, rlr]
def main():
    x_load = np.load('../input/Sign-language-digits-dataset/X.npy')
    y = np.load('../input/Sign-language-digits-dataset/Y.npy')
    print(y)
    X = x_load.reshape([len(x_load),64,64,1])
    
    x_train, x_valid, y_train, y_valid = train_test_split(X,y,
                                                        shuffle=True,
                                                        random_state=32,
                                                        train_size=0.75,
                                                        stratify = y)
    
    model = MobileNet(alpha=1.0)
    callbacks=get_callbacks('weight.hdf5', patience=4)
    #opt = SGD(lr=1e-1, momentum=0.9, decay=0.1, nesterov=True)
    opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    gen = ImageDataGenerator(rotation_range=10,
                            zoom_range=0.1,
                            vertical_flip=True,
                            width_shift_range=0.1,
                            height_shift_range=0.1)
    model.fit_generator(gen.flow(x_train, y_train,batch_size = 32),
            steps_per_epoch=600,
            epochs=40,
            validation_data=(x_valid, y_valid),
            verbose=2,
            callbacks=callbacks
    )
    
if __name__=='__main__':
    main()