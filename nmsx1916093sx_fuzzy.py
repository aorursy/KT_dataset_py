use_tpu = False



if use_tpu:

    import re

    import tensorflow as tf

    import numpy as np

    from matplotlib import pyplot as plt

    print("Tensorflow version " + tf.__version__)

    AUTO = tf.data.experimental.AUTOTUNE

    from kaggle_datasets import KaggleDatasets

    # Detect hardware, return appropriate distribution strategy

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

        print('Running on TPU ', tpu.master())

    except ValueError:

        tpu = None



    if tpu:

        tf.config.experimental_connect_to_cluster(tpu)

        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)

    else:

        strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



    print("REPLICAS: ", strategy.num_replicas_in_sync)



print('Done!')
import numpy as np

import tensorflow as tf

import keras

from keras import backend as K

from keras import activations, initializers, regularizers, constraints

from keras.models import Sequential, Model

from keras.layers import Layer, Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten



class fuzzyDense(Layer):

    '''

    fuzzy convolution layer

    

    outputs > 0：output_dim = outputs

    outputs == 0：# 不sum, output_dim = rules

    outputs == -1：# 不sum, 无k和b, output_dim = rules

        

    '''



    def __init__(self, rules,

                 outputs,

                 membership_function='gaussian',

                 residual=None, # can be None

                 constant_output_MF=True,

                 activation=None,

                 use_output_bias=False,

                 

                 premise_initializer='glorot_uniform',

                 consequent_kernel_initialize='glorot_uniform',

                 consequent_bias_initialize='glorot_uniform',

                 output_bias_initializer='zeros',

                 

                 premise_regularizer=None,

                 consequent_kernel_regularizer=None,

                 consequent_bias_regularizer=None,

                 output_bias_regularizer=None,

                 activity_regularizer=None,

                 

                 premise_constraint=None,

                 consequent_kernel_constraint=None,

                 consequent_bias_constraint=None,

                 output_bias_constraint=None,

                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:

            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(fuzzyDense, self).__init__(**kwargs)

        

        self.rules = rules

        self.outputs = outputs

        self.membership_function = membership_function

        self.residual = residual

        self.constant_output_MF = constant_output_MF

        self.activation = activations.get(activation)

        self.use_output_bias = use_output_bias

                 

        self.premise_initializer = initializers.get(premise_initializer)

        self.consequent_kernel_initialize = initializers.get(consequent_kernel_initialize)

        self.consequent_bias_initialize = initializers.get(consequent_bias_initialize)

        self.output_bias_initializer = initializers.get(output_bias_initializer)

                 

        self.premise_regularizer = regularizers.get(premise_regularizer)

        self.consequent_kernel_regularizer = regularizers.get(consequent_kernel_regularizer)

        self.consequent_bias_regularizer = regularizers.get(consequent_bias_regularizer)

        self.output_bias_regularizer = regularizers.get(output_bias_regularizer)

        self.activity_regularizer = regularizers.get(activity_regularizer)

                 

        self.premise_constraint = constraints.get(premise_constraint)

        self.consequent_kernel_constraint = constraints.get(consequent_kernel_constraint)

        self.consequent_bias_constraint = constraints.get(consequent_bias_constraint)

        self.output_bias_constraint = constraints.get(output_bias_constraint)

        

        #self.input_spec = InputSpec(min_ndim=2)

        #self.supports_masking = True



    def build(self, input_shape):

        assert len(input_shape) >= 2

        self.output_dim = self.outputs if self.outputs > 0 else self.rules

        self.input_dim = input_shape[-1]

        

        if self.membership_function == 'gaussian':

            self.premise_mu = self.add_weight(shape=(self.input_dim, self.rules),

                                              initializer=self.premise_initializer,

                                              name='premise_mu',

                                              regularizer=self.premise_regularizer,

                                              constraint=self.premise_constraint,

                                              trainable=True)

            self.premise_sigma = self.add_weight(shape=(self.input_dim, self.rules),

                                                 initializer=self.premise_initializer,

                                                 name='premise_sigma',

                                                 regularizer=self.premise_regularizer,

                                                 constraint=self.premise_constraint,

                                                 trainable=True)

        else:

            raise ValueError('you can only use gaussian membership function')

            

        if self.outputs >= 0:

            if self.residual is not None:

                consequent_kernel_shape = (self.rules+1, self.input_dim, self.output_dim)

                consequent_bias_shape = (self.rules+1, self.output_dim)

                self.residual = K.constant(self.residual)

            else:

                consequent_kernel_shape = (self.rules, self.input_dim, self.output_dim)

                consequent_bias_shape = (self.rules, self.output_dim)

            if self.constant_output_MF == False:

                self.consequent_kernel = self.add_weight(shape=consequent_kernel_shape,

                                                         initializer=self.consequent_kernel_initialize,

                                                         name='consequent_kernel',

                                                         regularizer=self.consequent_kernel_regularizer,

                                                         constraint=self.consequent_kernel_constraint,

                                                         trainable=True)

            else:

                self.consequent_kernel = None

            self.consequent_bias = self.add_weight(shape=consequent_bias_shape,

                                                   initializer=self.consequent_bias_initialize,

                                                   name='consequent_bias',

                                                   regularizer=self.consequent_bias_regularizer,

                                                   constraint=self.consequent_bias_constraint,

                                                   trainable=True)

            if self.use_output_bias:

                self.output_bias = self.add_weight(shape=(self.output_dim,),

                                                   initializer=self.output_bias_initialize,

                                                   name='output_bias',

                                                   regularizer=self.output_bias_regularizer,

                                                   constraint=self.output_bias_constraint,

                                                   trainable=True)

            else:

                self.output_bias = None



        super(fuzzyDense, self).build(input_shape)

        #self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})

        #self.built = True



    def call(self, inputs):

        a = K.expand_dims(inputs, axis=-1)  # [sample_num, input_dim, 1]

        out_1 = tf.add(a, self.premise_mu)  # [sample_num, input_dim, rules]

        out_1 = -tf.divide(K.square(out_1), K.square(self.premise_sigma) + K.epsilon()) # [sample_num, input_dim, rules]

        

        out_2 = K.sum(out_1, axis=1)  # [sample_num, rules]

        

        if self.residual is not None:

            out_3 = K.exp(out_2)  # [sample_num, rules]

            

            a = K.sum(out_3, axis=-1, keepdims=True) + self.residual # [sample_num, 1]

            b = tf.divide(out_3, a)  # [sample_num, rules]

            b0 = tf.divide(self.residual, a)  # [sample_num, 1]

            out_4 = K.concatenate([b, b0])  # [sample_num, rules+1]

        else:

            out_4 = K.softmax(out_2) # [sample_num, rules]

        

        if self.constant_output_MF:

            out_5 = self.consequent_bias # [rules+1, output_dim]

        else:

            out_5 = tf.add(K.dot(inputs, self.consequent_kernel), self.consequent_bias)  # [sample_num, rules+1, output_dim]

        

        out_4 = K.expand_dims(out_4, axis=-2)  # [sample_num, 1, rules+1]

        out_6 = tf.matmul(out_4, out_5)  # [sample_num, 1, output_dim]

        out_6 = K.squeeze(out_6, axis=-2)  # [sample_num, output_dim]        



        if self.use_output_bias:

            out_6 = K.bias_add(out_6, self.output_bias, data_format='channels_last')

        if self.activation is not None:

            out_6 = self.activation(out_6)

        return out_6



    def compute_output_shape(self, input_shape):

        assert input_shape and len(input_shape) >= 2

        assert input_shape[-1]

        output_shape = list(input_shape)

        output_shape[-1] = self.output_dim

        return tuple(output_shape)

    '''

    def get_config(self):

        config = {

            'units': self.units,

            'activation': activations.serialize(self.activation),

            'use_bias': self.use_bias,

            'kernel_initializer': initializers.serialize(self.kernel_initializer),

            'bias_initializer': initializers.serialize(self.bias_initializer),

            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),

            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'activity_regularizer':

                regularizers.serialize(self.activity_regularizer),

            'kernel_constraint': constraints.serialize(self.kernel_constraint),

            'bias_constraint': constraints.serialize(self.bias_constraint)

        }

        base_config = super(Dense, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    '''







#x = np.random.rand(10, 18)

#model = Sequential()

#model.add(fuzzyDense(rules=5, outputs=7, residual=0.01, constant_output_MF=False, input_shape=(18,)))

#print(model.summary())

#pred = model.predict(x)

#print(pred.shape)

print('Done!')
import numpy as np

import tensorflow as tf

import keras

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Layer, Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten

from keras.initializers import Constant

from keras import initializers, regularizers, constraints

from keras.constraints import MinMaxNorm



def normalize_tuple(value, n, name):

    """Transforms a single int or iterable of ints into an int tuple.

    # Arguments

        value: The value to validate and convert. Could be an int, or any iterable

          of ints.

        n: The size of the tuple to be returned.

        name: The name of the argument being validated, e.g. `strides` or

          `kernel_size`. This is only used to format error messages.

    # Returns

        A tuple of n integers.

    # Raises

        ValueError: If something else than an int/long or iterable thereof was

        passed.

    """

    if isinstance(value, int):

        return (value,) * n

    else:

        try:

            value_tuple = tuple(value)

        except TypeError:

            raise ValueError('The `{}` argument must be a tuple of {} '

                             'integers. Received: {}'.format(name, n, value))

        if len(value_tuple) != n:

            raise ValueError('The `{}` argument must be a tuple of {} '

                             'integers. Received: {}'.format(name, n, value))

        for single_value in value_tuple:

            try:

                int(single_value)

            except ValueError:

                raise ValueError('The `{}` argument must be a tuple of {} '

                                 'integers. Received: {} including element {} '

                                 'of type {}'.format(name, n, value, single_value,

                                                     type(single_value)))

    return value_tuple



def conv_output_length(input_length, filter_size, padding, stride, dilation=1):

    """Determines output length of a convolution given input length.

    # Arguments

        input_length: integer.

        filter_size: integer.

        padding: one of `"same"`, `"valid"`, `"full"`.

        stride: integer.

        dilation: dilation rate, integer.

    # Returns

        The output length (integer).

    """

    if input_length is None:

        return None

    assert padding in {'same', 'valid', 'full', 'causal'}

    dilated_filter_size = (filter_size - 1) * dilation + 1

    if padding == 'same':

        output_length = input_length

    elif padding == 'valid':

        output_length = input_length - dilated_filter_size + 1

    elif padding == 'causal':

        output_length = input_length

    elif padding == 'full':

        output_length = input_length + dilated_filter_size - 1

    return (output_length + stride - 1) // stride



class fuzzyConv(Layer):

    '''

    fuzzy convolution layer

    

    if rules > 0:

        outputs > 0：output_dim = outputs

        outputs == 0：# 不sum, output_dim = rules

        outputs == -1：# 不sum, 无k和b, output_dim = rules

    else:

        outputs must > 0, and output_dim = outputs,

        default use residual regardless residual=None

        fuzzyConv is performing a traditional convolution.

    

    # Input shape

        4D tensor with shape:

        `(batch, channels, rows, cols)`

        if `data_format` is `"channels_first"`

        or 4D tensor with shape:

        `(batch, rows, cols, channels)`

        if `data_format` is `"channels_last"`.

    # Output shape

        4D tensor with shape:

        `(batch, output_dim, new_rows, new_cols)`

        if `data_format` is `"channels_first"`

        or 4D tensor with shape:

        `(batch, new_rows, new_cols, output_dim)`

        if `data_format` is `"channels_last"`.

        `rows` and `cols` values might have changed due to padding.

        

    '''



    def __init__(self, kernel_size,

                 rules,

                 outputs,

                 membership_function='gaussian',

                 T_norm = 'prod', #or min

                 use_residual=True,

                 residual_trainable=False,

                 constant_output_MF=True,  # only work when outputs >= 0

                 

                 channel_multiplier=1,  # work only when kernel_size != (1,1) and constant_output_MF==False work

                 strides=(1, 1),

                 padding='valid',

                 data_format='channels_last',

                 dilation_rate=(1, 1),

                 

                 residual_initializer=Constant(value=0.01),

                 premise_kernel_initializer='glorot_uniform',

                 premise_bias_initializer='glorot_uniform',

                 consequent_kernel_initializer='glorot_uniform',  # only work when constant_output_MF==False work

                 consequent_bias_initializer='glorot_uniform',  # only work when outputs >= 0

                 

                 residual_regularizer=None,

                 premise_kernel_regularizer=None,

                 premise_bias_regularizer=None,

                 consequent_kernel_regularizer=None,

                 consequent_bias_regularizer=None,

                 

                 residual_constraint=MinMaxNorm(min_value=0.00001, max_value=1.0),

                 premise_kernel_constraint=None,

                 premise_bias_constraint=None,

                 consequent_kernel_constraint=None,

                 consequent_bias_constraint=None,

                 **kwargs):

        super(fuzzyConv, self).__init__(**kwargs)

        

        #self.padding = conv_utils.normalize_padding(padding)

        #self.data_format = K.normalize_data_format(data_format)

        

        self.kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')

        self.rules = rules

        self.outputs = outputs

        self.membership_function = membership_function

        self.T_norm = T_norm

        self.use_residual = use_residual

        self.residual_trainable = residual_trainable

        self.constant_output_MF = constant_output_MF

        

        self.residual_initializer = initializers.get(residual_initializer)

        self.premise_kernel_initializer = initializers.get(premise_kernel_initializer)

        self.premise_bias_initializer = initializers.get(premise_bias_initializer)

        self.consequent_kernel_initializer = initializers.get(consequent_kernel_initializer)

        self.consequent_bias_initializer = initializers.get(consequent_bias_initializer)

        

        self.channel_multiplier = channel_multiplier

        self.strides = normalize_tuple(strides, 2, 'strides')

        self.padding = padding

        self.data_format = data_format

        self.dilation_rate = normalize_tuple(dilation_rate, 2, 'dilation_rate')

        

        self.residual_regularizer = regularizers.get(residual_regularizer)

        self.premise_kernel_regularizer = regularizers.get(premise_kernel_regularizer)

        self.premise_bias_regularizer = regularizers.get(premise_kernel_regularizer)

        self.consequent_kernel_regularizer = regularizers.get(consequent_kernel_regularizer)

        self.consequent_bias_regularizer = regularizers.get(consequent_bias_regularizer)

        

        self.residual_constraint =  constraints.get(residual_constraint)

        self.premise_kernel_constraint = constraints.get(premise_kernel_constraint)

        self.premise_bias_constraint = constraints.get(premise_bias_constraint)

        self.consequent_kernel_constraint = constraints.get(consequent_kernel_constraint)

        self.consequent_bias_constraint = constraints.get(consequent_bias_constraint)

        

        #self.input_spec = InputSpec(ndim=self.rank + 2)



    def build(self, input_shape):

        if len(input_shape) != 4:

            raise ValueError('input rank should be 4.')

        if self.data_format == 'channels_first':

            channel_axis = 1

        else:

            channel_axis = -1

        if input_shape[channel_axis] is None:

            raise ValueError('The channel dimension of the inputs '

                             'should be defined. Found `None`.')

        

        if self.use_residual:

            self.residual = self.add_weight(shape = (1,),

                                            initializer = self.residual_initializer,

                                            name = 'residual',

                                            regularizer = self.residual_regularizer,

                                            constraint = self.residual_constraint,

                                            trainable = self.residual_trainable)

            temp = 1

        else:

            temp = 0

        

        self.output_dim = self.outputs if self.outputs > 0 else self.rules+temp

        self.output_mul = self.outputs if self.outputs > 0 else 1

        self.input_dim = input_shape[channel_axis]

        

        if self.membership_function == 'gaussian':

            if self.kernel_size == (1,1):

                self.premise_kernel = self.add_weight(shape = (self.rules, self.input_dim),

                                                      initializer = self.premise_kernel_initializer,

                                                      name = 'premise_sigma',

                                                      regularizer = self.premise_kernel_regularizer,

                                                      constraint = self.premise_kernel_constraint,

                                                      trainable = True)

                self.premise_bias = self.add_weight(shape = (self.rules, self.input_dim),

                                                    initializer = self.premise_bias_initializer,

                                                    name = 'premise_mu',

                                                    regularizer = self.premise_bias_regularizer,

                                                    constraint = self.premise_bias_constraint,

                                                    trainable = True)

                if self.constant_output_MF == False and self.outputs >= 0:

                    self.consequent_kernel = self.add_weight(shape = self.kernel_size + (self.input_dim, (self.rules+temp)*self.output_mul),

                                                             initializer = self.consequent_kernel_initializer,

                                                             name = 'consequent_kernel',

                                                             regularizer = self.consequent_kernel_regularizer,

                                                             constraint = self.consequent_kernel_constraint,

                                                             trainable = True)

                    #self.consequent_kernel = self.add_weight(shape = (self.rules+temp, self.input_dim, self.output_mul),

                    #                                         initializer = self.consequent_kernel_initialize,

                    #                                         name = 'consequent_kernel',

                    #                                         regularizer = self.consequent_kernel_regularizer,

                    #                                         constraint = self.consequent_kernel_constraint,

                    #                                         trainable = True)

            else:

                self.premise_kernel = self.add_weight(shape = self.kernel_size + (self.input_dim, self.rules),

                                                      initializer = self.premise_kernel_initializer,

                                                      name = 'premise_kernel',

                                                      regularizer = self.premise_kernel_regularizer,

                                                      constraint = self.premise_kernel_constraint,

                                                      trainable = True)

                self.premise_bias = self.add_weight(shape = (self.rules, self.input_dim),

                                                    initializer = self.premise_bias_initializer,

                                                    name = 'premise_bias',

                                                    regularizer = self.premise_bias_regularizer,

                                                    constraint = self.premise_bias_constraint,

                                                    trainable = True)



                if self.constant_output_MF == False and self.outputs >= 0:

                    if self.channel_multiplier is not None and self.channel_multiplier > 0:

                        self.consequent_kernel_1 = self.add_weight(shape = self.kernel_size + (self.input_dim, self.channel_multiplier),

                                                                   initializer = self.consequent_kernel_initializer,

                                                                   name = 'consequent_depthwise_kernel',

                                                                   regularizer = self.consequent_kernel_regularizer,

                                                                   constraint = self.consequent_kernel_constraint,

                                                                   trainable = True)

                        self.consequent_kernel_2 = self.add_weight(shape = (1, 1, self.input_dim*self.channel_multiplier, (self.rules+temp)*self.output_mul),

                                                                   initializer = self.consequent_kernel_initializer,

                                                                   name = 'consequent_pointwise_kernel',

                                                                   regularizer = self.consequent_kernel_regularizer,

                                                                   constraint = self.consequent_kernel_constraint,

                                                                   trainable = True)

                    else:

                        self.consequent_kernel = self.add_weight(shape = self.kernel_size + (self.input_dim, (self.rules+temp)*self.output_mul),

                                                                 initializer = self.consequent_kernel_initializer,

                                                                 name = 'consequent_kernel',

                                                                 regularizer = self.consequent_kernel_regularizer,

                                                                 constraint = self.consequent_kernel_constraint,

                                                                 trainable = True)

            

            if self.outputs >= 0:

                self.consequent_bias = self.add_weight(shape = (self.rules+temp, self.output_mul),

                                                       initializer = self.consequent_bias_initializer,

                                                       name = 'consequent_bias',

                                                       regularizer = self.consequent_bias_regularizer,

                                                       constraint = self.consequent_bias_constraint,

                                                       trainable = True)

        else:

            raise ValueError('you can only use gaussian membership function')



        super(fuzzyConv, self).build(input_shape)



    def call(self, inputs): # inputs_shape = (sample_num, rows, cols, channels)

        # step 1, -((x-mu)^2/(sigma^2+epsilon()), output_shape = (sample_num, rows, cols, input_dim, rules)

        # step 2, sum, output_shape = (sample_num, rows, cols, rules)

        # step 3, add residual, output_shape = (sample_num, rows, cols, rules+1)

        # step 4, normalize or rsoftmax， output_shape = (sample_num, rows, cols, rules+1)

        # step 5, kx+b, output_shape = (sample_num, rows, cols, rules+1, output_mul)

        # step 6, (kx+b)*softmax, output_shape = (sample_num, rows, cols, output_dim)

        

        if self.kernel_size == (1,1):

            out_1 = K.expand_dims(inputs, axis=-2)  # [sample_num, rows, cols, 1, input_dim]

            out_1 = tf.add(out_1, self.premise_bias)  # [sample_num, rows, cols, rules, input_dim]

            out_1 = tf.divide(K.square(out_1), K.square(self.premise_kernel) + K.epsilon()) # [sample_num, rows, cols, rules, input_dim]

        else:

            out_1 = K.depthwise_conv2d(inputs, self.premise_kernel, strides=self.strides, padding=self.padding,

                                       data_format=self.data_format, dilation_rate=self.dilation_rate)

            # [sample_num, rows, cols, rules*input_dim]

            # output[b, i, j, k * rules + q] = sum_{di, dj} filter[di, dj, k, q] * input[b, i + di, j + dj, k]

            out_1 = K.reshape(out_1, shape=(-1, out_1.shape[1], out_1.shape[2], self.rules, self.input_dim))  # [sample_num, rows, cols, rules, input_dim]

            out_1 = K.square(tf.add(out_1, self.premise_bias))  # [sample_num, rows, cols, rules, input_dim]

        

        if self.T_norm == 'prod':

            out_2 = -K.sum(out_1, axis=-1)  # [sample_num, rows, cols, rules]

        else:

            out_2 = -K.max(out_1, axis=-1)  # [sample_num, rows, cols, rules]

        

        if self.use_residual:

            out_3 = K.exp(out_2)  # [sample_num, rows, cols, rules]

        

            a = K.sum(out_3, axis=-1, keepdims=True) + self.residual # [sample_num, rows, cols, 1]

            b = tf.divide(out_3, a)  # [sample_num, rows, cols, rules]

            b0 = tf.divide(self.residual, a)  # [sample_num, rows, cols, 1]

            out_4 = K.concatenate([b, b0])  # [sample_num, rows, cols, rules+1]

        else:

            out_4 = K.softmax(out_2) # [sample_num, rows, cols, rules]

        

        if self.outputs < 0:

            return out_4

        

        if self.constant_output_MF:

            out_5 = self.consequent_bias # [rules+1, output_mul]

        else:    

            #if self.kernel_size == (1,1):

                #out_5 = K.dot(inputs, self.consequent_kernel)

            if self.kernel_size != (1,1) and self.channel_multiplier is not None and self.channel_multiplier > 0:  # use depthwise convolution   

                out_5 = K.separable_conv2d(inputs, self.consequent_kernel_1, self.consequent_kernel_2, strides=self.strides, padding=self.padding,

                                           data_format=self.data_format, dilation_rate=self.dilation_rate)

                # [sample_num, rows, cols, (rules+1)*output_mul]

                # output[b, i, j, k] = sum_{di, dj, q, r} input[b, i + di, j + dj, q] * depthwise_filter[di, dj, q, r] * pointwise_filter[0, 0, q * channel_multiplier + r, k]

            else:

                out_5 = K.conv2d(inputs, self.consequent_kernel, strides=self.strides, padding=self.padding,

                                 data_format=self.data_format, dilation_rate=self.dilation_rate)

                # [sample_num, rows, cols, (rules+1)*output_mul]

                

            out_5 = K.reshape(out_5, shape = (-1, out_5.shape[1], out_5.shape[2], out_4.shape[-1], self.output_mul))  # [sample_num, rows, cols, rules+1, output_mul]

            out_5 = tf.add(out_5, self.consequent_bias)  # [sample_num, rows, cols, rules+1, output_mul]

        

        if self.outputs > 0:

            out_4 = K.expand_dims(out_4, axis=-2)  # [sample_num, rows, cols, 1, rules+1]

            out_6 = tf.matmul(out_4, out_5)  # [sample_num, rows, cols, 1, output_mul]

            out_6 = K.squeeze(out_6, axis=-2)  # [sample_num, rows, cols, output_mul]

        else:

            out_5 = K.squeeze(out_5, axis=-1)  # [sample_num, rows, cols, rules+1]

            out_6 = tf.multiply(out_4, out_5)  # [sample_num, rows, cols, rules+1]



        return out_6

        

    def compute_output_shape(self, input_shape):

        if self.data_format == 'channels_last':

            space = input_shape[1:-1]

        elif self.data_format == 'channels_first':

            space = input_shape[2:]

        new_space = []

        for i in range(len(space)):

            new_dim = conv_output_length(

                space[i],

                self.kernel_size[i],

                padding=self.padding,

                stride=self.strides[i],

                dilation=self.dilation_rate[i])

            new_space.append(new_dim)

        if self.data_format == 'channels_last':

            return (input_shape[0],) + tuple(new_space) + (self.output_dim,)

        elif self.data_format == 'channels_first':

            return (input_shape[0], self.output_dim) + tuple(new_space)

    

'''   

x = np.random.rand(10, 27, 27, 3)

for k in [(1,1),(3,3)]:

    for r in [True, False]:

        for c in [True, False]:

            for cm in [None,1,2,3]:

                for p in ['same', 'valid']:

                    for outputs in [7, 0, 9, -1]:

                        for T in ['prod', 'min']:

                            model = Sequential()

                            model.add(fuzzyConv(k,rules=5, outputs=outputs, T_norm=T, use_residual=r, constant_output_MF=c, channel_multiplier=cm, padding=p,input_shape=(27, 27, 3)))

                            pred = model.predict(x)

                            del model

                            print(pred.shape)



'''



'''

x = np.random.rand(10, 27, 27, 3)

model = Sequential()

model.add(fuzzyConv((1,1),rules=5, outputs=7, residual=0.01, constant_output_MF=False, channel_multiplier=None, padding='same',input_shape=(27, 27, 3)))

print(model.summary())

pred = model.predict(x)

print(pred.shape)

'''

print('Done!')
def fuzzyConv_block(inputs, filters_list, rules_outputs_list, kernel_size=(3,3), residual=0.01, constant_output_MF=False, channel_multiplier=0):

    x = inputs

    for filters in filters_list:

        x = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)

    for rules, outputs in rules_outputs_list:

        inputs = fuzzyConv(kernel_size , rules, outputs, residual=0.01, constant_output_MF=constant_output_MF,

                           channel_multiplier=channel_multiplier, padding='same')(inputs)

    out = Concatenate(axis=-1)([x, inputs])

    out = MaxPooling2D(pool_size=(2, 2))(out)

    return out



def fuzzyConv_block_V2(inputs, filters_list, rules_outputs_list, kernel_size=(3,3), residual=0.01, constant_output_MF=False, channel_multiplier=0):

    assert len(filters_list)==len(rules_outputs_list)

    for i in range(len(filters_list)):

        x = Conv2D(filters_list[i], kernel_size, activation='relu', padding='same')(inputs)

        y = fuzzyConv(kernel_size , rules=rules_outputs_list[i][0], outputs=rules_outputs_list[i][1],

                      residual=0.01, constant_output_MF=constant_output_MF,

                      channel_multiplier=channel_multiplier, padding='same')(inputs)

        inputs = Concatenate(axis=-1)([x, y])

    out = MaxPooling2D(pool_size=(2, 2))(inputs)

    return out



print('Done!')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.utils import HDF5Matrix

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sys import exit

from PIL import Image



save_h5py = False

resize_image = False





data = np.genfromtxt('/kaggle/input/labels.csv', dtype=['|S19', '<f8', '|S4'], names=[

                         'path', 'probability', 'type'])

image_fnames = np.char.decode(data['path'])

probs = data['probability']

types = np.char.decode(data['type'])



images = np.array([np.array(Image.open('/kaggle/input/'+filename))

                       for filename in image_fnames])





#import matplotlib.pyplot as plt



#for i in range(10):

#    plt.figure()

#    #plt.imshow(images[i])

#    plt.imshow(images[i], cmap='Greys_r')

#    plt.show()

    

#print('Done!')



images = np.expand_dims(images, axis=3)



# 2,624 samples

# 300x300 pixels

# 8-bit grayscale

# 44 different solar modules



if save_h5py:

    import h5py

    with h5py.File('image_data.h5', 'w') as f:

        f.create_dataset('data', data=images)

        f.create_dataset('label', data=probs)



if resize_image:

    images = np.array(tf.image.resize(images, (224,224), method='bilinear', preserve_aspect_ratio=False, antialias=False, name=None))

    input_shape = (224, 224, 1)

    if save_h5py:

        import h5py

        with h5py.File('image_data_resize.h5', 'w') as f:

            f.create_dataset('data', data=images)

            f.create_dataset('label', data=probs)

else:

    input_shape = (300, 300, 1)

'''



if resize_image:

    images = HDF5Matrix('/kaggle/input/fuzzynet/image_data_resize.h5', 'data')

    probs = HDF5Matrix('/kaggle/input/fuzzynet/image_data_resize.h5', 'label')

    input_shape = (224, 224, 1)

else:

    images = HDF5Matrix('image_data.h5', 'data')

    probs = HDF5Matrix('image_data.h5', 'label')

    input_shape = (300, 300, 1)

import h5py

with h5py.File('image_data.h5', 'r') as f:

    print(f.keys())

'''

print('Done!')
#use_model = 'resnet'

#use_model = 'vgg19'

#use_model = 'Fuzzykernel'

#use_model = 'my_vgg19'

#use_model = 'my_vgg16'

#use_model = 'other'

#use_model = 'FuzzyBlock'

use_model = 'FuzzyBlockV2'





num_classes=2

one_hot = True

#ues_loss = 'categorical_crossentropy'

use_loss = 'binary_crossentropy'

use_data_augmentation = True

test_split = 0.2

validation_split = 0.2

validation_from_generator = False



def brightness_adj(images, brightness_range):

    m = np.min(np.min(images,axis=2,keepdims=True),axis=1,keepdims=True)

    M = np.max(np.max(images,axis=2,keepdims=True),axis=1,keepdims=True)

    images = (brightness_range[1]-brightness_range[0])*(images-m)/(M-m) + brightness_range[0]

    return images

def brightness_std(images):

    temp = np.reshape(images,(len(images),-1))

    m = np.reshape(np.mean(temp,axis=-1),(len(images),1,1,1))

    s = np.reshape(np.std(temp,axis=-1),(len(images),1,1,1))

    images = (images-m)/s

    return images

def brightness_std_v2(images_train, images_test):

    m = np.mean(images_train)

    s = np.std(images_train)

    images_train = (images_train-m)/s

    images_test = (images_test-m)/s

    return images_train, images_test

    



if use_model == 'vgg19':

    from keras.applications.vgg19 import preprocess_input

    images=np.repeat(images, 3, axis=3)

    images=preprocess_input(images)#/255

    input_shape[-1]=3

elif use_model == 'resnet':

    from keras.applications.resnet_v2 import preprocess_input

    images=np.repeat(images, 3, axis=3)

    images=preprocess_input(images)#/255

    input_shape[-1]=3

else:

    # for simple CNN

    #1

    #images = images

    #images = images/255-0.5

    #images = brightness_adj(images,[0,255])

    #images = brightness_adj(images,[0,1])

    #images = brightness_adj(images,[-0.5,0.5])

    #images = brightness_adj(images,[-1,1])

    images = brightness_std(images)

    #feature_wise_std = False



if num_classes==2:

    probs_type = np.array(probs/0.6,dtype='int8')

else:

    probs_type = np.array(probs/0.3,dtype='int8')

    

#probs_type = np.array(probs_type, dtype='float32')

    

# Generate dummy data

if one_hot:

    from keras.utils import to_categorical

    probs_categorical = to_categorical(probs_type, num_classes=num_classes)

else:

    probs_categorical = np.expand_dims(probs_type, axis=1)

    

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, probs_categorical, test_size = test_split, random_state= 55) # 3 is fucking shit



#if feature_wise_std:

#    x_train, x_test = brightness_std_v2(x_train, x_test)



del data, image_fnames, probs, probs_type, types, images, probs_categorical



if use_data_augmentation:

    from keras.preprocessing.image import ImageDataGenerator

    datagen_args = dict(#featurewise_center=True,

                        #featurewise_std_normalization=True,

                        #zca_whitening=True,  # Fuck, can not use

                        #brightness_range=(0,1),

                        rotation_range=2,

                        width_shift_range=5,

                        height_shift_range=5,

                        horizontal_flip=True,

                        vertical_flip=True)

    if validation_from_generator:

        datagen_args['validation_split'] = validation_split

        datagen = ImageDataGenerator(**datagen_args)

    else:

        datagen = ImageDataGenerator(**datagen_args)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = validation_split, random_state= 7)  # 7 is fucking shit

    

    # compute quantities required for featurewise normalization

    # (std, mean, and principal components if ZCA whitening is applied)

    #datagen.fit(x_train)

    

print('Done!')
from keras.models import Sequential, Model

from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, GlobalMaxPooling2D, Add

from keras.initializers import he_normal

'''

def my_vgg19(input_shape, output_dim, weight_decay=0.0001, dropout=0.3):

    # build model

    model = Sequential()



    # Block 1

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block1_conv1', input_shape=input_shape))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block1_conv2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))



    # Block 2

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block2_conv1'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block2_conv2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))



    # Block 3

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block3_conv1'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block3_conv2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block3_conv3'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block3_conv4'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))



    # Block 4

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block4_conv1'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block4_conv2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block4_conv3'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block4_conv4'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))



    # Block 5

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block5_conv1'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block5_conv2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block5_conv3'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),

                     kernel_initializer=he_normal(), name='block5_conv4'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))



    # model modification

    #model.add(Flatten(name='flatten'))

    model.add(GlobalAveragePooling2D())

    model.add(Dense(1024, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),

                    kernel_initializer=he_normal(), name='fc_cifa10'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(dropout))

    model.add(Dense(1024, kernel_regularizer=keras.regularizers.l2(weight_decay),

                    kernel_initializer=he_normal(), name='fc2'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(dropout))

    model.add(Dense(output_dim, kernel_regularizer=keras.regularizers.l2(weight_decay),

                    kernel_initializer=he_normal(), name='predictions_cifa10'))

    model.add(BatchNormalization())

    model.add(Activation('softmax'))

    

    return model



def my_vgg16(input_shape=input_shape, output_dim=2, dropout=0.2):

    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    model.add(Flatten())

    #model.add(GlobalAveragePooling2D())

    model.add(Dense(4096, activation='relu'))

    #model.add(Dropout(dropout))

    model.add(Dense(4096, activation='relu'))

    #model.add(Dropout(dropout))

    model.add(Dense(output_dim, activation='softmax'))

    

    return model

'''

print('Done!')
from keras.optimizers import SGD, adam, RMSprop

from keras.applications.vgg19 import VGG19

from keras.applications.resnet_v2 import ResNet50V2

from keras.callbacks.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.utils import plot_model, print_summary

        

if one_hot:

    loss = 'categorical_crossentropy'

else:

    loss = use_loss

        

output_dim = num_classes if one_hot else 1



if use_model == 'vgg19' or use_model == 'resnet':

    

    epochs0=100

    opt0=SGD(lr=0.0001, momentum=0.9)

    #opt2=SGD(lr=0.001, momentum=0.9, nesterov=True)

    #opt2=adam(learning_rate=0.001, amsgrad=True)

    

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=5, factor=0.5, min_lr=1e-7, cooldown=5)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]

    

    #input_tensor = Input(shape=(300, 300, 1))

    if use_model == 'vgg19':

        #base_model = VGG19(weights=None, include_top=False, input_shape=(300,300,3))

        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    else:

        base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

    #print(base_model.output_shape)

    # for resnet, (None, 10, 10, 2048)

    # for vgg19, (None, 9, 9, 512)

    

    x = base_model.output

    if use_flatten:

        x = Flatten()(x)

    else:

        x = GlobalAveragePooling2D()(x)

    if use_model == 'vgg19':

        x = Dense(512, activation='relu')(x)

        x = Dense(512, activation='relu')(x)

        #x = Dropout(dropout)(x)

        #x = Dense(1024, activation='relu')(x)

    else:

        x = Dense(2048, activation='relu')(x)

    #x = Dropout(dropout)(x)

    

    predictions = Dense(output_dim, activation='softmax')(x)



    model = Model(inputs=base_model.input, outputs=predictions)



    for layer in base_model.layers:

        layer.trainable = False

    '''

    if use_model == 'vgg19':

        for layer in model.layers[:17]:

            layer.trainable = False

        for layer in model.layers[17:]:

            layer.trainable = True

    else:

        for layer in model.layers[:152]:

            layer.trainable = False

        for layer in model.layers[152:]:

            layer.trainable = True

    '''

    

    model.compile(loss=loss, optimizer=opt0, metrics=['accuracy'])

    

    

    if use_data_augmentation:

        if validation_from_generator:

            hist0 = model.fit_generator(train_generator, steps_per_epoch=len(x_train) / batch_size, epochs=epochs0,

                                validation_data=validation_generator, callbacks=callbacks)

        else:

            hist0 = model.fit_generator(train_generator, steps_per_epoch=len(x_train) / batch_size, epochs=epochs0,

                                validation_data=(x_val, y_val), callbacks=callbacks)

    else:

        hist0 = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs0, 

                  validation_split=validation_split, callbacks=callbacks)



    if use_model == 'vgg19': # 17, 12, 7, 4

        for layer in model.layers[:17]:

            layer.trainable = False

        for layer in model.layers[17:]:

            layer.trainable = True

    else:

        for layer in model.layers[:152]: # 152

            layer.trainable = False

        for layer in model.layers[152:]:

            layer.trainable = True



if use_model == 'my_vgg19':

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.8, min_lr=1e-7, cooldown=5)

    early_stop = EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]

    

    model = my_vgg19(input_shape=input_shape, output_dim=output_dim, weight_decay=0.0001, dropout=0.3)

    

if use_model == 'my_vgg16':

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.8, min_lr=1e-7, cooldown=5)

    early_stop = EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]

    

    model = my_vgg16(input_shape=input_shape, output_dim=output_dim, dropout=0.2)

    

if use_model == 'other':

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=16, factor=0.618, min_lr=1e-7)

    early_stop = EarlyStopping(monitor='val_loss', patience=64, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]



    #with strategy.scope():

    

    model = Sequential()

    

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    #model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    #model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    #model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))

    model.add(Conv2D(1024, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(dropout))

    if use_flatten:

        model.add(Flatten())

    else:

        model.add(GlobalAveragePooling2D())

        #model.add(GlobalMaxPooling2D())

    model.add(Dense(1024, activation='relu'))

    #model.add(Dropout(dropout))

    model.add(Dense(1024, activation='relu'))

    #model.add(Dropout(dropout))

    model.add(Dense(output_dim, activation='softmax'))

    

    

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    



print('Done')
try:

    del model

except:

    model = None



batch_size=48

epochs=200

dropout = 0.2



opt=adam(learning_rate=0.001, amsgrad=True)

#opt=SGD(lr=0.001, momentum=0.9)



use_flatten = False



if use_data_augmentation:

    if validation_from_generator:

        train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset="training")

        validation_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset="validation")

    else:

        train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

    #test_generator = datagen.flow(x_test, y_test, batch_size=batch_size)

    #pred_generator = datagen.flow(x_test, batch_size=batch_size, shuffle=False)



if use_model == 'FuzzyBlockV2':

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.618, min_lr=1e-7, cooldown=2)

    early_stop = EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]

    

    #T = 'prod'

    T = 'min'

    

    rt = True

    

    #coM = True

    coM = False

    

    rules = [8,8,8,8,8] # batch_size=90

    Ac = 'relu'

    

    x0 = Input(shape=input_shape)

    x1 = Conv2D(8, (3,3), activation=Ac, padding='same')(x0)

    x = fuzzyConv(3, rules[0], 16, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x0)

    x = Concatenate(axis=-1)([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(16, (3,3), activation=Ac, padding='same')(x)

    x = fuzzyConv(3, rules[1], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = Concatenate(axis=-1)([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(32, (3,3), activation=Ac, padding='same')(x)

    x = fuzzyConv(3, rules[2], 64, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = Concatenate(axis=-1)([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(64, (3,3), activation=Ac, padding='same')(x)

    x = fuzzyConv(3, rules[3], 128, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = Concatenate(axis=-1)([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(64, (3,3), activation=Ac)(x)

    x = fuzzyConv(3, rules[4], 128, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0)(x)

    x = Concatenate(axis=-1)([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)



    x = GlobalMaxPooling2D()(x)

    x = Dense(192, activation='relu')(x)

    x = Dense(192, activation='relu')(x)

    y = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=x0, outputs=y)

    

if use_model == 'FuzzyBlock':

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.618, min_lr=1e-7, cooldown=2)

    early_stop = EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]

    

    #T = 'prod'

    T = 'min'

    

    rt = True

    

    #coM = True

    coM = False

    

    rules = [16,16,16,16,16]  # batch_size=110

    

    x0 = Input(shape=input_shape)

    x1 = Conv2D(16, (3,3), activation='relu', padding='same')(x0)

    x = fuzzyConv(3, rules[0], 16, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x0)

    x = Add()([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(32, (3,3), activation='relu', padding='same')(x)

    x = fuzzyConv(3, rules[1], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = Add()([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(64, (3,3), activation='relu', padding='same')(x)

    x = fuzzyConv(3, rules[2], 64, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = Add()([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(128, (3,3), activation='relu', padding='same')(x)

    x = fuzzyConv(3, rules[3], 128, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = Add()([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    x1 = Conv2D(128, (3,3), activation='relu')(x)

    x = fuzzyConv(3, rules[4], 128, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0)(x)

    x = Add()([x1, x])

    x = MaxPooling2D(pool_size=(2, 2))(x)



    x = GlobalMaxPooling2D()(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(128, activation='relu')(x)

    y = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=x0, outputs=y)

    

if use_model == 'Fuzzykernel':

    reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.618, min_lr=1e-7, cooldown=5)

    early_stop = EarlyStopping(monitor='val_loss', patience=32, restore_best_weights=True)

    callbacks = [reduce_lr, early_stop]

    

    #T = 'prod'

    T = 'min'

    

    rt = True

    

    #coM = True

    coM = False

    

    rules = [8,8,8,8,8]

    #with strategy.scope():

    

    x0 = Input(shape=input_shape)

    x = fuzzyConv(3, rules[0], 16, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x0)

    #x = fuzzyConv(3, rules[0], 4, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)



    #x = fuzzyConv(3, rules[1], 8, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = fuzzyConv(3, rules[1], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    #x = fuzzyConv(3, rules[2], 16, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    #x = fuzzyConv(3, rules[2], 16, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = fuzzyConv(3, rules[2], 64, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)



    #x = fuzzyConv(3, rules[3], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    #x = fuzzyConv(3, rules[3], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = fuzzyConv(3, rules[3], 128, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    

    #x = fuzzyConv(3, rules[4], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    #x = fuzzyConv(3, rules[4], 32, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0, padding='same')(x)

    x = fuzzyConv(3, rules[4], 128, T_norm=T, residual_trainable=rt, constant_output_MF=coM, channel_multiplier=0)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)



    #x = fuzzyConv_block(x0, [16, 16], [(8, 8)])

    #x = fuzzyConv_block(x, [32, 32], [(16, 16)])

    #x = fuzzyConv_block(x, [64, 64], [(32, 32)])

    #x = fuzzyConv_block(x, [64, 64], [(32, 32)])

    #x = fuzzyConv_block(x, [128, 128], [(64, 64)])

    #x = fuzzyConv_block(x, [128, 128], [(64, 64)])

    

    

    #x = fuzzyConv_block_V2(x0, [16, 16], [(8, 8),(8, 8)])

    #x = fuzzyConv_block_V2(x, [32, 32], [(16, 16),(16, 16)])

    #x = fuzzyConv_block_V2(x, [64, 64], [(32, 32),(32, 32)])

    #x = fuzzyConv_block_V2(x, [64, 64], [(32, 32),(32, 32)])

    #x = fuzzyConv_block_V2(x, [128, 128], [(64, 64),(64, 64)])

    #x = fuzzyConv_block_V2(x, [128, 128], [(64, 64),(64, 64)])

    

    if use_flatten:

        x = Flatten()(x)

    else:

        #x = GlobalAveragePooling2D()(x)

        x = GlobalMaxPooling2D()(x)

    x = Dense(128, activation='relu')(x)

    x = Dense(128, activation='relu')(x)

    y = Dense(output_dim, activation='softmax')(x)

    model = Model(inputs=x0, outputs=y)

    

    #model.add(fuzzyConv((3,3), rules=128, outputs=0, residual=0.01, constant_output_MF=True, channel_multiplier=1, padding='same'))

    #model.add(fuzzyDense(rules=100, outputs=output_dim, residual=0.01, constant_output_MF=True, activation='softmax'))



model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    

print_summary(model)

#plot_model(model, to_file='model.pdf', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True, dpi=100)

print('use model: '+ use_model)



#del model

#exit(0)



if use_data_augmentation:

    if validation_from_generator:

        hist = model.fit_generator(train_generator, steps_per_epoch=len(x_train) / batch_size, epochs=epochs,

                                   validation_data=validation_generator, callbacks=[reduce_lr, early_stop])

    else:

        hist = model.fit_generator(train_generator, steps_per_epoch=len(x_train) / batch_size, epochs=epochs,

                                   validation_data=(x_val, y_val), callbacks=[reduce_lr, early_stop])

else:

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,

                     validation_split=validation_split, callbacks=[reduce_lr, early_stop])



#if use_data_augmentation:

#    score = model.evaluate_generator(test_generator, steps=len(x_test) / batch_size)

#    y_pred = model.predict_generator(pred_generator, steps=len(x_test) / batch_size)

#else:

score = model.evaluate(x_test, y_test, batch_size=batch_size)

y_pred = model.predict(x_test, batch_size=batch_size)



del model

if use_model == 'vgg19' or use_model == 'resnet':

    del base_model



print(score)

print('Done')
import matplotlib.pyplot as plt



epochs = len(hist.history['loss'][9:])

plt.figure(figsize=(16, 9))

plt.plot(range(epochs), hist.history['loss'][9:], label='loss')

plt.plot(range(epochs), hist.history['val_loss'][9:], label='val_loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')

plt.show()

plt.figure(figsize=(16, 9))

plt.plot(range(epochs), hist.history['accuracy'][9:], label='accuracy')

plt.plot(range(epochs), hist.history['val_accuracy'][9:], label='val_accuracy')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()

plt.figure(figsize=(16, 9))

plt.plot(range(epochs), hist.history['lr'][9:], label='leaning rate')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('leaning rate')

plt.show()



from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report



if one_hot:

    y1_test = np.argmax(y_test, axis=1)

    y1_pred = np.argmax(y_pred, axis=1)

else:

    y1_test = y_test

    y1_pred=np.zeros(shape=y_pred.shape)

    y1_pred[y_pred > 0.5] = 1



print(confusion_matrix(y1_test, y1_pred))

print(accuracy_score(y1_test,y1_pred))

print(f1_score(y1_test, y1_pred, average='weighted'))

print(classification_report(y1_test, y1_pred))
print(hist.history['loss'])

print(hist.history['val_loss'])

print(hist.history['accuracy'])

print(hist.history['val_accuracy'])

print(hist.history['lr'])
''' 

0 input_3

1 block1_conv1

2 block1_conv2

3 block1_pool

4 block2_conv1

5 block2_conv2

6 block2_pool

7 block3_conv1

8 block3_conv2

9 block3_conv3

10 block3_conv4

11 block3_pool

12 block4_conv1

13 block4_conv2

14 block4_conv3

15 block4_conv4

16 block4_pool

17 block5_conv1

18 block5_conv2

19 block5_conv3

20 block5_conv4

21 block5_pool

22 global_average_pooling2d_3

23 dense_5

24 dense_6

'''
'''

a = np.random.rand(10, 50, 50, 110, 80)

b = np.random.rand(10, 50, 50, 80, 1)

c = np.random.rand(10, 50, 50, 80, 1)

d = np.random.rand(10, 50, 50, 80, 110)

#c = np.dot(a,b)

#print(c.shape)

aa = K.variable(value=a, dtype='float64', name='example_a')

bb = K.variable(value=b, dtype='float64', name='example_b')

cc = K.variable(value=c, dtype='float64', name='example_c')

dd = K.variable(value=d, dtype='float64', name='example_d')

#ccc = aa*bb

f = tf.multiply(cc,dd) # =aa*bb, 对应元素相乘

e = tf.matmul(aa,bb)  # 

#cc = K.dot(aa,bb)

#cc = K.batch_dot(aa,bb, axes=[(1,1),(2,2)])

#cc = tf.tensordot(aa,bb, axes=[(0,1,2), (0,1,2)])

print('done')

print(e.shape)

print(f.shape)

'''
'''

def confusion_matrix(y, y_pred):

    dim = y.shape[1]

    cm = np.zeros(shape=(dim, dim))

    y = np.argmax(y, axis=1)

    y_pred = np.argmax(y_pred, axis=1)

    num = y_pred.shape[0]

    for i in range(num):

        cm[y[i]][y_pred[i]] += 1

    true_pred = 0

    for i in range(dim):

        true_pred += cm[i][i]

    return cm, true_pred, num



cm, true_pred, test_num = confusion_matrix(y_test, y_pred)

print(cm)

print(true_pred/test_num)

print('Done')

'''
'''

from PIL import Image

import numpy as np

import os





def load_dataset(fname=None):

    if fname is None:

        # Assume we are in the utils folder and get the absolute path to the

        # parent directory.

        fname = os.path.abspath(os.path.join(os.path.dirname(__file__),

                                             os.path.pardir))

        fname = os.path.join(fname, 'labels.csv')



    data = np.genfromtxt(fname, dtype=['|S19', '<f8', '|S4'], names=[

                         'path', 'probability', 'type'])

    image_fnames = np.char.decode(data['path'])

    probs = data['probability']

    types = np.char.decode(data['type'])



    def load_cell_image(fname):

        with Image.open(fname) as image:

            return np.asarray(image)



    dir = os.path.dirname(fname)



    images = np.array([load_cell_image(os.path.join(dir, fn))

                       for fn in image_fnames])



    return images, probs, types





images, proba, types = load_dataset()

'''