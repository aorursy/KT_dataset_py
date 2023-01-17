from functools import reduce

from keras import backend as K

from keras.layers import (Activation, Add, GlobalAveragePooling2D,

                          BatchNormalization, Conv2D, Dense, Flatten, Input,

                          MaxPooling2D)

from keras.models import Model

from keras.regularizers import l2



def compose(*funcs):

    if funcs:

        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)

    else:

        raise ValueError('Composition of empty sequence not supported.')



def ResNetConv2D(*args, **kwargs):

    conv_kwargs = {

        'strides': (1, 1),

        'padding': 'same',

        'kernel_initializer': 'he_normal',

        'kernel_regularizer': l2(1.e-4)

    }

    conv_kwargs.update(kwargs)



    return Conv2D(*args, **conv_kwargs)



def bn_relu_conv(*args, **kwargs):

    return compose(

        BatchNormalization(),

        Activation('relu'),

        ResNetConv2D(*args, **kwargs))



def shortcut(x, residual):

    x_shape = K.int_shape(x)

    residual_shape = K.int_shape(residual)



    if x_shape == residual_shape:

        shortcut = x

    else:

        stride_w = int(round(x_shape[1] / residual_shape[1]))

        stride_h = int(round(x_shape[2] / residual_shape[2]))



        shortcut = Conv2D(filters=residual_shape[3],

                          kernel_size=(1, 1),

                          strides=(stride_w, stride_h),

                          kernel_initializer='he_normal',

                          kernel_regularizer=l2(1.e-4))(x)

    return Add()([shortcut, residual])



def basic_block(filters, first_strides, is_first_block_of_first_layer):

    def f(x):

        if is_first_block_of_first_layer:

            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)

        else:

            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3),

                                 strides=first_strides)(x)



        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)



        return shortcut(x, conv2)



    return f



def bottleneck_block(filters, first_strides, is_first_block_of_first_layer):

    def f(x):

        if is_first_block_of_first_layer:

            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)

        else:

            conv1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),

                                 strides=first_strides)(x)



        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)



        return shortcut(x, conv3)



    return f



def residual_blocks(block_function, filters, repetitions, is_first_layer):

    def f(x):

        for i in range(repetitions):

            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)



            x = block_function(filters=filters, first_strides=first_strides,

                               is_first_block_of_first_layer=(i == 0 and is_first_layer))(x)

        return x



    return f



class ResnetBuilder():

    @staticmethod

    def build(input_shape, num_outputs, block_type, repetitions):

        if block_type == 'basic':

            block_fn = basic_block

        elif block_type == 'bottleneck':

            block_fn = bottleneck_block



        input = Input(shape=input_shape)



        conv1 = compose(ResNetConv2D(filters=64, kernel_size=(7, 7), strides=(2, 2)),

                        BatchNormalization(),

                        Activation('relu'))(input)



        pool1 = MaxPooling2D(

            pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)



        block = pool1

        filters = 64

        for i, r in enumerate(repetitions):

            block = residual_blocks(block_fn, filters=filters, repetitions=r,

                                    is_first_layer=(i == 0))(block)

            filters *= 2



        block = compose(BatchNormalization(),

                        Activation('relu'))(block)



        pool2 = GlobalAveragePooling2D()(block)



        fc1 = Dense(units=num_outputs,

                    kernel_initializer='he_normal',

                    activation='softmax')(pool2)



        return Model(inputs=input, outputs=fc1)



    @staticmethod

    def build_resnet_18(input_shape, num_outputs):

        return ResnetBuilder.build(

            input_shape, num_outputs, 'basic', [2, 2, 2, 2])



    @staticmethod

    def build_resnet_34(input_shape, num_outputs):

        return ResnetBuilder.build(

            input_shape, num_outputs, 'basic', [3, 4, 6, 3])



    @staticmethod

    def build_resnet_50(input_shape, num_outputs):

        return ResnetBuilder.build(

            input_shape, num_outputs, 'bottleneck', [3, 4, 6, 3])



    @staticmethod

    def build_resnet_101(input_shape, num_outputs):

        return ResnetBuilder.build(

            input_shape, num_outputs, 'bottleneck', [3, 4, 23, 3])



    @staticmethod

    def build_resnet_152(input_shape, num_outputs):

        return ResnetBuilder.build(

            input_shape, num_outputs, 'bottleneck', [3, 8, 36, 3])
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs

(there is *a lot* of margin for parameter tuning).

2 seconds per epoch on a K520 GPU.

'''



import keras

from keras.datasets import mnist

# from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam



batch_size = 128

num_classes = 10

epochs = 20



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()



x_train = x_train.reshape(60000, 28, 28, 1)

x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



# ResNet sample

# RMSprop

# resnet34 -> acc: 98.60

# resnet18 -> acc: 97.43

# Adam

# resnet18 -> acc: 97.95

model = ResnetBuilder.build_resnet_18((28, 28, 1), num_classes)

# MLP sample

# RMSprop

#   -> acc: 98.31

# model = Sequential()

# model.add(Dense(512, activation='relu', input_shape=(784,)))

# model.add(Dropout(0.2))

# model.add(Dense(512, activation='relu'))

# model.add(Dropout(0.2))

# model.add(Dense(num_classes, activation='softmax'))



model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer=Adam(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])