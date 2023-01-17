from __future__ import print_function

from keras import backend as K

from keras import activations

from keras import utils

from keras.models import Model

from keras.layers import *

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.optimizers import RMSprop, Adam, SGD, Nadam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from keras import regularizers





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



IMG_SIZE = 299
def DataGenerator(train_batch, val_batch, IMG_SIZE):

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                 rescale=1./255,

                                 rotation_range=10,

                                 horizontal_flip=True,

                                 vertical_flip=True)



    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)



    train_gen = datagen.flow_from_directory('../input/chest_xray/chest_xray/train/',

                                            target_size=(IMG_SIZE, IMG_SIZE),

                                            color_mode='rgb', 

                                            class_mode='categorical',

                                            batch_size=train_batch)



    val_gen = datagen.flow_from_directory('../input/chest_xray/chest_xray/val/', 

                                          target_size=(IMG_SIZE, IMG_SIZE),

                                          color_mode='rgb', 

                                          class_mode='categorical',

                                          batch_size=val_batch)



    datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                 rescale=1./255)

    

    datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32).reshape(1,1,3)



    test_gen = datagen.flow_from_directory('../input/chest_xray/chest_xray/test/', 

                                           target_size=(IMG_SIZE, IMG_SIZE),

                                           color_mode='rgb', 

                                           class_mode='categorical',

                                           shuffle=False)

    

    return train_gen, val_gen, test_gen
# the squashing function.

# we use 0.5 in stead of 1 in hinton's paper.

# if 1, the norm of vector will be zoomed out.

# if 0.5, the norm will be zoomed in while original norm is less than 0.5

# and be zoomed out while original norm is greater than 0.5.

def squash(x, axis=-1):

    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()

    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)

    return scale * x
# define our own softmax function instead of K.softmax

# because K.softmax can not specify axis.

def softmax(x, axis=-1):

    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))

    return ex / K.sum(ex, axis=axis, keepdims=True)
# define the margin loss like hinge loss

def margin_loss(y_true, y_pred):

    lamb, margin = 0.5, 0.1 #default lambda 0.5 - but test with lambda with 0.9 - 0.1

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

                 routings=3, # Test number of routing with (1, 2, 3, 4) - Default = 3

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
train_batch = 32

val_batch = 1



train, val, test = DataGenerator(train_batch, val_batch, IMG_SIZE)
input_image = Input(shape=(IMG_SIZE, IMG_SIZE, 3))





# A InceptionResNetV2 Conv2D model

base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_image)



base_model.summary()
for layer in base_model.layers:

    layer.trainable = False

    print(layer, layer.trainable)
x = GlobalAveragePooling2D()(base_model.output)

x = Dense(4096, activation='relu')(x)

x = Dropout(0.5)(x)

output = Dense(2, activation='softmax')(x)



model = Model(inputs=input_image, outputs=output)



model.summary()
lr=1e-4



model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
model.fit_generator(train,

                    epochs=1,

                    validation_data=val, 

                    validation_steps = len(val.classes)//val_batch,

                    steps_per_epoch=(len(train.classes)//train_batch) * 2) 

    

loss, acc = model.evaluate_generator(test, len(test))



print ("\n\n================================\n\n")

print ("Loss: {}".format(loss))

print ("Accuracy: {0:.2f} %".format(acc * 100))

print ("\n\n================================\n\n")



test.reset()
for i, layer in enumerate(model.layers):

    if i < 15:

        layer.trainable = False

    else:

        layer.trainable = True
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr, momentum=0.9), metrics=['accuracy'])
model.fit_generator(train,

                    epochs=10,

                    validation_data=val, 

                    validation_steps = len(val.classes)//val_batch,

                    steps_per_epoch=(len(train.classes)//train_batch) * 2) 

    

loss, acc = model.evaluate_generator(test, len(test))



print ("\n\n================================\n\n")

print ("Loss: {}".format(loss))

print ("Accuracy: {0:.2f} %".format(acc * 100))

print ("\n\n================================\n\n")



test.reset()
output = Conv2D(256, kernel_size=(9, 9), strides=(1, 1), activation='relu')(base_model.get_layer(name='block5_pool').output)



x = Reshape((-1, 256))(output)

capsule = Capsule(2, 16, 4, True)(x)

output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)

model = Model(inputs=input_image, outputs=output)



model.summary()
lr=1e-4



checkpoint = ModelCheckpoint("weights.h5", 

                             monitor='val_loss', 

                             verbose=1, 

                             save_best_only=True, 

                             save_weights_only=False, 

                             mode='min')



early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=True)



callback_list = [checkpoint, early]
epochs=100



model.compile(loss=margin_loss, optimizer=SGD(lr=lr, momentum=0.9), metrics=['accuracy'])



model.fit_generator(train,

                    epochs=epochs,

                    validation_data=val, 

                    validation_steps = len(val.classes)//val_batch,

                    steps_per_epoch=len(train.classes)//train_batch,

                    callbacks=callback_list)

    

loss, acc = model.evaluate_generator(test, len(test))



print ("\n\n================================\n\n")

print ("Loss: {}".format(loss))

print ("Accuracy: {0:.2f} %".format(acc * 100))

print ("\n\n================================\n\n")



test.reset()