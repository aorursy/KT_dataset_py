from __future__ import division

from __future__ import print_function



from keras.layers import Dense, Flatten, Dropout

from keras.models import Model, Sequential

from keras.preprocessing import image

import keras.backend as K

from keras import optimizers, losses

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator

from keras.engine.topology import Layer



from PIL import Image

import numpy as np

import random



class SpatialPyramidPooling(Layer):

    def __init__(self, pool_list):

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__()



    def build(self, input_shape):

        self.nb_channels = input_shape[3]



    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)



    def call(self, x):



        input_shape = K.shape(x)



        num_rows = input_shape[1]

        num_cols = input_shape[2]



        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]

        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]



        outputs = []



        for pool_num, num_pool_regions in enumerate(self.pool_list):

            for jy in range(num_pool_regions):

                for ix in range(num_pool_regions):

                    x1 = ix * col_length[pool_num]

                    x2 = ix * col_length[pool_num] + col_length[pool_num]

                    y1 = jy * row_length[pool_num]

                    y2 = jy * row_length[pool_num] + row_length[pool_num]



                    x1 = K.cast(K.round(x1), 'int32')

                    x2 = K.cast(K.round(x2), 'int32')

                    y1 = K.cast(K.round(y1), 'int32')

                    y2 = K.cast(K.round(y2), 'int32')



                    new_shape = [input_shape[0], y2 - y1,

                                     x2 - x1, input_shape[3]]



                    x_crop = x[:, y1:y2, x1:x2, :]

                    xm = K.reshape(x_crop, new_shape)

                    pooled_val = K.max(xm, axis=(1, 2))

                    outputs.append(pooled_val)



        outputs = K.concatenate(outputs)



        return outputs
def vgg16():

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(680, 530, 3))



    base_model = Model(input=vgg.input, output=vgg.layers[-2].output)

    model = Sequential()

    model.add(base_model)

    model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(256, activation='softmax'))

    

    return model
train_dir = "../input/handcrafted-finetune-full-image/handcrafted_finetune_full_image/train"

test_dir = "../input/handcrafted-finetune-full-image/handcrafted_finetune_full_image/test"



model = vgg16()



model.load_weights("../input/finetune-classification-resize-16/finetune_classification_resize_16.h5")



train_datagen = ImageDataGenerator(rescale=1./255)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(directory=train_dir,

                                                    target_size=(680, 530),

                                                    batch_size=16,

                                                    class_mode='sparse')



test_generator = test_datagen.flow_from_directory(directory=test_dir,

                                                  target_size=(680, 530),

                                                  batch_size=16,

                                                  class_mode='sparse')



optimizer = optimizers.SGD(lr=0.0001)

model.compile(optimizer=optimizer,

              loss=losses.sparse_categorical_crossentropy,

              metrics=['acc'])





model.fit_generator(generator=train_generator,

                              epochs=2,

                              validation_data=test_generator)



model.save_weights('/kaggle/working/finetune_classification_resize_16.h5')