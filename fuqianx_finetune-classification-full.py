from __future__ import division

from __future__ import print_function





from keras.layers import Dense, Flatten, Dropout

from keras.models import Model, Sequential

from keras.preprocessing import image

import keras.backend as K

from keras import optimizers, losses

from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array



from PIL import Image

import numpy as np

import random



def vgg16():

    vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))



    x = Dense(586, activation='softmax')(vgg.layers[-2].output)



    model = Model(input=vgg.input, output=x)

    model.summary()

    return model



def my_preprocessing_train(img):

    # random crop

    ndarray_convert_img = array_to_img(img)



    # resize shortest side to [256, 512] range

    width, height = ndarray_convert_img.size

    scale = random.randint(256, 512) / min(width, height)

    new_size = (int(np.ceil(scale * width)), int(np.ceil(scale * height)))

    img_resized = ndarray_convert_img.resize(new_size)



    # crop (224, 224)

    crop_width_start = random.randint(0, img_resized.size[0] - 224)

    crop_height_start = random.randint(0, img_resized.size[1] - 224)

    box = (crop_width_start, crop_height_start, crop_width_start + 224, crop_height_start + 224)

    img_cropped = img_resized.crop(box)



    # convert to BGR

    img_convert_ndarray = np.array(img_cropped, dtype='float64')

    img_convert_ndarray = img_convert_ndarray[:,:,::-1]



    # subtract the BGR mean

    img_convert_ndarray[...,0] -= 103.939

    img_convert_ndarray[...,1] -= 116.779

    img_convert_ndarray[...,2] -= 123.68



    return img_convert_ndarray



def my_preprocessing_test(img):



    img.astype('float64')



    # convert to BGR

    img = img[:,:,::-1]



    # subtract the BGR mean

    img[...,0] -= 103.939

    img[...,1] -= 116.779

    img[...,2] -= 123.68



    return img
train_dir = "../input/landmark/landmark/full/train"

test_dir = "../input/landmark/landmark/full/val"



model = vgg16()



model.load_weights("../input/finetune-c-full/finetune_classification_full.h5")



train_datagen = ImageDataGenerator(rotation_range=10,

                                   preprocessing_function=my_preprocessing_train)



test_datagen = ImageDataGenerator(preprocessing_function=my_preprocessing_test)



train_generator = train_datagen.flow_from_directory(directory=train_dir,

                                                    target_size=(224, 224),

                                                    batch_size=128,

                                                    class_mode='sparse')



test_generator = test_datagen.flow_from_directory(directory=test_dir,

                                                         target_size=(224, 224),

                                                         batch_size=128,

                                                         class_mode='sparse')



model.compile(optimizer=optimizers.SGD(lr=0.0005),

              loss=losses.sparse_categorical_crossentropy,

              metrics=['acc'])





history = model.fit_generator(generator=train_generator,

                              epochs=4,

                              validation_data=test_generator)



model.save_weights('/kaggle/working/finetune_classification_full.h5')