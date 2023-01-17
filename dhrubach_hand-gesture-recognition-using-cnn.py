# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rn

import cv2

import datetime

import os

import shutil



import tensorflow as tf



from matplotlib import pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%config IPCompleter.greedy = True
import warnings

warnings.filterwarnings('ignore')
# np.random.seed(50)

# rn.seed(30)



tf.set_random_seed(30)
train_path = '../input/gesturevideos/data/train/'

val_path = '../input/gesturevideos/data/val/'



train_doc = np.random.permutation(open('../input/gesturevideos/data/train.csv').readlines())

val_doc = np.random.permutation(open('../input/gesturevideos/data/val.csv').readlines())
num_train_sequences = len(train_doc)

num_val_sequences = len(val_doc)



print('total number of training sequences - {0}'.format(num_train_sequences))

print('total number of validation sequences - {0}'.format(num_val_sequences))
def plot_image(images, cmap=None):

    total_images = len(images)

    rows = 0



    if total_images < 6:

        rows = 1

    elif total_images % 5 == 0:

        rows = total_images / 5

    else:

        rows = (total_images // 5) + 1



    f, axes = plt.subplots(int(rows),

                           (5 if total_images > 5 else total_images),

                           sharex=True,

                           sharey=True)

    f.set_figwidth(15)

    f.set_figheight(5)

    row_elements = 5



    if rows == 1:

        for ax, image in zip(axes, images):

            ax.imshow(image, cmap)

    else:

        for i, row_ax in enumerate(axes):

            start_index = i * row_elements



            for ax, image in zip(row_ax, images[start_index:start_index + 5]):

                ax.imshow(image, cmap)
random_train_index = rn.randint(0, len(train_doc))



sequence_dir = train_doc[random_train_index].split(';')[0]



# list of all images in the folder

image_names = os.listdir(train_path + sequence_dir)



images = []

for name in image_names[0:10:3]:

    img = cv2.imread(train_path + sequence_dir + '/' + name)

    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



print('shape of an image in the selected sequence : {0}'.format(images[0].shape))    

    

# plot_image(images)
class ImagePreprocessor:

    def __init__(self,

                 height=120,

                 width=120,

                 channel=3,

                 affine=False,

                 crop=False,

                 edge=False,

                 normalize=False,

                 mnormalize=False,

                 resize=False,

                 smoothing=False):

        self.__height = height

        self.__width = width

        self.__channel = channel

        self._affine = affine

        self._crop = crop

        self._edge = edge

        self._normalize = normalize

        self._normalize_m = mnormalize

        self._resize = resize

        self._smoothing = smoothing



    def process_image(self, orig_image):

        pr_image = orig_image



        if self._crop:

            pr_image = self.__crop(pr_image)



        if self._resize:

            pr_image = self.__resize(pr_image)



        if self._normalize:

            pr_image = self.__normalize(pr_image)



        if self._normalize_m:

            pr_image = self.__normalize_mean(pr_image)

            

        if self._affine:

            pr_image = self.__affine(pr_image)

        

        if self._smoothing:

            pr_image = self.__smoothing(pr_image)



        if self._edge:

            pr_image = self.__edge_detection(pr_image)



        return pr_image



    def __crop(self, image):

        i_shape = image.shape

        return image[20:i_shape[0], 0:i_shape[1]]



    def __normalize(self, image):

        n_image = np.zeros((self.__height, self.__width, self.__channel))

        n_image = cv2.normalize(image, n_image, 0, 255, cv2.NORM_MINMAX)



        return n_image

    

    def __normalize_mean(self, image):

        n_image = np.zeros((self.__height, self.__width, self.__channel))

        n_image = cv2.normalize(image, n_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        

        return n_image



    def __resize(self, image):

        return cv2.resize(image, (self.__height, self.__width),

                          interpolation=cv2.INTER_AREA)



    def __smoothing(self, image):

        return cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)



    def __edge_detection(self, image):

        return cv2.Canny(image, 100, 150)

    

    def __affine(self, image):

        dx, dy = np.random.randint(-1.7, 1.8, 2)

        M = np.float32([[1, 0, dx], [0, 1, dy]])



        return cv2.warpAffine(image, M, (image.shape[0], image.shape[1]))
imagePreprocess = ImagePreprocessor(crop=True, mnormalize=True, resize=True, smoothing=True)



pr_images = []



for name in image_names[0:10:3]:

    img = cv2.imread(train_path + sequence_dir + '/' + name)

    pr_image = imagePreprocess.process_image(orig_image=img)

    pr_images.append(cv2.cvtColor(pr_image, cv2.COLOR_BGR2RGB))



print('shape of an image in the selected sequence : {0}'.format(pr_images[0].shape))    

    

# plot_image(pr_images)
class ModelParameters:

    def __init__(self,

                 im_height=120,

                 im_width=120,

                 filters=[8, 16, 32, 64],

                 dense=[5000, 500, 5],

                 epochs=20,

                 batch_size=10,

                 lrate=0.001):

        self.__height = im_height

        self.__width = im_width

        self.__channels = 3



        self.FrameIndexes = 'ALL'

        self.Filters = filters

        self.DenseLayers = dense



        self.Epochs = epochs

        self.BatchSize = batch_size

        self.LRate = lrate



    @property

    def Filters(self):

        return self.__filters



    @Filters.setter

    def Filters(self, val):

        self.__filters = val



    @property

    def DenseLayers(self):

        return self.__dense



    @DenseLayers.setter

    def DenseLayers(self, val):

        self.__dense = val



    @property

    def FrameIndexes(self):

        return self.__frames



    @FrameIndexes.setter

    def FrameIndexes(self, val):

        if val == 'ALL':

            self.__frames = [x for x in range(0, 30)]

        else:

            self.__frames = [x for x in range(0, 30, 3)]

            self.__frames.append(30)



    @property

    def Epochs(self):

        return self.__epochs



    @Epochs.setter

    def Epochs(self, val):

        self.__epochs = val



    @property

    def BatchSize(self):

        return self.__bsize



    @BatchSize.setter

    def BatchSize(self, val):

        self.__bsize = val



    @property

    def LRate(self):

        return self.__lrate



    @LRate.setter

    def LRate(self, val):

        self.__lrate = val



    def set_image_dimensions(self, height, width):

        self.__height = height

        self.__width = width



    def input_shape(self):

        return (len(self.FrameIndexes), self.__height, self.__width, self.__channels)



    def optimizer(self, optimizers, otype='SGD'):

        if otype == 'ADAM':

            return optimizers.Adam(self.LRate)

        elif otype == 'ADELTA':

            return optimizers.Adadelta()

        else:

            return optimizers.SGD(lr=self.LRate, momentum=0.9, nesterov=True)



    def steps_per_epoch(self, num_train_sequences):

        if (num_train_sequences % self.BatchSize) == 0:

            return int(num_train_sequences / self.BatchSize)

        else:

            return (num_train_sequences // self.BatchSize) + 1



    def validation_steps(self, num_val_sequences):

        if (num_val_sequences % self.BatchSize) == 0:

            return int(num_val_sequences / self.BatchSize)

        else:

            return (num_val_sequences // self.BatchSize) + 1
def generator(source_path, folder_list, ipreprocessor, mparameters):



    img_idx = mparameters.FrameIndexes

    input_shape = mparameters.input_shape()

    batch_size = mparameters.BatchSize

    

    while True:

        # shuffle the array of video sequences prior to creating individual batches

        t = np.random.permutation(folder_list)



        num_batches = len(t) // batch_size



        for batch in range(num_batches):



            # x is the number of images you use for each video,

            # (y,z) is the final size of the input images

            # 3 is the number of channels RGB

            batch_data = np.zeros((batch_size, len(img_idx), input_shape[1],

                                   input_shape[2], input_shape[3]))

            

            # batch_labels is the one hot representation of the output

            batch_labels = np.zeros((batch_size, 5))



            for folder in range(batch_size):



                fl_index = folder + (batch * batch_size)

                fl_name = t[fl_index].strip().split(';')[0]



                # read all the images in the folder

                imgs = os.listdir(source_path + '/' + fl_name)



                # iterate iver the frames/images of a folder to read them in

                for idx, item in enumerate(img_idx):



                    image = cv2.imread(source_path + '/' + fl_name + '/' + imgs[item])



                    pr_image = ipreprocessor.process_image(orig_image=image)

                    

                    batch_data[folder, idx, :, :, 0] = pr_image[:, :, 0]

                    batch_data[folder, idx, :, :, 1] = pr_image[:, :, 1]

                    batch_data[folder, idx, :, :, 2] = pr_image[:, :, 2]



                batch_labels[folder, int(t[fl_index].strip().split(';')[2])] = 1



            yield batch_data, batch_labels



        # write the code for the remaining data points which are left after full batches

        if (len(folder_list) != batch_size * num_batches):



            batch_size = len(folder_list) - (batch_size * num_batches)

            batch_data = np.zeros((batch_size, len(img_idx), input_shape[1],

                                   input_shape[2], input_shape[3]))

            batch_labels = np.zeros((batch_size, 5))



            for folder in range(batch_size):



                fl_index = folder + (batch * batch_size)

                fl_name = t[fl_index].strip().split(';')[0]



                # read all the images in the folder

                imgs = os.listdir(source_path + '/' + fl_name)



                for idx, item in enumerate(img_idx):



                    image = cv2.imread(source_path + '/' + fl_name + '/' + imgs[item])



                    pr_image = ipreprocessor.process_image(orig_image=image)



                    batch_data[folder, idx, :, :, 0] = pr_image[:, :, 0]

                    batch_data[folder, idx, :, :, 1] = pr_image[:, :, 1]

                    batch_data[folder, idx, :, :, 2] = pr_image[:, :, 2]



                batch_labels[folder, int(t[fl_index].strip().split(';')[2])] = 1



            yield batch_data, batch_labels
from keras.models import Sequential, Model

from keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout

from keras.layers.convolutional import Conv3D, MaxPooling3D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import Input, optimizers
curr_dt_time = datetime.datetime.now()

model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ', '').replace(':', '_') + '/'



if not os.path.exists(model_name):

    os.mkdir(model_name)



filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'
checkpoint = ModelCheckpoint(filepath,

                             monitor='val_loss',

                             verbose=1,

                             save_best_only=True,

                             save_weights_only=False,

                             mode='auto',

                             period=1)



LR = ReduceLROnPlateau(monitor='val_loss',

                       factor=0.5,

                       patience=2,

                       cooldown=1,

                       verbose=1)



callbacks_list = [checkpoint, LR]
def con3d_default_model(input_shape, nb_filters, nb_dense):



    model = Sequential()



    # layer 1

    model.add(

        Conv3D(nb_filters[0],

               kernel_size=(3, 3, 3),

               input_shape=input_shape,

               padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    # layer 2

    model.add(Conv3D(nb_filters[1], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    # layer 3

    model.add(Conv3D(nb_filters[2], kernel_size=(1, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    # layer 4

    model.add(Conv3D(nb_filters[3], kernel_size=(1, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    #Flatten Layers

    model.add(Flatten())



    # dense layer 1

    model.add(Dense(nb_dense[0], activation='relu'))

    model.add(Dropout(rate=0.5))



    # dense layer 2

    model.add(Dense(nb_dense[1], activation='relu'))

    model.add(Dropout(rate=0.5))



    #softmax layer

    model.add(Dense(nb_dense[2], activation='softmax'))



    return model
imagePreprocessor = ImagePreprocessor(crop=True,

                                      normalize=True,

                                      resize=True,

                                      smoothing=True)



modelParameters = ModelParameters()



train_generator = generator(train_path, train_doc, imagePreprocessor, modelParameters)

val_generator = generator(val_path, val_doc, imagePreprocessor, modelParameters)
model_a = con3d_default_model(modelParameters.input_shape(),

                            modelParameters.Filters,

                            modelParameters.DenseLayers)



model_a.compile(optimizer=modelParameters.optimizer(optimizers),

              loss='categorical_crossentropy',

              metrics=['categorical_accuracy'])



print(model_a.summary())
# model_a.fit_generator(train_generator,

#                       steps_per_epoch=modelParameters.steps_per_epoch(num_train_sequences),

#                       epochs=modelParameters.Epochs,

#                       verbose=1,

#                       callbacks=callbacks_list,

#                       validation_data=val_generator,

#                       validation_steps=modelParameters.validation_steps(num_val_sequences),

#                       class_weight=None,

#                       workers=1,

#                       initial_epoch=0)
def con3d_model_b(input_shape, nb_filters, nb_dense):

    model = Sequential()

    

    # layer 1

    model.add(Conv3D(nb_filters[0], 

                     kernel_size=(3,3,3), 

                     input_shape=input_shape,

                     padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2,2,2)))

    

    # layer 2

    model.add(Conv3D(nb_filters[1], 

                     kernel_size=(3,3,3), 

                     padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2,2,2)))

    

    # layer 3

    model.add(Conv3D(nb_filters[2], 

                     kernel_size=(1,3,3), 

                     padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2,2,2)))

    

    # layer 4

    model.add(Conv3D(nb_filters[3], 

                     kernel_size=(1,3,3), 

                     padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    model.add(MaxPooling3D(pool_size=(2,2,2)))



    #Flatten Layers

    model.add(Flatten())

    

    # dense layer 1

    model.add(Dense(nb_dense[0], activation='relu'))

    model.add(Dropout(0.25))

    

    # dense layer 2

    model.add(Dense(nb_dense[1], activation='relu'))

    model.add(Dropout(0.25))



    #softmax layer

    model.add(Dense(nb_dense[2], activation='softmax'))

    

    return model
imagePreprocessor = ImagePreprocessor(crop=True,

                                      normalize=True,

                                      resize=True,

                                      smoothing=True)



modelParameters = ModelParameters()



train_generator = generator(train_path, train_doc, imagePreprocessor, modelParameters)

val_generator = generator(val_path, val_doc, imagePreprocessor, modelParameters)
model_b = con3d_model_b(modelParameters.input_shape(),

                            modelParameters.Filters,

                            modelParameters.DenseLayers)



model_b.compile(optimizer=modelParameters.optimizer(optimizers),

              loss='categorical_crossentropy',

              metrics=['categorical_accuracy'])



print(model_b.summary())
# model_b.fit_generator(train_generator,

#                       steps_per_epoch=modelParameters.steps_per_epoch(num_train_sequences),

#                       epochs=modelParameters.Epochs,

#                       verbose=1,

#                       callbacks=callbacks_list,

#                       validation_data=val_generator,

#                       validation_steps=modelParameters.validation_steps(num_val_sequences),

#                       class_weight=None,

#                       workers=1,

#                       initial_epoch=0)
def con3d_model_2(input_shape, nb_filters, nb_dense):



    model = Sequential()



    # layer 1 - 8 kernels

    model.add(

        Conv3D(nb_filters[0],

               kernel_size=(3, 3, 3),

               input_shape=input_shape,

               padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    

    # pooling layer 1

    model.add(MaxPooling3D(pool_size=(1, 2, 2)))



    # layer 2 - 16 kernels

    model.add(Conv3D(nb_filters[1], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # pooling layer 2

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    # layer 3 - 32 kernels

    model.add(Conv3D(nb_filters[2], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # layer 4 - 32 kernels

    model.add(Conv3D(nb_filters[2], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # pooling layer 3

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    # layer 5 - 64 kernels

    model.add(Conv3D(nb_filters[3], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # layer 6 - 64 kernels

    model.add(Conv3D(nb_filters[3], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # pooling layer 4

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))



    # layer 7 - 64 kernels

    model.add(Conv3D(nb_filters[3], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # layer 8 - 64 kernels

    model.add(Conv3D(nb_filters[3], kernel_size=(3, 3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))



    # pooling layer 5

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    

    #Flatten Layers

    model.add(Flatten())



    # dense layer 1

    model.add(Dense(nb_dense[0], activation='relu'))



    # dense layer 2

    model.add(Dense(nb_dense[1], activation='relu'))



    #softmax layer

    model.add(Dense(nb_dense[2], activation='softmax'))



    return model
imagePreprocessor = ImagePreprocessor(crop=True,

                                      normalize=True,

                                      resize=True,

                                      smoothing=False)



modelParameters = ModelParameters(filters=[16, 32, 64, 128], dense = [4096, 4096, 5],batch_size = 20)

modelParameters.Epochs = 30



train_generator = generator(train_path, train_doc, imagePreprocessor, modelParameters)

val_generator = generator(val_path, val_doc, imagePreprocessor, modelParameters)
model_c = con3d_model_2(modelParameters.input_shape(),

                            modelParameters.Filters,

                            modelParameters.DenseLayers)



model_c.compile(optimizer=modelParameters.optimizer(optimizers),

              loss='categorical_crossentropy',

              metrics=['categorical_accuracy'])



print(model_c.summary())
# model_c.fit_generator(train_generator,

#                       steps_per_epoch=modelParameters.steps_per_epoch(num_train_sequences),

#                       epochs=modelParameters.Epochs,

#                       verbose=1,

#                       callbacks=callbacks_list,

#                       validation_data=val_generator,

#                       validation_steps=modelParameters.validation_steps(num_val_sequences),

#                       class_weight=None,

#                       workers=1,

#                       initial_epoch=0)
imagePreprocessor = ImagePreprocessor(crop=True,

                                      normalize=True,

                                      resize=True,

                                      smoothing=False)



modelParameters = ModelParameters(filters=[32, 64, 128, 256], dense = [4096, 4096, 5], batch_size = 10)

modelParameters.Epochs = 20



train_generator = generator(train_path, train_doc, imagePreprocessor, modelParameters)

val_generator = generator(val_path, val_doc, imagePreprocessor, modelParameters)
model_e = con3d_model_2(modelParameters.input_shape(),

                            modelParameters.Filters,

                            modelParameters.DenseLayers)



model_e.compile(optimizer=modelParameters.optimizer(optimizers),

              loss='categorical_crossentropy',

              metrics=['categorical_accuracy'])



print(model_e.summary())
# model_e.fit_generator(train_generator,

#                       steps_per_epoch=modelParameters.steps_per_epoch(num_train_sequences),

#                       epochs=modelParameters.Epochs,

#                       verbose=1,

#                       callbacks=callbacks_list,

#                       validation_data=val_generator,

#                       validation_steps=modelParameters.validation_steps(num_val_sequences),

#                       class_weight=None,

#                       workers=1,

#                       initial_epoch=0)
def conv3d_model_3(input_shape, nb_filters, nb_dense):

    model = Sequential()



    # layer 1 - 32 kernels

    model.add(

        Conv3D(nb_filters[0],

               kernel_size=(3, 3, 1),

               strides=(2, 2, 1),

               padding='same',

               activation='relu',

               input_shape=input_shape))

    model.add(BatchNormalization())



    # pooling layer 1

    model.add(MaxPooling3D(pool_size=(2, 2, 1)))



    # layer 2 - 32 kernels

    model.add(

        Conv3D(nb_filters[0],

               kernel_size=(3, 3, 1),

               padding='same',

               activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 1)))



    # layer 3 - 64 kernels

    model.add(

        Conv3D(nb_filters[1],

               kernel_size=(3, 3, 1),

               padding='same',

               activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 1)))



    # layer 4 - 128 kernels

    model.add(

        Conv3D(nb_filters[2],

               kernel_size=(2, 2, 1),

               padding='same',

               activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(1, 1, 1)))



    # flatten layer

    model.add(Flatten())

    model.add(Dropout(0.25))



    # dense layer 1 - 128

    model.add(Dense(nb_dense[0], activation='relu'))

    model.add(Dropout(0.25))



    # softmax layer

    model.add(Dense(nb_dense[1], activation='softmax'))



    return model
imagePreprocessor = ImagePreprocessor(crop=False,

                                      mnormalize=True,

                                      resize=True,

                                      smoothing=False)



modelParameters = ModelParameters(filters=[32, 64, 128], dense = [128, 5],batch_size = 20)

modelParameters.Epochs = 20



train_generator = generator(train_path, train_doc, imagePreprocessor, modelParameters)

val_generator = generator(val_path, val_doc, imagePreprocessor, modelParameters)
model_f = conv3d_model_3(modelParameters.input_shape(),

                            modelParameters.Filters,

                            modelParameters.DenseLayers)



model_f.compile(optimizer=modelParameters.optimizer(optimizers, otype = 'ADELTA'),

              loss='categorical_crossentropy',

              metrics=['categorical_accuracy'])



print(model_f.summary())
model_f.fit_generator(train_generator,

                      steps_per_epoch=modelParameters.steps_per_epoch(num_train_sequences),

                      epochs=modelParameters.Epochs,

                      verbose=1,

                      callbacks=callbacks_list,

                      validation_data=val_generator,

                      validation_steps=modelParameters.validation_steps(num_val_sequences),

                      class_weight=None,

                      workers=1,

                      initial_epoch=0)