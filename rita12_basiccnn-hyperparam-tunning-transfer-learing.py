import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img

import os

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

import random
# example of loading an image with the Keras API

#from keras.preprocessing.image import load_img

# load the image

img = load_img('/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/cat.1.jpg')

# report details about the image

print(type(img))

print(img.format)

print(img.mode)

print(img.size)

# show the image

# img.show()

plt.imshow(img)
# example of converting an image with the Keras API

#from keras.preprocessing.image import img_to_array

#from keras.preprocessing.image import array_to_img



# convert to numpy array

img_array = img_to_array(img)

# print(img_array)

print(img_array.dtype)

print(img_array.shape)

# convert back to image

# img_pil = array_to_img(img_array)

print(type(img))



# the image's H:W 280:300, with RBG - 3 color channel
filenames = os.listdir("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/")

# filenames

# clearly the target labels are embeded in the filename, Let's parse them out
dog_filenames = []

cat_filenames = []

for i in filenames:

    if i.split('.')[0] == 'dog':

        dog_filenames.append(i)

    else:

        cat_filenames.append(i)



# At first I was using 0,1 to label target, but it throw error when using imagedatagenerator.flow_from_dataframe()

# function -- obviously if you are using 'binary' as class_mode, the target label data type has to be string

# Don't know why

dog_df = pd.DataFrame({'filename':dog_filenames, 'label':'dog'})

cat_df = pd.DataFrame({'filename':cat_filenames, 'label':'cat'})

all_df = cat_df.append(dog_df)

#train_data_array= train_df['filename'].apply(lambda x: load_img_array('dogs-vs-cats/train/'+ x))

#train_label_array = train_df['label'].values

all_df.head()
all_df.label.value_counts()

# perfectly balanced dataset
# first Let's hold out 20% of data to be our testing set

# the images in '~/test1/' folder do not have label, we will only be apply best performance model on test dataset

# to get optimal submission score

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(all_df, test_size=0.2, random_state = 42)
test_df.shape

train_df.shape
image_size = 224 # the input shape for VGG net has to be 224 * 224, so here I set it up this way for later adoption on VGG 



train_datagen=ImageDataGenerator(

    rescale=1./255., # scaling

    rotation_range = 15, shear_range = 0.1, zoom_range = 0.1, 

    horizontal_flip = True, width_shift_range=0.1,height_shift_range=0.1) # give more variants to the training dataset



train_generator=train_datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/", # dir path which contains all the training dataset

    x_col="filename", # the col that contains the file name

    y_col="label", # the col that shows the class label

    batch_size=32, # ImageDataGenerator is also a batch loading data function

    seed=42, # for reproductivbility

    shuffle=True,

    class_mode="binary", 

    target_size=(image_size,image_size)) # The dimensions to which all images found will be resized. in this case I am just using 150
# check out the training data shape for each batch

for data_batch, labels_batch in train_generator:

    print('data batch shape: ', data_batch.shape)

    print('labels batch shape: ', labels_batch.shape)

    break
# also build a test data generator for later use.

test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_dataframe(

        dataframe=test_df,

        directory='/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/',

        x_col="filename",

        y_col="label",

        target_size=(image_size, image_size),

        batch_size=32,

        class_mode='binary')
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout

from keras import optimizers

input_shape = (image_size, image_size, 3)
# build model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                 activation='relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (3,3), activation='relu'))

model.add(MaxPooling2D((2,2)))



model.add(Flatten())

model.add(Dense(1000, activation='relu'))

model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizers.rmsprop(lr=1e-3, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])
# fit the model

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size # number of training instances // each batch size

# in our case it is 25000 // 32 = 781

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN, 

                    # Total number of steps (batches of samples) to yield from generator 

                    # before declaring one epoch finished and starting the next epoch.

                    validation_data=test_generator,

                    validation_steps=STEP_SIZE_TEST,

                    epochs=3

)
model.save("./model1_simpleCNN.hdf5")

# we only got accuracy of 0.7175 not great, but we have a base line model
# just to download the output model file from kaggle to laptop

from IPython.display import FileLink

FileLink(r'model1_simpleCNN.hdf5')
import keras

def create_CNN(activation = 'relu', dropout_flag = False, batchnorm_flag = False, 

               dropout_rate = 0.5, learning_rate = 1e-4, optimizer = 'rmsprop'):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                     activation = activation,

                     input_shape=input_shape))

    if batchnorm_flag:

        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if dropout_flag:

        model.add(Dropout(dropout_rate))



    model.add(Conv2D(64, (3, 3), activation=activation))

    if batchnorm_flag:

        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    if dropout_flag:

        model.add(Dropout(dropout_rate))



    model.add(Conv2D(128, (3,3), activation=activation))

    if batchnorm_flag:

        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    if dropout_flag:

        model.add(Dropout(dropout_rate))



    model.add(Flatten())

    model.add(Dense(1000, activation=activation))

    model.add(Dense(1, activation='sigmoid')) 

    if optimizer == 'sgd':

        optimize_instance = keras.optimizers.SGD(lr = learning_rate)

    elif optimizer == 'adam':

        optimize_instance = keras.optimizers.Adam(learning_rate = learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    elif optimizer == 'rmsprop':

        optimize_instance = keras.optimizers.RMSprop(learning_rate= learning_rate, rho=0.9, decay = 1e-6)

    elif optimizer == 'adagrad':

        optimize_instance = keras.optimizers.Adagrad(learning_rate= learning_rate)

    else:

        print('Not a valid optimizer') 

    model.compile(optimize_instance,loss="binary_crossentropy",metrics=["accuracy"])

    return model
activations = ['relu', 'tanh', 'sigmoid', 'linear']

optimizers = ['adam', 'rmsprop', 'adagrad']

# I cross out SGD optimizer Because the accuracy over 3 epochs are around 57%, barely smarter than a ramdom guess..

# 

learning_rates = [1e-5, 1e-4,1e-3]

dropoutrates = [0.5, 0.25]

initializers = ['glorot_uniform','glorot_normal','he_normal', 'he_uniform']
def sample_params(param_grid, n): # n = how many sample combination you want

    l = []

    num = 0

    for i in range(n):

        d = {}

        for k, v in param_grid.items():

            d[k] = random.sample(v,1)

        l.append(d)

    return l
param_grid = dict(

#            activation = activations, 

            optimizer = optimizers, 

            learning_rate = learning_rates,

            batchnorm_flag = [True],

            dropout_flag = [True],

#            dropout_rate = dropoutrates

            

)

import itertools as it

param_combinations = [dict(zip(param_grid,v)) for v in it.product(*param_grid.values())]

#len(param_combinations) # 9 combinations
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size # number of training instances // each batch size

# in our case it is 25000 // 32 = 781

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



# iterate over each combination of hyperparameters

count = 1

for i in param_combinations:

    print('--------train CNN ',count, ' -----------')

    print('HyperParameters: ', i)

    model = create_CNN(**i)

    model.fit_generator(generator=train_generator,

                        steps_per_epoch=STEP_SIZE_TRAIN, 

                        # Total number of steps (batches of samples) to yield from generator 

                        # before declaring one epoch finished and starting the next epoch.

                        validation_data=test_generator,

                        validation_steps=STEP_SIZE_TEST,

                        epochs=3

    )

    count +=1

# Now you just wait and see..
import keras

def create_CNN3(activation = 'relu', dropout_flag = False, batchnorm_flag = False, 

               dropout_rate = 0.5, learning_rate = 1e-4, optimizer = 'rmsprop', initializer = 'glorot_uniform'):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),

                     activation = activation,

                     input_shape=input_shape,

                     kernel_initializer = initializer))

    if batchnorm_flag:

        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if dropout_flag:

        model.add(Dropout(dropout_rate))



    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer = initializer))

    if batchnorm_flag:

        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    if dropout_flag:

        model.add(Dropout(dropout_rate))



    model.add(Conv2D(128, (3,3), activation=activation, kernel_initializer = initializer))

    if batchnorm_flag:

        model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    if dropout_flag:

        model.add(Dropout(dropout_rate))



    model.add(Flatten())

    model.add(Dense(1000, activation=activation, kernel_initializer = initializer))

    # add drop out layer and fc layer

    model.add(Dropout(dropout_rate))

    model.add(Dense(1000, activation = activation, kernel_initializer = initializer))

    

    model.add(Dense(1, activation='sigmoid')) 

    if optimizer == 'sgd':

        optimize_instance = keras.optimizers.SGD(lr = learning_rate)

    elif optimizer == 'adam':

        optimize_instance = keras.optimizers.Adam(learning_rate = learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    elif optimizer == 'rmsprop':

        optimize_instance = keras.optimizers.RMSprop(learning_rate= learning_rate, rho=0.9, decay = 1e-6)

    elif optimizer == 'adagrad':

        optimize_instance = keras.optimizers.Adagrad(learning_rate= learning_rate)

    else:

        print('Not a valid optimizer') 

    model.compile(optimize_instance,loss="binary_crossentropy",metrics=["accuracy"])

    return model
param_grid = dict(

#            activation = activations, 

#            optimizer = optimizers, 

#            learning_rate = learning_rates,

#            batchnorm_flag = [True],

#            dropout_flag = [True],

            dropout_rate = dropoutrates,

            initializer = initializers

            

)

import itertools as it

param_combinations = [dict(zip(param_grid,v)) for v in it.product(*param_grid.values())]

#len(param_combinations) # 9 combinations
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size # number of training instances // each batch size

# in our case it is 25000 // 32 = 781

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



# iterate over each combination of hyperparameters

count = 1

for i in param_combinations:

    if count != 1 and count != 2:

        print('--------train CNN ',count, ' -----------')

        print('HyperParameters: ', i)

        model = create_CNN3(**i)

        model.fit_generator(generator=train_generator,

                            steps_per_epoch=STEP_SIZE_TRAIN, 

                            # Total number of steps (batches of samples) to yield from generator 

                            # before declaring one epoch finished and starting the next epoch.

                            validation_data=test_generator,

                            validation_steps=STEP_SIZE_TEST,

                            epochs=3

        )

    count +=1

# Now you just wait and see..
m3 = create_CNN3(dropout_rate = 0.25)
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, patience = 3)

# patience = 3 means that if the accuracy on test set does not improve over 3 epochs we will stop the training
# fit the model

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size # number of training instances // each batch size

# in our case it is 25000 // 32 = 781

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history = m3.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN, 

                    # Total number of steps (batches of samples) to yield from generator 

                    # before declaring one epoch finished and starting the next epoch.

                    validation_data=test_generator,

                    validation_steps=STEP_SIZE_TEST,

                    epochs=80,

                    callbacks=[es]

)
# plot traning histoy -- loss

plt.plot(history.history['loss'], label = 'train')

plt.plot(history.history['val_loss'], label = 'test')

plt.title('loss')

plt.xlabel('epochs')

plt.legend()

plt.show()
# plot traning histoy

plt.plot(history.history['accuracy'], label = 'train')

plt.plot(history.history['val_accuracy'], label = 'test')

plt.title('accuracy')

plt.xlabel('epochs')

plt.legend()

plt.show()
# save model

m3.save("./model3_tunnedCNN.hdf5")

# to download the output model file from kaggle to laptop

from IPython.display import FileLink

FileLink(r'model3_tunnedCNN.hdf5')
from keras.applications.inception_v3 import InceptionV3

base_model = InceptionV3(weights='imagenet', include_top=False) 

# we don't need the final output layer, we just need the feature activation map
# freeze each layer of base model, which means that we are not going to update weights or biases along the training process

for layer in base_model.layers:

    layer.trainable = False
from keras.layers import GlobalAveragePooling2D, Dropout, Dense

from keras.models import Model

# keras model API

x = base_model.output

x = GlobalAveragePooling2D(name='avg_pool')(x) # you don't have to name the layer

x = Dropout(0.25)(x) # based on the tunned hyperparamters of model 3

x = Dense(256, activation = 'relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)



model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, patience = 3)

# patience = 3 means that if the accuracy on test set does not improve over 3 epochs we will stop the training
# fit the model

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size # number of training instances // each batch size

# in our case it is 25000 // 32 = 781

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN, 

                    # Total number of steps (batches of samples) to yield from generator 

                    # before declaring one epoch finished and starting the next epoch.

                    validation_data=test_generator,

                    validation_steps=STEP_SIZE_TEST,

                    epochs=80,

                    callbacks=[es]

)
# plot traning histoy -- loss

plt.plot(history.history['loss'], label = 'train')

plt.plot(history.history['val_loss'], label = 'test')

plt.title('model4 - transfered model - loss')

plt.xlabel('epochs')

plt.legend()

plt.show()
# plot traning histoy

plt.plot(history.history['accuracy'], label = 'train')

plt.plot(history.history['val_accuracy'], label = 'test')

plt.title('model4 - transfered model - accuracy')

plt.xlabel('epochs')

plt.legend()

plt.show()
model.save('./model4_transfer_lr_feature_extractor.hdf5')

# to download the output model file from kaggle to laptop

from IPython.display import FileLink

FileLink(r'model4_transfer_lr_feature_extractor.hdf5')
filenames = os.listdir("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/")

predictset = pd.DataFrame({'filename': filenames})

predictset.sort_values(by='filename', inplace = True)
predictset
# what is the last row.by index..delete that 

predictset.drop(4505, inplace = True)
predictIterator = test_datagen.flow_from_dataframe(dataframe = predictset,

                                                    directory = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/',

                                                    x_col = 'filename', y_col = None,

                                                    target_size = (image_size, image_size),

                                                     batch_size = 32,

                                                     class_mode = None,

                                                    shuffle = False) # don't shuffle, we are here to predict each image
labels = model.predict_generator(predictIterator)
labels.shape
submit = pd.DataFrame({})

submit['id'] = predictset.filename.str.split('.').str[0]

submit['label']  = np.round(labels[:,0]).astype(int)

submit.to_csv('submission_pa_hw6.csv', index=False)