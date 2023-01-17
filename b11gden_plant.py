import numpy as np

import pandas as pd 



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
import keras

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

from keras import regularizers, optimizers

from keras import Input, Model

from keras.applications import MobileNet, ResNet50
main_path   = '/kaggle/input/plant-pathology-2020-fgvc7/'

batch_size  = 128

image_size  = 224

epochs      = 50

val_rate    = 0.1

bn_momentum = 0.9
def plot_batch(batch):

    fig, axs = plt.subplots(4, 4, figsize=(15,10))

    

    for i in range(4):

        for j in range(4):

            img   = batch[0][i*4 + j]

            axs[i, j].imshow(img)

            

    plt.show()

    

def plot_history(history):

    fig, axs = plt.subplots(1, 2, figsize=(15,5))

    

    axs[0].plot(history['loss'],     label = 'train_loss')

    axs[0].plot(history['val_loss'], label = 'val_loss')

    axs[0].legend()

    axs[1].plot(history['categorical_accuracy'],     label = 'train_acc')

    axs[1].plot(history['val_categorical_accuracy'], label = 'val__acc')

    axs[1].legend()

    

    

    plt.show()
train_df = pd.read_csv(main_path + 'train.csv')

train_df['image_id']                                  = train_df['image_id'] + '.jpg'

train_df['y_col']                                     = train_df['image_id']

train_df['y_col'][train_df['healthy'] == 1]           = 'healthy'

train_df['y_col'][train_df['multiple_diseases'] == 1] = 'multiple_diseases'

train_df['y_col'][train_df['rust'] == 1]              = 'rust'

train_df['y_col'][train_df['scab'] == 1]              = 'scab'

train_df
plt.hist(train_df['y_col'])

plt.show()
data_gen = ImageDataGenerator(

    rescale = 1./255,

    horizontal_flip = True,

    vertical_flip = True,

    validation_split=val_rate

)



train = data_gen.flow_from_dataframe(

    train_df,

    directory = main_path + 'images',

    x_col = 'image_id',

    y_col = 'y_col',

    class_mode = 'categorical',

    target_size=(image_size, image_size),

    batch_size = batch_size,

    shuffle = True,

    subset="training",

)



val = data_gen.flow_from_dataframe(

    train_df,

    directory = main_path + 'images',

    x_col = 'image_id',

    y_col = 'y_col',

    class_mode = 'categorical',

    target_size=(image_size, image_size),

    batch_size = batch_size,

    shuffle = False,

    subset="validation",

)
classes = train.class_indices

classes
for batch in train:

    plot_batch(batch)

    break
inputs = Input(shape = (image_size, image_size, 3))



model = ResNet50(

    weights = None,

    classes = 4

)



model.compile(optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['categorical_accuracy']

)



for layer in model.layers:

    if layer.name.split('_')[-1] == 'bn':

        layer.momentum = bn_momentum
model.summary()
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)
history = model.fit(

    train,

    epochs=epochs,

    validation_data = val

)
plot_history(history.history)
submit = pd.read_csv(main_path + 'sample_submission.csv')

submit['filename'] = submit['image_id'] + '.jpg'

submit
test_data_gen = ImageDataGenerator(

    rescale = 1./255, 

)



test = test_data_gen.flow_from_dataframe(

    submit,

    directory = main_path + 'images',

    x_col = 'filename',

    class_mode = None,

    target_size=(image_size, image_size),

    batch_size = batch_size,

    shuffle = False,

)
predict = model.predict(test, verbose = 1)
submit['healthy']           = predict[:, classes['healthy']]

submit['multiple_diseases'] = predict[:, classes['multiple_diseases']]

submit['rust']              = predict[:, classes['rust']]

submit['scab']              = predict[:, classes['scab']]



del submit['filename']
submit
submit.to_csv("submission.csv", index=False)