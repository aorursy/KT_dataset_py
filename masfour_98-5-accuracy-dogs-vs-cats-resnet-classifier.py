# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from pathlib import Path

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras import backend as K

import keras

from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input, Flatten, Conv2D, MaxPooling2D

from keras.models import Model, load_model

from keras.optimizers import RMSprop, SGD, Adam

from keras.callbacks import Callback

from keras.applications.resnet50 import ResNet50

from keras.regularizers import l1, l2

from keras.initializers import he_normal

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from IPython.display import clear_output



import os





# Any results you write to the current directory are saved as output.
!mkdir /root/data
!unzip /kaggle/input/dogs-vs-cats/train.zip -d /root/data

!unzip /kaggle/input/dogs-vs-cats/test1.zip -d /root/data
train_dir = '/root/data/train/'  # the images directory

!ls $train_dir -U | head -5  # view sample images filenames - bash command
images_names = os.listdir(train_dir)  # names of the files in the directory

images_num = len(images_names)

print(f'Number of images: {images_num}')
# dimensions of images to use for plt.imshow

width, height, channels = 256, 256, 3
images_samples = np.zeros((4, height, width, 3), dtype=int)

samples_labels = []

# get random 4 images

for i in range(4):

    rnd_img = np.random.randint(0, images_num)

    img_filename = images_names[rnd_img]

    img_bgr = cv2.imread(train_dir + img_filename)  # loads the images channels in (blue, green, red) order

    images_samples[i] = cv2.resize(src=img_bgr[:, :, [2, 1, 0]], dsize=(width, height))  # store the random image

    samples_labels.append(img_filename[:3])  # store the random images' label
# view the 4 samples

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):

    ax.imshow(img);

    ax.set_title(f'Class: {label}', size=15);
new_train_dir = '/root/train/'  # new directory for traininng data 

!mkdir $new_train_dir
!ls /root
# create subdirectories to use as classes for keras to feed the model - bash command

os.mkdir(new_train_dir+'dogs')

os.mkdir(new_train_dir+'cats')
# move the images to their corresponding subdirectories - bash command

!cp $train_dir/cat.* $new_train_dir/cats/

!cp $train_dir/dog.* $new_train_dir/dogs/
# check the directory - bash command

!ls $new_train_dir
def scale_images(x):

  return x / 255



# Augmentation Ranges

transform_params = {

    'featurewise_center': False,

    'featurewise_std_normalization': False,

    'samplewise_center': False,

    'samplewise_std_normalization': False,

    'rotation_range': 30, 

    'width_shift_range': 0.15, 

    'height_shift_range': 0.15,

    'horizontal_flip': True,

    'validation_split': 0.25,

    'preprocessing_function': scale_images

}

img_gen = ImageDataGenerator(**transform_params) 
fig, axs = plt.subplots(2, 4, figsize=(20,10))  # let's see 4 augmentation examples

fig.suptitle('Augmentation Results', size=32)



for axs_col in range(axs.shape[1]):

    viz_transoform_params = {  # defined each iteration to get new augmentation values each time

        'theta': np.random.randint(-transform_params['rotation_range'], transform_params['rotation_range']),

        'tx': np.random.uniform(0, transform_params['width_shift_range']),

        'ty': np.random.uniform(0, transform_params['height_shift_range']),

        'flip_horizontal': np.random.choice([True, False], p=[0.5, 0.5])

    }



    img = images_samples[axs_col]  # the original image

    aug_img = img_gen.apply_transform(img, viz_transoform_params)  # the same image after augmentation

    

    axs[0, axs_col].imshow(img);

    axs[0, axs_col].set_title('Original Image', size=15)

    

    axs[1, axs_col].imshow(aug_img);

    axs[1, axs_col].set_title('Augmented Image', size=15)
# a Fully connected layer with activation, batchnorm and dropout

def dense_block(x, neurons, layer_no):

    x = Dense(neurons, kernel_initializer=he_normal(layer_no), name=f'Dense{layer_no}')(x)

    x = Activation('relu', name=f'Relu{layer_no}')(x)

    x = BatchNormalization(name=f'BatchNorm{layer_no}')(x)

    x = Dropout(0.5, name=f'Dropout{layer_no}')(x)

    return x
def create_model(shape):

    input_layer = Input(shape, name='input_layer')  # input layer with given shape

    

    # load ResNet50 with initialized weights and remove final dense layers - keep as trainable layers

    resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)



    # dense layers after the ResNet50 initialized layers

    flat1 = Flatten(name='Flatten')(resnet.output)

    flat_bn = BatchNormalization()(flat1)

    drop1 = Dropout(0.5)(flat_bn)

    dens1 = dense_block(drop1, neurons=512, layer_no=1)

    dens2 = dense_block(dens1, neurons=256, layer_no=3)

    

    dens4 = Dense(1, name='Dense4')(dens2)

    output_layer = Activation('sigmoid')(dens4)

    

    model = Model(inputs=[input_layer], outputs=[output_layer])



    return model
# used to plot training curves (accuracy, loss) while model is training

class Plotter(Callback):

    def plot(self):  # Updates the graph

        clear_output(wait=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        

        # plot the losses

        ax1.plot(self.epochs, self.losses, label='train_loss')

        ax1.plot(self.epochs, self.val_losses, label='val_loss')

        

        # plot the accuracies

        ax2.plot(self.epochs, self.acc, label='train_acc')

        ax2.plot(self.epochs, self.val_acc, label='val_acc')

    

        ax1.set_title(f'Loss vs Epochs')

        ax1.set_xlabel("Epochs")

        ax1.set_ylabel("Loss")

        

        ax2.set_title(f'Accuracy vs Epochs')

        ax2.set_xlabel("Epoches")

        ax2.set_ylabel("Accuracy")

        

        ax1.legend()

        ax2.legend()

        plt.show()

        

        # print out the accuracies at each epoch

        print(f'Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}')

        print(f'Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}')

        

    def on_train_begin(self, logs={}):

        # initialize lists to store values from training

        self.losses = []

        self.val_losses = []

        self.epochs = []

        self.batch_no = []

        self.acc = []

        self.val_acc = []

    

    def on_epoch_end(self, epoch, logs={}):

        # append values from the last epoch

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.epochs.append(epoch)

        self.plot()  # update the graph

        

    def on_train_end(self, logs={}):

        self.plot()

               

plotter = Plotter()
# used to decrease the learning rate if val_acc doesn't enhance

plateau_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.75,

                              patience=1, min_lr=0.000001)
callbacks = [plotter, plateau_reduce] 
# hyperparameters

height, width, channels_num = 256, 256, 3

learning_rate = 0.00005

epochs = 8

batch_size = 64  # if increased to 64 you may run out of gpu memory - reduce image size if you want to increase the batch size
model = create_model((height, width, channels_num))

optimizer = Adam(learning_rate)



model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

model.summary()
# used to feed the model augmented training data after being loaded from the directory

train_gen = img_gen.flow_from_directory(directory=new_train_dir, target_size=(height, width), color_mode='rgb', classes=['cats', 'dogs'], 

                                        class_mode='binary', batch_size=batch_size, shuffle=True, subset='training', interpolation='nearest')



# used to feed the model augmented validation data after being loaded from the directory

valid_gen = img_gen.flow_from_directory(directory=new_train_dir, target_size=(height, width), color_mode='rgb', classes=['cats', 'dogs'], 

                                        class_mode='binary', batch_size=batch_size, shuffle=True, subset='validation', interpolation='nearest')





model.fit_generator(train_gen, validation_data=valid_gen, epochs=epochs, 

                        steps_per_epoch=images_num*0.75//batch_size + 1, 

                        validation_steps=images_num*0.25//batch_size + 1, callbacks=callbacks)



# model.save('my_resnet_model.h5')
# check the corresponding classes of the binary encoding

train_gen.class_indices
# get first 3 convolutional layers to visualize their activations

layers_viz = []

for layer in model.layers:

  if len(layers_viz) < 3:

    if layer.__class__.__name__ == 'Conv2D':

      layers_viz.append(layer)

  else:

    break



# using keras backend functions to obtain layers activtions - check keras documentation

inp = model.input

outputs = [layer.output for layer in layers_viz]

functor = K.function([inp, K.learning_phase()], outputs)

test_idx = np.random.randint(0, images_samples.shape[0])

test_img = images_samples[test_idx]

test_img = test_img.reshape(1, *test_img.shape)

print(test_img.shape)

layers_outs = functor([test_img, 0])
# shapes of the selected layers' activations

for layer in layers_outs:

  print(layer.shape)
# plots activations in a grid of axes

def plot_activations(conv_layer):

  dim_sqrt = np.sqrt(conv_layer.shape[-1])

  rows_num = int(dim_sqrt)  # get integer number of rows

  cols_num = conv_layer.shape[-1] // rows_num

  fig, axs = plt.subplots(rows_num, cols_num, figsize=(16, 16))

  for filter_map, ax in zip(range(conv_layer.shape[-1]), axs.ravel()):

    activations = conv_layer[0, :, :, filter_map]

    activations = (activations - np.min(activations)) / (np.max(activations - np.min(activations)))  # normalize to give to plt.imshow 

    ax.set_axis_off()

    ax.imshow(activations, cmap='viridis')
# get a random sample from the 4 samples

test_idx = np.random.randint(0, images_samples.shape[0]-1)

test_img = images_samples[test_idx]

test_img = test_img.reshape(1, *test_img.shape)

layers_outs = functor([test_img, 0])  # the output of the keras functor we made - the activations
# the 1st conv layer activations

conv_layer = layers_outs[0]

print(f'Number of Filters: {conv_layer.shape[-1]}')

plot_activations(conv_layer=conv_layer)
# the 2nd conv layer activations

conv_layer = layers_outs[1]

print(f'Number of Filters: {conv_layer.shape[-1]}')

plot_activations(conv_layer=conv_layer)
# the 3rd conv layer activations

conv_layer = layers_outs[2]

print(f'Number of Filters: {conv_layer.shape[-1]}')

plot_activations(conv_layer=conv_layer)
!ls '/root/data/test1/' -U | head -5
test_dir = '/root/data/test1/'  # the images directory

test_images_names = os.listdir(test_dir)  # names of the files in the directory

test_images_num = len(test_images_names)

print(f'Number of images: {test_images_num}')
test_ids = np.apply_along_axis(lambda x: x[0][:-4], axis=0, 

                               arr=np.array(test_images_names).reshape(1, test_images_num))
test_ids.shape
test_images_names[4]
test_ids[4]
test_ids = test_ids.tolist()
parent_dir = '/root/data'

# generator to feed the test images to model in batches - doesn't augment, just the preprocess function

test_gen = ImageDataGenerator(preprocessing_function=scale_images).flow_from_directory(

                                        directory=parent_dir, target_size=(height, width), color_mode='rgb', classes=['test1'],

                                        class_mode=None, batch_size=50, shuffle=False, interpolation='nearest')
preds = model.predict_generator(test_gen)  # model predictions

print(preds[:10])
binary_preds = (preds >= 0.5).astype(int)  # convert predictions to binary 

print(binary_preds[:10])
ids_arr = np.array(test_ids).astype(int).reshape(-1, 1)  # reshape ids as rows

ids_arr.shape
sub_data = np.hstack([ids_arr, binary_preds])  # data to put in a dataframe for submission
sub_df = pd.DataFrame(data=sub_data, columns=['id', 'label'])
sub_df.head(5)
sub_df.sort_values(by='id', inplace=True)  # order by id

sub_df.reset_index(inplace=True, drop=True)  # reset the index

sub_df.head(5)
# save to csv

sub_df.to_csv('submission.csv', index=False)