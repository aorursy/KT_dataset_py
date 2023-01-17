import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import disk

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, AveragePooling2D

from tqdm import tqdm_notebook
img_size_target = 101

# removed upsample and downsample code, since resizing is not used
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])

train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
# Simple split of images into training and testing sets
ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.2)
# Nikhil Tomar -- Unet with layer concatenation in downblock
# https://www.kaggle.com/nikhilroxtomar/unet-with-layer-concatenation-in-downblock

def inception(input_layer, base_name, num_filters=16):
    # Inception module
    incep_1x1 = Conv2D(num_filters, (1,1), padding='same', activation='relu', name=base_name + 'incep_1x1')(input_layer)
    incep_3x3_reduce = Conv2D(int(num_filters*1.5), (1,1), padding='same', activation='relu', name=base_name + 'incep_3x3_reduce')(input_layer)
    incep_3x3 = Conv2D(num_filters, (3,3), padding='same', activation='relu', name=base_name + 'incep_3x3')(incep_3x3_reduce)
    incep_5x5_reduce = Conv2D(int(num_filters/4), (1,1), padding='same', activation='relu', name=base_name + 'incep_5x5_reduce')(input_layer)
    incep_5x5 = Conv2D(int(num_filters/2), (5,5), padding='same', activation='relu', name=base_name + 'incep_5x5')(incep_5x5_reduce)
    incep_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same', name=base_name + 'incep_pool')(input_layer)
    incep_pool_proj = Conv2D(int(num_filters/2), (1,1), padding='same', activation='relu', name=base_name + 'incep_pool_proj')(incep_pool)
    incep_output = concatenate([incep_1x1, incep_3x3, incep_5x5, incep_pool_proj], axis = 3, name=base_name + 'incep_output')
    return incep_output

def down_block(input_layer, base_name,  num_filters=16, padding='same', dropout=0.25):
    conv1 = Conv2D(num_filters, (3, 3), activation="relu", padding=padding, name=base_name + 'conv1')(input_layer)
    conv2 = Conv2D(num_filters, (3, 3), activation="relu", padding=padding, name=base_name + 'conv2')(conv1)
    pool = MaxPooling2D((2, 2), name=base_name + 'pool')(conv2)
    output = Dropout(dropout)(pool)
    return output, conv2

def down_inception(input_layer, base_name, num_filters=16, padding='same', dropout=0.25):
    #First Inception module
    incep_1_output = inception(input_layer, base_name+"incep1", num_filters)
    #Second Inception module
    incep_2_output = inception(incep_1_output, base_name+"incep2", num_filters)
    pool_incep = MaxPooling2D((2, 2), name=base_name + 'pool')(incep_2_output)
    output_layer = Dropout(dropout, name=base_name + 'drop')(pool_incep)
    return output_layer, incep_2_output


def down_block_resnet(x, num_filters=16, kernel_size=(3, 3), padding='same', activation='relu', pool_size=(2, 2), dropout=0.25):
    conv = resnet_block(x, num_filters=num_filters, kernel_size=kernel_size, padding=padding,
    activation=activation)
    pool = conv
    if pool_size != None:
        #pool = MaxPooling2D(pool_size) (conv)
        pool = Conv2D(num_filters, kernel_size, padding='same', strides=pool_size, activation='tanh') (conv)
        #pool = Conv2D(num_filters, kernel_size, padding='same', activation='tanh') (pool)
    if dropout != None:
        pool = Dropout(dropout) (pool)
    return pool, conv

def up_block(uconv_input, conv_input, base_name, num_filters=16, padding='same', dropout=0.25):
    deconv = Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding=padding, name=base_name + 'deconv')(uconv_input)
    uconv1 = concatenate([deconv, conv_input], name=base_name + 'concate')
    uconv2 = Dropout(dropout, name=base_name + 'drop')(uconv1)
    uconv3 = Conv2D(num_filters, (3, 3), padding="same", name=base_name + 'conv1')(uconv2)
    uconv4 = Conv2D(num_filters, (3, 3), padding="same", name=base_name + 'conv2')(uconv3)
    return uconv4

def resnet_block(x, num_filters=16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'):
    conv1 = Conv2D(num_filters, (7, 7), padding=padding) (x)
    conv1 = _activation(activation, conv1)

    conv2 = Conv2D(num_filters, (5, 5), padding=padding) (x)
    conv2 = _activation(activation, conv2)

    conv3 = Conv2D(num_filters, (3, 3), padding=padding) (x)
    conv3 = _activation(activation, conv3)

    return concatenate([conv1, conv2, conv3, x])

def Unet_standard(num_filters=16,):
    input_img = Input((img_size_target, img_size_target, 1), name='img')
    input_features = Input((1, ), name='feat')

    pool1, conv1 = down_block(input_img, "down1", num_filters * 1, dropout=0.3)
    pool2, conv2 = down_block(pool1, "down2", num_filters * 2, dropout=0.5)
    pool3, conv3 = down_block(pool2, "down3", num_filters * 4, dropout=0.5)
    pool4, conv4 = down_block(pool3, "down4", num_filters * 8, dropout=0.5)

    # Middle
    middle1 = Conv2D(num_filters * 16, (3, 3), activation="relu", padding="same")(pool4)
    middle2 = Conv2D(num_filters * 16, (3, 3), activation="relu", padding="same")(middle1)

    deconv4 = up_block(middle2, conv4, "up4", num_filters * 8, dropout=0.5)
    deconv3 = up_block(deconv4, conv3, "up3", num_filters * 4, padding='valid', dropout=0.5)
    deconv2 = up_block(deconv3, conv2, "up2", num_filters * 2, dropout=0.5)
    deconv1 = up_block(deconv2, conv1, "up1", num_filters * 1, padding='valid', dropout=0.5)
    deconv1 = Dropout(0.5) (deconv1)
    output_layer = Conv2D(1, (1, 1), padding='same', activation='sigmoid') (deconv1)

    return Model(inputs=[input_img], outputs=[output_layer])

def Unet_incept(num_filters=16,):
    input_img = Input((img_size_target, img_size_target, 1), name='img')
    input_features = Input((1, ), name='feat')

    pool1, conv1 = down_inception(input_img, "down1", num_filters * 1, dropout=0.3)
    pool2, conv2 = down_inception(pool1, "down2", num_filters * 2, dropout=0.5)
    pool3, conv3 = down_inception(pool2, "down3", num_filters * 4, dropout=0.5)
    pool4, conv4 = down_inception(pool3, "down4", num_filters * 8, dropout=0.5)

    # Middle
    middle1 = Conv2D(num_filters * 16, (3, 3), activation="relu", padding="same")(pool4)
    middle2 = Conv2D(num_filters * 16, (3, 3), activation="relu", padding="same")(middle1)

    deconv4 = up_block(middle2, conv4, "up4", num_filters * 8, dropout=0.5)
    deconv3 = up_block(deconv4, conv3, "up3", num_filters * 4, padding='valid', dropout=0.5)
    deconv2 = up_block(deconv3, conv2, "up2", num_filters * 2, dropout=0.5)
    deconv1 = up_block(deconv2, conv1, "up1", num_filters * 1, padding='valid', dropout=0.5)
    deconv1 = Dropout(0.5) (deconv1)
    output_layer = Conv2D(1, (1, 1), padding='same', activation='sigmoid') (deconv1)

    return Model(inputs=[input_img], outputs=[output_layer])


model = Unet_standard()
#model = Unet_incept()
optimizer_Adam = Adam(lr=0.0001)
model.compile(loss="binary_crossentropy", optimizer=optimizer_Adam, metrics=['accuracy'])
model.summary()
model.save_weights('imageWeights.h5')


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=0.000001)
epochs = 5          # REDUCE FROM 100 FOR KAGGLE RUNTIME LIMIT
batch_size = 16     # REDUCED TO 16 FOR MEMORY USE

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[reduce_lr])
def plot_training_results(history):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_acc.plot(history.epoch, history.history["acc"], label="Train accuracy")
    ax_acc.plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
plot_training_results(history)


def contrast_stretch(img):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    if (p2==p98):
        return img      # some images are just one color, so they gerenate an divide by zero error, so return original image
    img_contrast_stretch = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_contrast_stretch

def equalization(img):
    # some images are just one color, so they gerenate a divide by zero error
    #     so return original image if the min and max values are the same
    if (np.max(img) == np.min(img) ):
        return img      
    # Equalization
    img_equalized = exposure.equalize_hist(img)
    return img_equalized

def adaptive_equalization(img):
    # some images are just one color, so they gerenate a divide by zero error
    #     so return original image if the min and max values are the same
    if (np.max(img) == np.min(img) ):
        return img      
    # Adaptive Equalization
    img_adaptive_equalized = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adaptive_equalized

def local_equalization(img):
    # some images are just one color, so they gerenate a divide by zero error
    #     so return original image if the min and max values are the same
    if (np.max(img) == np.min(img) ):
        return img      
    # Local Equalization--for details see http://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_local_equalize.html
    selem = disk(30)
    img_local_equal = rank.equalize(img, selem=selem)
    return img_local_equal

def display_equalizations(img):
    fix, axs = plt.subplots(1, 5, figsize=(15,5))
    axs[0].imshow(img, cmap="Greys")
    axs[0].set_title("Original image")

    axs[1].imshow(contrast_stretch(img), cmap="Greys")
    axs[1].set_title("Contrast stretching")

    axs[2].imshow(equalization(img), cmap="Greys")
    axs[2].set_title("Equalized image")

    axs[3].imshow(adaptive_equalization(img), cmap="Greys")
    axs[3].set_title("Adaptive Equalization image")

    axs[4].imshow(local_equalization(img), cmap="Greys")
    axs[4].set_title("Local Equalization image")

img = train_df.images.loc[ids_train[11]]
display_equalizations(img)
img = train_df.images.loc[ids_train[14]]
display_equalizations(img)
img = train_df.images.loc[ids_train[27]]
display_equalizations(img)
# Redo the train/test split with the contrast_stretch added in
# --- TODO: there should be a better way to do this directly on train/test arrays
ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(contrast_stretch).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.2, random_state=77777)

# Run the training 
model.load_weights('imageWeights.h5')       # reload the initial, untrained wieghts to keep each trial the same
optimizer_Adam = Adam(lr=0.0001)            # reset the learning rate back to the starting value 
model.compile(loss="binary_crossentropy", optimizer=optimizer_Adam, metrics=['accuracy'])
history_contrast_stretch = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[reduce_lr])

plot_training_results(history_contrast_stretch)
# Redo the train/test split with the contrast_stretch added in
# --- TODO: there should be a better way to do this directly on train/test arrays
ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(equalization).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.2, random_state=77777)

# Run the training 
model.load_weights('imageWeights.h5')       # reload the initial, untrained wieghts to keep each trial the same
optimizer_Adam = Adam(lr=0.0001)            # reset the learning rate back to the starting value 
model.compile(loss="binary_crossentropy", optimizer=optimizer_Adam, metrics=['accuracy'])
history_equalization = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[reduce_lr])

plot_training_results(history_equalization)
# Redo the train/test split with the contrast_stretch added in
# --- TODO: there should be a better way to do this directly on train/test arrays
ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(adaptive_equalization).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.2, random_state=77777)

# Run the training 
model.load_weights('imageWeights.h5')       # reload the initial, untrained wieghts to keep each trial the same
optimizer_Adam = Adam(lr=0.0001)            # reset the learning rate back to the starting value 
model.compile(loss="binary_crossentropy", optimizer=optimizer_Adam, metrics=['accuracy'])
history_adaptive_equalization = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[reduce_lr])

plot_training_results(history_adaptive_equalization)
# Redo the train/test split with the contrast_stretch added in
# --- TODO: there should be a better way to do this directly on train/test arrays
ids_train, ids_valid, x_train, x_valid, y_train, y_valid = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(local_equalization).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    test_size=0.2, random_state=77777)

# Run the training 
model.load_weights('imageWeights.h5')       # reload the initial, untrained wieghts to keep each trial the same
optimizer_Adam = Adam(lr=0.0001)            # reset the learning rate back to the starting value 
model.compile(loss="binary_crossentropy", optimizer=optimizer_Adam, metrics=['accuracy'])

history_local_equalization = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[reduce_lr])

plot_training_results(history_local_equalization)

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1, figsize=(15,15))
#ax[0].title = "Loss"
ax[0].plot(history.history['loss'], color='b', label="Stock image")
ax[0].plot(history_contrast_stretch.history['loss'], color='g', label="contrast_stretch ")
ax[0].plot(history_equalization.history['loss'], color='y', label="equalization ")
ax[0].plot(history_adaptive_equalization.history['loss'], color='r', label="adaptive_equalization ")
ax[0].plot(history_local_equalization.history['loss'], color='c', label="local_equalization")

ax[0].plot(history.history['val_loss'], color='b', linestyle=':')
ax[0].plot(history_contrast_stretch.history['val_loss'], color='g', linestyle=':')
ax[0].plot(history_equalization.history['val_loss'], color='y', linestyle=':')
ax[0].plot(history_adaptive_equalization.history['val_loss'], color='r', linestyle=':')
ax[0].plot(history_local_equalization.history['val_loss'], color='c', linestyle=':')
linestyle=':'
legend = ax[0].legend(loc='best', shadow=True)
#plt.ylim(0,1)

#ax[0].title = "Accuracy"
ax[1].plot(history.history['acc'], color='b', label="Stock Image")
ax[1].plot(history_contrast_stretch.history['acc'], color='g', label="contrast_stretch")
ax[1].plot(history_equalization.history['acc'], color='y', label="equalization")
ax[1].plot(history_adaptive_equalization.history['acc'], color='r', label="adaptive_equalization")
ax[1].plot(history_local_equalization.history['acc'], color='c', label="local_equalization")

ax[1].plot(history.history['val_acc'], color='b', linestyle=':')
ax[1].plot(history_contrast_stretch.history['val_acc'], color='g', linestyle=':')
ax[1].plot(history_equalization.history['val_acc'], color='y', linestyle=':')
ax[1].plot(history_adaptive_equalization.history['val_acc'], color='r', linestyle=':')
ax[1].plot(history_local_equalization.history['val_acc'], color='c', linestyle=':')
legend = ax[1].legend(loc='best', shadow=True)

