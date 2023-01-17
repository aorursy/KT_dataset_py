"""Python file path, image, and data processing libraries."""

import random

import os

import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image
"""Deep learning libraries."""

import tensorflow as tf

import keras

from keras import backend as K

from keras.optimizers import Adam, SGD, Adagrad, Adadelta

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, CSVLogger

from keras.models import Model, Sequential, load_model, model_from_json

from keras.layers import Flatten, Dense, Activation, Input, Dropout, Activation, BatchNormalization, Reshape

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.regularizers import l2

from keras.layers import Dense, Conv2D, BatchNormalization, Activation

from keras.layers import AveragePooling2D, Input, Flatten

from keras.optimizers import Adam

from imgaug import augmenters as iaa
"""Sklearn functions that will help training"""

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer
!nvidia-smi
VERSION = 'v0'  # Model version

NUM_CLASSES = 28

INPUT_SHAPE = [300, 300, 1]

TRAIN_BATCH_SIZE = 8

VAL_BATCH_SIZE = 8



data_path = '/kaggle/input'

checkpoint_path = '/kaggle/working'

train_path = os.path.join(data_path, 'train')

test_path = os.path.join(data_path,'test')

labels_path = os.path.join(data_path, 'train.csv')

print(os.listdir(data_path))

train_label_csv = pd.read_csv(labels_path, index_col=False)  # Pandas for reading csv



def curate_dataset(data_csv):

    """Convert data csv into a list of dicts."""

    dataset = []

    for name, labels in zip(data_csv.Id, data_csv.Target.str.split(' ')):

        dataset += [{

            'path': os.path.join(train_path, name),

            'labels': np.array([int(label) for label in labels])}]

    dataset = np.array(dataset)

    return dataset





train_dataset = curate_dataset(train_label_csv)

print(train_dataset[:10])

train_ids, test_ids, train_targets, test_target = train_test_split(

    train_label_csv.Id,

    train_label_csv.Target,

    test_size=0.1,

    random_state=42)

# print(train_ids)

# print(train_targets)
class DataGenerator:

    """Data generator for feeding data to keras"""

    def __init__(self,

            label_dims=28,

            max_image=255.,

            batch_size=16,

            proc_img_size=[300, 300, 4],

            train=False,

            imagenet_proc=False):

        self.label_dims = label_dims

        self.max_image = max_image

        self.batch_size = batch_size

        self.proc_img_size = proc_img_size  # Crop to this size

        self.train = train

        self.imagenet_proc = imagenet_proc



    def build(self, dataset_info, augument=True):

        """Data processing routines for training."""

        while True:

            random_indexes = np.random.choice(len(dataset_info), self.batch_size)

            batch_images = np.empty(([self.batch_size] + self.proc_img_size))

            batch_labels = np.zeros((self.batch_size, self.label_dims))

            for i, idx in enumerate(random_indexes):

                image = self.load_image(dataset_info[idx]['path']).astype(np.float32)

                image = self.augmentations(image)

                image /= self.max_image  # Normalize

                image = np.maximum(np.minimum(image, 1), 0)  # Clip

                batch_images[i] = image

                batch_labels[i][dataset_info[idx]['labels']] = 1

            yield batch_images, batch_labels

    

    def load_image(self, path):

        """Preprocess image."""

        if self.proc_img_size[-1] == 1:

            image = np.array(Image.open(path + '_green.png'))

            if len(image.shape) == 2:

                image = image[..., None]

        else:

            R = np.array(Image.open(path + '_red.png'))

            G = np.array(Image.open(path + '_green.png'))

            B = np.array(Image.open(path + '_blue.png'))

            Y = np.array(Image.open(path + '_yellow.png'))



            if self.imagenet_proc:

                image = np.stack((

                    R/2 + Y/2, 

                    G/2 + Y/2, 

                    B),-1)

            else:

                image = np.stack((R, G, B, Y), axis=-1)

            # image = cv2.resize(image, (self.proc_img_size[0], self.proc_img_size[1]))

        return image



    def augmentations(self, image):

        """Apply data augmentations to training images."""

        if self.train:

            augment_img = iaa.Sequential([

                iaa.OneOf([

                    iaa.Affine(rotate=0),

                    iaa.Affine(rotate=90),

                    iaa.Affine(rotate=180),

                    iaa.Affine(rotate=270),

                    iaa.Fliplr(0.5),

                    iaa.Flipud(0.5),

                    # iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)

                    # iaa.PiecewiseAffine(scale=(0.01, 0.05))

                ]),

                iaa.Fliplr(0.5),

                iaa.Flipud(0.5),

                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                iaa.CropToFixedSize(

                    width=self.proc_img_size[0],

                    height=self.proc_img_size[1],

                    position='uniform')],

            random_order=True)

        else:

            augment_img = iaa.Sequential([

                iaa.CropToFixedSize(

                    width=self.proc_img_size[0],

                    height=self.proc_img_size[1],

                    position='center')])

        image_aug = augment_img.augment_image(image)

        return image_aug



    

# Create train/val datagens

train_datagen = DataGenerator(

    batch_size=TRAIN_BATCH_SIZE,

    proc_img_size=INPUT_SHAPE,

    label_dims=NUM_CLASSES,

    train=True)

train_datagen = train_datagen.build(

    dataset_info=train_dataset[train_ids.index])

val_datagen = DataGenerator(

    batch_size=VAL_BATCH_SIZE,

    proc_img_size=INPUT_SHAPE,

    label_dims=NUM_CLASSES,

    train=False)

val_datagen = val_datagen.build(

    dataset_info=train_dataset[test_ids.index])

# train_datagen = DataGenerator(proc_img_size=[224, 224, 3], imagenet_proc=True)
def plot_images(images, labels, title, num_ims=5):

    """Plot mosaic of images with matplotlib."""

    fig, axs = plt.subplots(1, num_ims, figsize=(25,5))

    plt.suptitle(title)

    for idx, (ax, im, lab) in enumerate(zip(axs, images, labels)):

        ax.imshow(im.squeeze())

        ax.axis('off')

        ax.set_title('Label: %s' % np.where(lab)[0])

    # plt.show()  # Only if not executing in ipython notebook

    print('{0} range, min: {1}, max: {2}'.format(title, images.min(), images.max()))



images, labels = next(train_datagen)

plot_images(images=images, labels=labels, title='Train')

images, labels = next(val_datagen)

plot_images(images=images, labels=labels, title='Val')

def resnet_layer(

        inputs,

        num_filters=16,

        kernel_size=3,

        strides=1,

        activation='relu',

        batch_normalization=True,

        conv_first=True):

    """2D Convolution-Batch Normalization-Activation stack builder



    # Arguments

        inputs (tensor): input tensor from input image or previous layer

        num_filters (int): Conv2D number of filters

        kernel_size (int): Conv2D square kernel dimensions

        strides (int): Conv2D square stride dimensions

        activation (string): activation name

        batch_normalization (bool): whether to include batch normalization

        conv_first (bool): conv-bn-activation (True) or

            bn-activation-conv (False)



    # Returns

        x (tensor): tensor as input to the next layer

    """

    conv = Conv2D(num_filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-4))



    x = inputs

    if conv_first:

        x = conv(x)

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

    else:

        if batch_normalization:

            x = BatchNormalization()(x)

        if activation is not None:

            x = Activation(activation)(x)

        x = conv(x)

    return x



def resnet_v2(input_shape, depth, num_classes=NUM_CLASSES):

    """ResNet Version 2 Model builder [b]



    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as

    bottleneck layer

    First shortcut connection per layer is 1 x 1 Conv2D.

    Second and onwards shortcut connection is identity.

    At the beginning of each stage, the feature map size is halved (downsampled)

    by a convolutional layer with strides=2, while the number of filter maps is

    doubled. Within each stage, the layers have the same number filters and the

    same filter map sizes.

    Features maps sizes:

    conv1  : 32x32,  16

    stage 0: 32x32,  64

    stage 1: 16x16, 128

    stage 2:  8x8,  256



    # Arguments

        input_shape (tensor): shape of input image tensor

        depth (int): number of core convolutional layers

        num_classes (int): number of classes (CIFAR10 has 10)



    # Returns

        model (Model): Keras model instance

    """

    if (depth - 2) % 9 != 0:

        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

    # Start model definition.

    num_filters_in = 16

    num_res_blocks = int((depth - 2) / 9)



    inputs = Input(shape=input_shape)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths

    x = resnet_layer(inputs=inputs,

                     num_filters=num_filters_in,

                     conv_first=True)



    # Instantiate the stack of residual units

    for stage in range(3):

        for res_block in range(num_res_blocks):

            activation = 'relu'

            batch_normalization = True

            strides = 1

            if stage == 0:

                num_filters_out = num_filters_in * 4

                if res_block == 0:  # first layer and first stage

                    activation = None

                    batch_normalization = False

            else:

                num_filters_out = num_filters_in * 2

                if res_block == 0:  # first layer but not first stage

                    strides = 2    # downsample



            # bottleneck residual unit

            y = resnet_layer(inputs=x,

                             num_filters=num_filters_in,

                             kernel_size=1,

                             strides=strides,

                             activation=activation,

                             batch_normalization=batch_normalization,

                             conv_first=False)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters_in,

                             conv_first=False)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters_out,

                             kernel_size=1,

                             conv_first=False)

            if res_block == 0:

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters_out,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 batch_normalization=False)

            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out



    # Add classifier on top.

    # v2 has BN-ReLU before Pooling

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(

        num_classes,

        activation='softmax',

        kernel_initializer='he_normal')(y)



    # Instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model





keras.backend.clear_session()

model = resnet_v2(input_shape=INPUT_SHAPE, depth=56)

# print(model.summary())
def f1(y_true, y_pred):

    """Keras function for F1 score, which is the harmonic mean of precision/recall."""

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2 * p * r / (p + r + K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)



def show_history(history):

    """Plot training and validation performance."""

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].set_title('loss')

    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1].set_title('f1')

    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")

    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")

    ax[2].set_title('acc')

    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")

    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")

    ax[0].legend()

    ax[1].legend()

    ax[2].legend()



checkpointer = ModelCheckpoint(

    os.path.join(checkpoint_path, '%s_model.model' % VERSION),

    verbose=2,

    save_best_only=True)

earlyStopping = EarlyStopping(

    monitor='val_loss',

    min_delta=0,

    mode='min',

    patience=6,

    verbose=0,

    restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    mode='min',

    factor=0.2,

    patience=3,

    min_lr=1e-6,

    cooldown=1,

    verbose=1)

model.compile(

    loss='binary_crossentropy',  

    optimizer=Adam(1e-3),

    metrics=['acc', f1])
history = model.fit_generator(

    train_datagen,

    steps_per_epoch=20,

    validation_data=next(val_datagen),

    epochs=3, 

    verbose=1,

    callbacks=[checkpointer, earlyStopping, reduce_lr])
show_history(history)

model = load_model(

    os.path.join(checkpoint_path, '%s_model.model' % VERSION), 

    custom_objects={'f1': f1})
submit = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

test_dataset = []

for name in submit.Id:

    test_dataset += [{'path': os.path.join(test_path, name), 'labels': 0}]

test_dataset = np.array(test_dataset)



test_datagen = DataGenerator(

    batch_size=1,

    proc_img_size=INPUT_SHAPE,

    label_dims=NUM_CLASSES,

    train=False)

test_datagen = test_datagen.build(

    dataset_info=test_dataset)



predicted = []

threshold = 0.5  # Sigmoid probability

for _ in tqdm(test_dataset, total=len(test_dataset), desc='Generating test predictions.'):

    image = next(test_datagen)[0]

    score_predict = model.predict(image)[0]

    label_predict = np.arange(NUM_CLASSES)[score_predict >= threshold]

    str_predict_label = ' '.join(str(l) for l in label_predict)

    predicted += [str_predict_label]

submit['Predicted'] = predicted

submit.to_csv('submission.csv', index=False)

print(predicted)