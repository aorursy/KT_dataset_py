!pip install --upgrade -q tensorflow

!pip install -q image-classifiers

!pip install -q tensorflow-addons
import numpy as np

import pandas as pd

import scipy as sp



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import StratifiedKFold, train_test_split



import tensorflow as tf

from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input, Flatten, BatchNormalization

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau



import tensorflow_addons as tfa



from classification_models.tfkeras import Classifiers



from tqdm.notebook import tqdm



import skimage.io

import cv2

from PIL import Image



from functools import partial



import os, gc, time, random

from datetime import datetime



from math import ceil

import warnings

warnings.filterwarnings('ignore')



import albumentations
class config:

    seed = 2020

    batch_size = 16

    img_size = 64

    num_tiles = 16

    num_classes = 6

    num_splits = 5

    num_epochs = 4

    learning_rate = 3e-3

    num_workers = 1

    verbose = True

    backbone_train_path = '../input/prostate-cancer-grade-assessment/train_images/'

    backbone_test_path = '../input/prostate-cancer-grade-assessment/test_images/'
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



seed_everything(config.seed)
def get_axis_max_min(array, axis=0):

    one_axis = list((array != 255).sum(axis=tuple([x for x in (0, 1, 2) if x != axis])))

    axis_min = next((i for i, x in enumerate(one_axis) if x), 0)

    axis_max = len(one_axis) - next((i for i, x in enumerate(one_axis[::-1]) if x), 0)

    return axis_min, axis_max
class PANDAGenerator(Sequence):

    def __init__(self, df, config, mode='fit', apply_tfms=True, shuffle=True):

        super(PANDAGenerator, self).__init__()

        

        self.image_ids = df['image_id'].reset_index(drop=True).values

        self.labels = df['isup_grade'].reset_index(drop=True).values

        

        self.config = config

        self.shuffle = shuffle

        self.mode = mode

        

        self.apply_tfms = apply_tfms

        

        self.side = int(self.config.num_tiles ** 0.5)

        

        self.tfms = albumentations.Compose([

            albumentations.HorizontalFlip(p=0.5),

            albumentations.VerticalFlip(p=0.5),

            albumentations.ShiftScaleRotate(shift_limit=.1, scale_limit=.1, rotate_limit=20, p=0.5),

        ])

        

        self.on_epoch_end()

    

    def __len__(self):

        return int(np.floor(len(self.image_ids) / self.config.batch_size))

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.image_ids))

        

        if self.shuffle:

            np.random.shuffle(self.indexes)

    

    def __getitem__(self, index):

        X = np.zeros((self.config.batch_size, self.side * self.config.img_size, \

                      self.side * self.config.img_size, 3), dtype=np.float32)

        

        imgs_batch = self.image_ids[index * self.config.batch_size : (index + 1) * self.config.batch_size]

        

        for i, img_name in enumerate(imgs_batch):

            img_path = '{}/{}.tiff'.format(self.config.backbone_train_path, img_name)

            img_patches = self.get_patches(img_path)

            X[i, ] = self.glue_to_one(img_patches)

            

        if self.mode == 'fit':

            y = np.zeros((self.config.batch_size, self.config.num_classes), dtype=np.float32)

            lbls_batch = self.labels[index * self.config.batch_size : (index + 1) * self.config.batch_size]

            

            for i in range(self.config.batch_size):

                y[i, lbls_batch[i]] = 1

            return X, y

        

        elif self.mode == 'predict':

            return X

        

        else:

            raise AttributeError('mode parameter error')

            

    def get_patches(self, img_path):

        num_patches = self.config.num_tiles

        p_size = self.config.img_size

        img = skimage.io.MultiImage(img_path)[-1] / 255

        

        if self.apply_tfms:

            img = self.tfms(image=img)['image'] 

        

        pad0, pad1 = (p_size - img.shape[0] % p_size) % p_size, (p_size - img.shape[1] % p_size) % p_size

        

        img = np.pad(

            img,

            [

                [pad0 // 2, pad0 - pad0 // 2], 

                [pad1 // 2, pad1 - pad1 // 2], 

                [0, 0]

            ],

            constant_values=1

        )

        

        img = img.reshape(img.shape[0] // p_size, p_size, img.shape[1] // p_size, p_size, 3)

        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, p_size, p_size, 3)

        

        if len(img) < num_patches:

            img = np.pad(

                img, 

                [

                    [0, num_patches - len(img)],

                    [0, 0],

                    [0, 0],

                    [0, 0]

                ],

                constant_values=1

            )

            

        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_patches]

        return np.array(img[idxs])

    

    def glue_to_one(self, imgs_seq):

        img_glue = np.zeros((self.config.img_size * self.side, self.config.img_size * self.side, 3), dtype=np.float32)

        

        for i, ptch in enumerate(imgs_seq):

            x = i // self.side

            y = i % self.side

            img_glue[x * self.config.img_size : (x + 1) * self.config.img_size, 

                     y * self.config.img_size : (y + 1) * self.config.img_size, :] = ptch

            

        return img_glue
df = pd.read_csv("../input/prostate-cancer-grade-assessment/train.csv", nrows=6000)

df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=config.seed)
train_datagen = PANDAGenerator(

    df=train_df, 

    config=config,

    mode='fit', 

    apply_tfms=False,

    shuffle=True, 

)



val_datagen = PANDAGenerator(

    df=valid_df, 

    config=config,

    mode='fit', 

    apply_tfms=False,

    shuffle=False, 

)
Xt, yt = train_datagen.__getitem__(0)



print('Shape of X: ', Xt.shape)

print('Shape of y: ', yt.shape)



fig, ax = plt.subplots(figsize=(15, 15), ncols=3)



for i in range(3):

    ax[i].imshow(Xt[i])

    ax[i].set_title('label {}'.format(np.argmax(yt[i, ])))

plt.show()
def build_seresnext():

        

    SEResNEXT50, _ = Classifiers.get('seresnext50')

    base_model = SEResNEXT50(input_shape=(config.img_size*int(config.num_tiles**0.5), \

                                          config.img_size*int(config.num_tiles**0.5), 3), \

                             weights='imagenet', include_top=False)

    

    x = GlobalAveragePooling2D()(base_model.output)

    output = Dense(config.num_classes, activation='softmax')(x)

    model = Model(inputs=[base_model.input], outputs=[output])



    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=config.learning_rate), \

              metrics=[tfa.metrics.CohenKappa(weightage='quadratic', num_classes=6)])

    

    return model
model = build_seresnext()

model.summary()
cb1 = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, verbose=1, min_lr=1e-6)

cb2 = ModelCheckpoint("best_seresnext50.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
history = model.fit_generator(

    train_datagen,

    validation_data=val_datagen,

    callbacks=[cb1, cb2],

    epochs=config.num_epochs,

    verbose=1

)
#  "Accuracy"

plt.plot(history.history['cohen_kappa'])

plt.plot(history.history['val_cohen_kappa'])

plt.title('model cohen kappa')

plt.ylabel('cohen kappa')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
model.load_weights('best_seresnext50.h5')
def get_img_array(img_path):

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(config.img_size, config.img_size))

    array = tf.keras.preprocessing.image.img_to_array(img)

    array = np.expand_dims(array, axis=0) # Add one dimension to transform our array into a batch

    return array



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):

    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = Model(model.inputs, last_conv_layer.output)

    

    classifier_input = Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for layer_name in classifier_layer_names:

        x = model.get_layer(layer_name)(x)

    classifier_model = Model(classifier_input, x)

    

    with tf.GradientTape() as tape:

        last_conv_layer_output = last_conv_layer_model(img_array)

        tape.watch(last_conv_layer_output)

        

        preds = classifier_model(last_conv_layer_output)

        top_pred_index = tf.argmax(preds[0])

        top_class_channel = preds[:, top_pred_index]

        

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    

    last_conv_layer_output = last_conv_layer_output.numpy()[0]

    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):

        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    

    heatmap = np.mean(last_conv_layer_output, axis=-1)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap



def create_superimposed_visualization(img, heatmap):

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    

    return superimposed_img
# We need to get the names of the last convolution layer of ResNet50

last_conv_layer_name = 'activation_80'



# We also need the names of all the layers that are part of the model head

classifier_layer_names = [

    'global_average_pooling2d_16',

    'dense'

]
# Let's take the first 5 images of the dataset



fig, ax = plt.subplots(figsize=(15, 10), ncols=3, nrows=2)



for i in range(3):

    raw_image = Xt[i]



    image = np.expand_dims(raw_image, axis=0)



    heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name, classifier_layer_names)

    superimposed_image = create_superimposed_visualization(raw_image, heatmap)



    ax[0][i].imshow(raw_image)

    ax[0][i].set_title('Original - label {}'.format(np.argmax(yt[i, ])))

    ax[1][i].imshow(superimposed_image)

    ax[1][i].set_title('GradCAM - label {}'.format(np.argmax(yt[i, ])))



fig.suptitle('SE_ResNeXT_50')

plt.show()