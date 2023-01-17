import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
from imgaug import augmenters as iaa

import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle
import keras
import warnings
from keras.utils import Sequence
warnings.filterwarnings("ignore")
SIZE = 299
SEED = 777
THRESHOLD = 0.2
# Load dataset info
DIR = '../input/'
data = pd.read_csv('../input/train.csv')

# train_dataset_info = []
# for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
#     train_dataset_info.append({
#         'path':os.path.join(path_to_train, name),
#         'labels':np.array([int(label) for label in labels])})
# train_dataset_info = np.array(train_dataset_info)
def getTrainDataset():
    
    path_to_train = DIR + '/train/'
    data = pd.read_csv(DIR + '/train.csv')

    paths = []
    labels = []
    
    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)

def getTestDataset():
    
    path_to_test = DIR + '/test/'
    data = pd.read_csv(DIR + '/sample_submission.csv')

    paths = []
    labels = []
    
    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)
paths, labels = getTrainDataset()
# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
from random import randint
class ProteinDataGenerator(keras.utils.Sequence):
            
    def __init__(self, paths, labels, batch_size, shape, channels = [], shuffle = False, use_cache = False, augmentor = False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.channels = channels
        self.augmentor = augmentor
        self.clahe = cv2.createCLAHE()
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], len(channels)))
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], len(self.channels)))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]
        
        return X, y
    
    def on_epoch_end(self):
        
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            if self.augmentor == True:
                item = self.augment(item)
            yield item
            
    def __load_image(self, path):
        images = []
        for channel in self.channels:
            im = np.array(Image.open(path + '_' + channel + '.png'))
            
#             im = clahe.apply(im)
            images.append(im)
            
        if len(self.channels) >= 2:
            im = np.stack((
                images
            ), -1)
            im = cv2.resize(im, (SIZE,SIZE))
            im = np.divide(im, 255)

        else:
            im = images[0]
            im = cv2.resize(im, (SIZE,SIZE))
            im = np.divide(im, 255)
            im = np.expand_dims(im, 2)
        return im
    def augment(self, image):
        if randint(0,1) == 1:
            augment_img = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Flipud(0.5), # horizontal flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-180, 180),
                        shear=(-4, 4)
                    )
                ])], random_order=True)


            image_aug = augment_img.augment_image(image)
            return image_aug
        else:
            return image
    
SHAPE = (299, 299, 4)
# channels = ["red", "green", "blue"]
# for path in paths[0:10]:
#     images = []
#     for channel in channels:
#         im = np.array(Image.open(path + '_' + channel + '.png'))
# #         im = cv2.equalizeHist(im)
#         clahe = cv2.createCLAHE()
#         im = clahe.apply(im)
# #         plt.imshow(im)
#         images.append(im)

#     if len(channels) >= 2:
#         im = np.stack((
#             images
#         ), -1)
#         im = cv2.resize(im, (SIZE,SIZE))
#         im = np.divide(im, 255)
        
        
#     else:
#         im = images[0]
#         im = cv2.resize(im, (SIZE,SIZE))
#         im = np.divide(im, 255)
#         im = np.expand_dims(im, 2)
#     plt.imshow(augment(im))

# class data_generator:
    
#     def create_train(dataset_info, batch_size, shape, augument=True):
#         assert shape[2] == 3
#         while True:
#             dataset_info = shuffle(dataset_info)
#             for start in range(0, len(dataset_info), batch_size):
#                 end = min(start + batch_size, len(dataset_info))
#                 batch_images = []
#                 X_train_batch = dataset_info[start:end]
#                 batch_labels = np.zeros((len(X_train_batch), 28))
#                 for i in range(len(X_train_batch)):
#                     image = data_generator.load_image(
#                         X_train_batch[i]['path'], shape)   
#                     if augument:
#                         image = data_generator.augment(image)
#                     batch_images.append(image/255.)
#                     batch_labels[i][X_train_batch[i]['labels']] = 1
#                 yield np.array(batch_images, np.float32), batch_labels

#     def load_image(path, shape):
#         image_red_ch = Image.open(path+'_red.png')
#         image_yellow_ch = Image.open(path+'_yellow.png')
#         image_green_ch = Image.open(path+'_green.png')
#         image_blue_ch = Image.open(path+'_blue.png')
#         image = np.stack((
#         np.array(image_red_ch), 
#         np.array(image_green_ch), 
#         np.array(image_blue_ch)), -1)
#         image = cv2.resize(image, (shape[0], shape[1]))
#         return image

#     def augment(image):
#         augment_img = iaa.Sequential([
#             iaa.OneOf([
#                 iaa.Affine(rotate=0),
#                 iaa.Affine(rotate=90),
#                 iaa.Affine(rotate=180),
#                 iaa.Affine(rotate=270),
#                 iaa.Fliplr(0.5),
#                 iaa.Flipud(0.5),
#             ])], random_order=True)

#         image_aug = augment_img.augment_image(image)
#         return image_aug
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D, MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model
from keras.utils import multi_gpu_model

def create_model(input_shape, n_out, channels):
    input_tensor = Input(shape=(299,299,len(channels)))

    base_model = InceptionV3(include_top=False,
                   weights='imagenet',
                   input_shape=(299,299,3)
                            )
    bn = BatchNormalization()(input_tensor)
    x = Conv2D(3, kernel_size=(1,1), activation='relu', padding = "same")(bn)
    x = base_model(x)
    bn = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
#     output = Dense(n_out, activation='sigmoid')(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model
def simple_model(input_shape, n_out, channels):
    input_tensor = Input(shape=(299,299,len(channels)))
    bn = BatchNormalization()(input_tensor)
    x = Conv2D(8, kernel_size=(3,3), activation='relu', padding = "same")(bn)
    x = Conv2D(8, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = Conv2D(16, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = Conv2D(32, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = Conv2D(64, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = Conv2D(128, kernel_size=(3,3), activation='relu', padding = "same")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding = "valid")(x)
    x = Conv2D(256, kernel_size=(3,3), activation='relu', padding = "valid")(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
#     output = Dense(n_out, activation='sigmoid')(x)
    output = Dense(n_out, activation="sigmoid")(x)
    model = Model(input_tensor, output)
    
    return model
def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)
# create callbacks list
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

epochs = 10; batch_size = 128;VAL_RATIO = .1;DEBUG = False
# split data into train, valid
paths, labels = getTrainDataset()

# divide to 
keys = np.arange(paths.shape[0], dtype=np.int)  
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1-VAL_RATIO) * paths.shape[0])

if DEBUG == True:  # use only small subset for debugging, Kaggle's RAM is limited
    pathsTrain = paths[0:256]
    labelsTrain = labels[0:256]
    pathsVal = paths[lastTrainIndex:lastTrainIndex+256]
    labelsVal = labels[lastTrainIndex:lastTrainIndex+256]
    use_cache = True
else:
    pathsTrain = paths[0:lastTrainIndex]
    labelsTrain = labels[0:lastTrainIndex]
    pathsVal = paths[lastTrainIndex:]
    labelsVal = labels[lastTrainIndex:]
    use_cache = False

print(paths.shape, labels.shape)
print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
use_cache = True
channels = ["green", "blue", "red", "yellow"]
tg = ProteinDataGenerator(pathsTrain, labelsTrain, batch_size, SHAPE, channels, use_cache=False, augmentor = False)
vg = ProteinDataGenerator(pathsVal, labelsVal, batch_size, SHAPE, channels, use_cache=False, augmentor = False)
# create train and valid datagens
# train_generator = data_generator.create_train(
#     train_dataset_info[train_indexes], batch_size, (SIZE,SIZE,3), augument=True)
# validation_generator = data_generator.create_train(
#     train_dataset_info[valid_indexes], 32, (SIZE,SIZE,3), augument=False)
len(channels)
checkpoint = ModelCheckpoint('../working/InceptionV3.h5', monitor='val_f1', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_f1', factor=0.5, patience=10, 
                                   verbose=1, mode='max', epsilon=0.0001)
callbacks_list = [checkpoint, reduceLROnPlat]
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pt_1 = K.clip(pt_1, 1e-3, .999)
        pt_0 = K.clip(pt_0, 1e-3, .999)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
def KerasFocalLoss(target, input):
    
    gamma = 2.
    input = tf.cast(input, tf.float32)
    
    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss
    
    return K.mean(K.sum(loss, axis=1))
len([color])
"".join(colors)
history = {}
colorslist = ["green", "red", "blue", "yellow"]
from itertools import combinations
comb = combinations( colorslist, 2)
for colors in comb:
    colorset = "".join(colors)
    SHAPE = (299,299, len([colors]))
    checkpointer = ModelCheckpoint('../working/' + colorset + '.model', verbose=2, save_best_only=True)
    model = simple_model(
    input_shape=(299,299,len(colors)), 
    n_out=28, channels = colors)

    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam',
        metrics=['acc', f1])

    model.summary()
    tg = ProteinDataGenerator(pathsTrain, labelsTrain, batch_size, SHAPE, colors, use_cache=False, augmentor = False)
    vg = ProteinDataGenerator(pathsVal, labelsVal, batch_size, SHAPE, colors, use_cache=False, augmentor = False)
    # train model
    history[colorset] = model.fit_generator(
        tg,
        steps_per_epoch=(len(pathsTrain)//batch_size)/8,
        validation_data=vg,
        validation_steps=(len(pathsVal)//batch_size)/8,
        epochs=5, 
        verbose=1,
        callbacks=[checkpointer])
comb = combinations( ['red', 'green', 'blue', 'yellow'], 2)
for colors in comb:
    colorset = "".join(colors)
    history1 = history[colorset]
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title(colorset + 'loss')
    ax[0].plot(history1.epoch, history1.history["loss"], label="Train loss")
    ax[0].plot(history1.epoch, history1.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history1.epoch, history1.history["acc"], label="Train acc")
    ax[1].plot(history1.epoch, history1.history["val_acc"], label="Validation acc")
    ax[2].set_title('F1')
    ax[2].plot(history1.epoch, history1.history["f1"], label="Train F1")
    ax[2].plot(history1.epoch, history1.history["val_f1"], label="Validation F1")
    ax[0].legend()
    _ = ax[1].legend()
