# tensorflow
import tensorflow as tf

# Keras modules
import keras
from keras.callbacks import ModelCheckpoint
from keras import backend
# from tensorflow.keras.applications import DenseNet --> Discarded due to too much memory usage
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import MobileNet
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import Xception
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import Sequence

# data processing modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# plotting
import matplotlib.pyplot as plt

# image processing
from PIL import Image
from imgaug import augmenters as iaa

# python support libraries
import os
import datetime
from zipfile import ZipFile
from collections import Iterable
import warnings
warnings.filterwarnings("ignore")
def get_classes(x):
    """
    Transform the Target column from the train dataset into an array of classes found in the image
    to be later used as a target variable
    :param x: row value for column Target from apply function
    :return: array of length 27 with 0 and 1
    """
    return np.array([0 if str(i) not in str(x).split(' ') else 1 for i in range(28)])


def load_image(file_path, image_name, shape=(512, 512, 3)):
    """
    Load image contained within a zip file
    :param file_path:  (string) Path to image on disk
    :param image_name: (string) Image name contained in zip file
    :param shape:      (tuple) Shape of image to be outputed
    :return:           (array) 3D of RGBY divided by 255
    """
    # load images by channel
    channel_list = list()
    for c in ['red', 'green', 'blue', 'yellow']:
        img = Image.open(file_path + '/' + image_name + '_' + c + '.png')
        img = img.resize((shape[0], shape[1]), Image.ANTIALIAS)
        img = np.array(img)
        channel_list.append(img)
    
    # stack pixels of image
    if shape[2] == 3:
        image = np.stack((
            channel_list[0]/2 + channel_list[3]/2, 
            channel_list[1]/2 + channel_list[3]/2, 
            channel_list[2]
        ),-1)
    else:
        image = np.stack(channel_list, -1)

    # normalize pixels range
    image = np.divide(image, 255)

    # return array with normalized colors
    return image


def f1(y_true, y_pred):
    """
    Calculate the f1 score given the true values and predictions
    :param y_true: (array) true value array
    :param y_pred: (array) predictions array
    :return: (float) f1 score
    """
    tp = backend.sum(backend.cast(y_true * y_pred, 'float'), axis=0)
    fp = backend.sum(backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2 * p * r / (p + r + backend.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return backend.mean(f1)


def augment(image):
    """
    Apply transformations to images
    source: https://www.kaggle.com/rejpalcz/cnn-128x128x4-keras-from-scratch-lb-0-328
    :param image:
    :return:
    """
    augment_img = iaa.Sequential([
        iaa.OneOf([
            # flip the image
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),

            # random crops
            iaa.Crop(percent=(0, 0.1)),

            # Strengthen or weaken the contrast in each image
            iaa.ContrastNormalization((0.75, 1.5)),

            # Add gaussian noise
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

            # Make some images brighter and some darker
            iaa.Multiply((0.8, 1.2), per_channel=0.2),

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-180, 180),
                shear=(-8, 8)
            )
        ])], random_order=True)

    image_aug = augment_img.augment_image(image)
    return image_aug


def show_history(history):
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


def sequencial_model(input_shape, n_out):
    # Initialising the CNN
    model = Sequential()

    # ##### LAYER 1
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # ##### LAYER 2
    # model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # ##### FLATTENING
    model.add(Flatten())

    # ##### ANN
    model.add(Dense(activation='relu', units=1024))
    model.add(Dense(activation='relu', units=128))
    model.add(Dense(activation='softmax', units=n_out))

    return model


def vgg_model(input_shape, n_out):
    model = VGG19(
        include_top=False, weights='imagenet', input_shape=input_shape
    )

    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = model(bn)
    x = Conv2D(128, kernel_size=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    return Model(input_tensor, output)


def inception_res_net_model(input_shape, n_out):    
    pretrain_model = InceptionResNetV2(
        include_top=False, 
        weights='imagenet', 
        input_shape=input_shape
    )    
    
    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    
    return model
class DataGenerator(keras.utils.Sequence):
    """
    Extend the Sequence class from the keras.utils module to create a class
    capable of loading the images from the zipfile in batches, resizing it and
    selecting specific channels

    source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, train_data, batch_size, file_path, shape, augment_flag=True):
        assert isinstance(train_data, pd.DataFrame)
        assert train_data.shape[0] > 0
        assert batch_size > 0
        assert (os.path.isfile(file_path) and '.zip' in file_path) or os.path.isdir(file_path)
        assert isinstance(shape, Iterable)
        assert len(shape) == 3
        assert type(augment_flag) == bool

        # saved arguments
        self.train_data = train_data
        self.batch_size = batch_size
        self.file_path = file_path
        self.shape = shape
        self.augment_flag = augment_flag

        # get the number of images in the train dataset
        self.size = train_data.shape[0]

        # get list of images
        self.image_ids = train_data['Id'].values

        # set a list of indexes to be extracted from the image
        self.indexes = np.arange(len(self.image_ids))

        # calculate the required number of batches
        self.batches = int(np.floor(self.size / batch_size))

    def __len__(self):
        return self.batches

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: (int) index of the total batches size to be loaded
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # select images ids
        batch_ids = [self.image_ids[k] for k in indexes]

        # create array to hold images
        batch_images = np.empty((self.batch_size, self.shape[0], self.shape[1], self.shape[2]))
        batch_labels = np.zeros((self.batch_size, 28))

        # load images into array
        for i in range(self.batch_size):
            # apply transformations to images based on augment flag
            if self.augment_flag:
                batch_images[i] = augment(self.__load_image(batch_ids[i]))
            else:
                batch_images[i] = self.__load_image(batch_ids[i])
            batch_labels[i] = self.train_data.loc[self.train_data['Id'] == batch_ids[i], 'Classes'].values[0]

        # return the images batch
        return batch_images, batch_labels

    def __iter__(self):
        """
        Create a generator that iterate over the Sequence
        """
        for i in range(self.batches):
            yield self[i]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.image_ids))
        np.random.shuffle(self.indexes)

    def __load_image(self, image_name):
        """
        Load image contained within a zip file
        :param image_name: (string) image name contained in zip file
        :return:           (array) 3D of RGBY divided by 255
        """
        if '.zip' in self.file_path:
            # open the zipfile
            with ZipFile(self.file_path) as z:
                # load images by channel
                channel_list = list()
                for c in ['red', 'green', 'blue', 'yellow']:
                    with z.open(image_name + '_' + c + '.png') as file:
                        img = Image.open(file)
                        img = img.resize((self.shape[0], self.shape[1]), Image.ANTIALIAS)
                        img = np.array(img)
                        channel_list.append(img)

        else:
            # load images by channel
            channel_list = list()
            for c in ['red', 'green', 'blue', 'yellow']:
                img = Image.open(self.file_path + '/' + image_name + '_' + c + '.png')
                img = img.resize((self.shape[0], self.shape[1]), Image.ANTIALIAS)
                img = np.array(img)
                channel_list.append(img)
        
        # stack pixels of image
        if self.shape[2] == 3:
            image = np.stack((
                channel_list[0]/2 + channel_list[3]/2, 
                channel_list[1]/2 + channel_list[3]/2, 
                channel_list[2]
            ),-1)
        else:
            image = np.stack(channel_list, -1)
        
        # normalize pixels range
        image = np.divide(image, 255)
        
        # return array with normalized colors
        return image
INPUT_SHAPE = (299, 299, 3)
TRAIN_BATCH = 10
TEST_BATCH = 256
TEST_SIZE = 0.2
C_MIN = 5
RANDOM_STATE = 42
STEPS_PER_EPOCH = 100
EPOCHS = 15
VALIDATION_STEPS = 50
VERBOSE = 1
VERBOSE_CK = 2
MODEL_NAME = 'InceptionResNet'
keras.backend.clear_session()

print('DEFINING CNN MODEL')
if MODEL_NAME == 'Sequential':
    model = sequencial_model(INPUT_SHAPE, 28)
elif MODEL_NAME == 'InceptionResNet':
    model = inception_res_net_model(INPUT_SHAPE, 28)
elif MODEL_NAME == 'VGG19':
    model = vgg_model(INPUT_SHAPE, 28)

# compile model
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy', f1])

# model.layers[2].trainable = False

# export model summary
model.summary()
print('LOADING TRAIN CSV FILE')
train_csv = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/train.csv')

print('OBTAINING CLASSES ARRAY')
train_csv['Classes'] = train_csv['Target'].apply(get_classes)

print('EVENLY DIVIDING CLASSES')
# count classes
vc = train_csv['Target'].value_counts()

# create train and test dataframes
train_df = pd.DataFrame()
test_df = pd.DataFrame()

print('    select well represented classes'.upper())
for target in vc[vc > C_MIN].index:
    # filter images
    images = train_csv[train_csv['Target'] == target]

    # apply train test split
    X_train, X_test, y_train, y_test = train_test_split(
        images['Id'].values.flatten(), images['Classes'].values.flatten(),
        test_size=TEST_SIZE#, random_state=RANDOM_STATE
    )

    # add to train and test dataframes
    train_df = train_df.append(pd.DataFrame(data={'Id': X_train, 'Classes': y_train}))
    test_df = test_df.append(pd.DataFrame(data={'Id': X_test, 'Classes': y_test}))

print('    select low represented classes'.upper())
pool = train_csv[train_csv['Target'].isin(vc[vc <= C_MIN].index)]

# go through each class
for target in np.argsort(train_csv['Classes'].sum()):
    # filter pool
    images = pool[pool['Classes'].apply(lambda x: x[target] == 1)]
    
    if images.shape[0] == 0:
        continue
        
    # apply train test split
    X_train, X_test, y_train, y_test = train_test_split(
        images['Id'].values.flatten(), images['Classes'].values.flatten(),
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # add to train and test dataframes
    train_df = train_df.append(pd.DataFrame({'Id': X_train, 'Classes': y_train}))
    test_df = test_df.append(pd.DataFrame({'Id': X_test, 'Classes': y_test}))

    # remove from pool
    pool = pool[~pool['Id'].isin(images['Id'])]

print('CREATING DATA GENERATOR')
train_gen = DataGenerator(train_df, TRAIN_BATCH, '/kaggle/input/human-protein-atlas-image-classification/train', INPUT_SHAPE)
test_gen = DataGenerator(test_df, TEST_BATCH, '/kaggle/input/human-protein-atlas-image-classification/test', INPUT_SHAPE, augment_flag=False)
print('CREATING MODEL CHECK POINT')
dt = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
model_path = '/kaggle/working/' + MODEL_NAME + dt + '.model'
checkpoint = ModelCheckpoint(model_path, verbose=VERBOSE_CK, save_best_only=True)

print('FITTING MODEL')
hist = model.fit_generator(
    generator=train_gen, 
    validation_data=test_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS, 
    validation_steps=VALIDATION_STEPS, 
    verbose=VERBOSE,
    callbacks=[checkpoint]
)
show_history(hist)
model_path = '/kaggle/working/' + MODEL_NAME + dt + '.model'
model = load_model(
    model_path, 
    custom_objects={'f1': f1}
)
submit = pd.read_csv('../input/sample_submission.csv')
%%time
predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = DataGenerator(train_data, TRAIN_BATCH, '/kaggle/input/train/', INPUT_SHAPE)
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)