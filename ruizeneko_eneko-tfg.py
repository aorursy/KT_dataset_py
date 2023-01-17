! /opt/conda/bin/python3.7 -m pip install --upgrade pip
! pip install -U tensorflow==2.2.0
! pip install -U keras==2.4.2
! pip install -U segmentation-models==1.0.0
! pip install keras-rectified-adam
! pip install albumentations
import segmentation_models as sm
from segmentation_models import get_preprocessing
from keras_radam.training import RAdamOptimizer
class DualTransform:

    identity_param = None

    def prepare(self, params):
        if isinstance(params, tuple):
            params = list(params)
        elif params is None:
            params = []
        elif not isinstance(params, list):
            params = [params]

        if not self.identity_param in params:
            params.append(self.identity_param)
        return params

    def forward(self, image, param):
        raise NotImplementedError

    def backward(self, image, param):
        raise NotImplementedError


class SingleTransform(DualTransform):

    def backward(self, image, param):
        return image


class HFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, image, param):
        return tf.image.flip_left_right(image) if param else image

    def backward(self, image, param):
        return self.forward(image, param)


class VFlip(DualTransform):

    identity_param = 0

    def prepare(self, params):
        if params == False:
            return [0]
        if params == True:
            return [1, 0]

    def forward(self, image, param):
        return tf.image.flip_up_down(image) if param else image

    def backward(self, image, param):
        return self.forward(image, param)


class Rotate(DualTransform):

    identity_param = 0

    def forward(self, image, angle):
        k = angle // 90 if angle >= 0 else (angle + 360) // 90
        return tf.image.rot90(image, k)

    def backward(self, image, angle):
        return self.forward(image, -angle)


class HShift(DualTransform):

    identity_param = 0

    def forward(self, image, param):
        return tf.roll(image, param, axis=0)

    def backward(self, image, param):
        return tf.roll(image, -param, axis=0)


class VShift(DualTransform):

    identity_param = 0

    def forward(self, image, param):
        return tf.roll(image, param, axis=1)

    def backward(self, image, param):
        return tf.roll(image, -param, axis=1)


class Contrast(SingleTransform):

    identity_param = 1

    def forward(self, image, param):
        return tf.image.adjust_contrast(image, param)


class Add(SingleTransform):

    identity_param = 0

    def forward(self, image, param):
        return image + param


class Multiply(SingleTransform):

    identity_param = 1

    def forward(self, image, param):
        return image * param


def gmean(x):
    g_pow = 1 / x.get_shape().as_list()[0]
    x = tf.reduce_prod(x, axis=0, keepdims=True)
    x = tf.pow(x, g_pow)
    return x


def mean(x):
    return tf.reduce_mean(x, axis=0, keepdims=True)


def tta_max(x):
    return tf.reduce_max(x, axis=0, keepdims=True)
import itertools

class Augmentation(object):

    transforms = {
        'h_flip': HFlip(),
        'v_flip': VFlip(),
        'rotation':Rotate(),
        'h_shift': HShift(),
        'v_shift': VShift(),
        'contrast':Contrast(),
        'add':Add(),
        'mul':Multiply(),
    }

    def __init__(self, **params):
        super().__init__()

        transforms = [Augmentation.transforms[k] for k in params.keys()]
        transform_params = [params[k] for k in params.keys()]

        # add identity parameters for all transforms and convert to list
        transform_params = [t.prepare(params) for t, params in zip(transforms, transform_params)]

        # get all combinations of transforms params
        transform_params = list(itertools.product(*transform_params))

        self.forward_aug = [t.forward for t in transforms]
        self.forward_params = transform_params

        self.backward_aug = [t.backward for t in transforms[::-1]] # reverse transforms
        self.backward_params = [p[::-1] for p in transform_params] # reverse params

        self.n_transforms = len(transform_params)

    @property
    def forward(self):
        return self.forward_aug, self.forward_params

    @property
    def backward(self):
        return self.backward_aug, self.backward_params
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Repeat(Layer):
    """
    Layer for cloning input information
    input_shape = (1, H, W, C)
    output_shape = (N, H, W, C)
    """
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, x):
        return tf.stack([x[0]] * self.n, axis=0)

    def compute_output_shape(self, input_shape):
        return (self.n, *input_shape[1:])


class TTA(Layer):

    def __init__(self, functions, params):
        super().__init__()
        self.functions = functions
        self.params = params

    def apply_transforms(self, images):
        transformed_images = []
        for i, args in enumerate(self.params):
            image = images[i]
            for f, arg in zip(self.functions, args):
                image = f(image, arg)
            transformed_images.append(image)
        return tf.stack(transformed_images, 0)

    def call(self, images):
        return self.apply_transforms(images)


class Merge(Layer):

    def __init__(self, type):
        super().__init__()
        self.type = type

    def merge(self, x):
        if self.type == 'mean':
            return mean(x)
        if self.type == 'gmean':
            return gmean(x)
        if self.type == 'tta_max':
            return tta_max(x)
        else:
            raise ValueError(f'Wrong merge type {type}')

    def call(self, x):
        return self.merge(x)

    def compute_output_shape(self, input_shape):
        return (1, *input_shape[1:])
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input



doc = """
    IMPORTANT constraints:
        1) model has to have 1 input and 1 output
        2) inference batch_size = 1
        3) image height == width if rotate augmentation is used
    Args:
        model: instance of Keras model
        h_flip: (bool) horizontal flip
        v_flip: (bool) vertical flip
        h_shifts: (list of int) list of horizontal shifts (e.g. [10, -10])
        v_shifts: (list of int) list of vertical shifts (e.g. [10, -10])
        rotation: (list of int) list of angles (deg) for rotation in range [0, 360),
            should be divisible by 90 deg (e.g. [90, 180, 270])
        contrast: (list of float) values for contrast adjustment
        add: (list of int or float) values to add on image (e.g. [-10, 10])
        mul: (list of float) values to multiply image on (e.g. [0.9, 1.1])
        merge: one of 'mean', 'gmean' and 'max' - mode of merging augmented
            predictions together.
    Returns:
        Keras Model instance
"""

def segmentation(
    model,
    h_flip=False,
    v_flip=False,  
    h_shift=None,
    v_shift=None,
    rotation=None,
    contrast=None,
    add=None,
    mul=None,
    merge='mean',
    input_shape=None,
):
    """
    Segmentation model test time augmentation wrapper.
    """
    tta = Augmentation(
        h_flip=h_flip,
        v_flip=v_flip,
        h_shift=h_shift,
        v_shift=v_shift,
        rotation=rotation,
        contrast=contrast,
        add=add,
        mul=mul,
    )

    if input_shape is None:
        try:
            input_shape = model.input_shape[1:]
        except AttributeError:
            raise AttributeError(
                'Can not determine input shape automatically, please provide `input_shape` '
                'argument to wrapper (e.g input_shape=(None, None, 3)).'
            )
    batch_shape = (1, *input_shape) # add batch dimension

    inp = Input(batch_shape=batch_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = TTA(*tta.backward)(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


def classification(
    model,
    h_flip=False,
    v_flip=False,
    h_shift=None,
    v_shift=None,
    rotation=None,
    contrast=None,
    add=None,
    mul=None,
    merge='mean',
    input_shape=None,
):
    """
    Classification model test time augmentation wrapper.
    """

    tta = Augmentation(
        h_flip=h_flip,
        v_flip=v_flip,
        h_shift=h_shift,
        v_shift=v_shift,
        rotation=rotation,
        contrast=contrast,
        add=add,
        mul=mul,
    )
    
    if input_shape is None:
        try:
            input_shape = model.input_shape[1:]
        except AttributeError:
            raise AttributeError(
                'Can not determine input shape automatically, please provide `input_shape` '
                'argument to wrapper (e.g input_shape=(None, None, 3)).'
            )
    batch_shape = (1, *input_shape) # add batch dimension

    inp = Input(batch_shape=batch_shape)
    x = Repeat(tta.n_transforms)(inp)
    x = TTA(*tta.forward)(x)
    x = model(x)
    x = Merge(merge)(x)
    tta_model = Model(inp, x)

    return tta_model


classification.__doc__ += doc
segmentation.__doc__ += doc

# legacy support
tta_classification = classification
tta_segmentation = segmentation
import os
import numpy as np
import pandas as pd
import pathlib
import cv2 as cv
import copy
import shutil
import time
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import albumentations as albu
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import math
import os

import tensorflow as tf

import cProfile
print(tf.__version__)
tf.executing_eagerly()

PROJECT_PATH = '../input/understanding_cloud_organization'
# First, we load the DF
train = pd.read_csv(PROJECT_PATH + "/" + "train.csv")

# We get images shape
img = cv.imread(PROJECT_PATH + '/train_images/' + '04df149.jpg', -1)
HEIGHT = img.shape[0]
WIDTH = img.shape[1]
DIMENSIONS = (HEIGHT, WIDTH)


train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

# Create one column for each mask
train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']
corrected_df = train_df
def get_train_val(path):
    """
    Extract the directory under study
    Parameters:
        path : whole path of the directory
    Return:
        string: the directory we are creating (train/val/test)
    """
    sub_string = path.split('/')[3]
    return sub_string.split('_')[0].capitalize()


def check_if_exist(list_files, path):
    """
    Check if N (here N = 50) images exist in the train/test/val directory. If they exist, do not move them again. If
    they do not, they move.
    Parameters:
        list_files: the list of files to check
        path: where the files are
    Return:
        boolean: if all the files exist or not
    """
    for i in range(len(list_files)):
        if os.path.isdir('/kaggle/working/' + path):
            if not os.path.isfile('/kaggle/working/' + path + '/' + list_files[i]):
                print("Files do not match. Creating {} directory...".format(get_train_val('/kaggle/working/' + path)))
                return False
    print('Files do exist in {} directory.'.format(get_train_val('/kaggle/working/' + path)))
    return True


def get_resize_image(img_name, shape, test_train):
    """
    Resizes and changes from BGR to RGB an image opened by OpenCV
    Parameters:
        img_name: the name of the image
        shape: the shape of the future image
    Returns:
        numpy-array: the img converted
    """
    if test_train == 'train':
        img = cv.imread(PROJECT_PATH + '/train_images/' + img_name)
    elif test_train =='test':
        img = cv.imread(PROJECT_PATH + '/test_images/' + img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resized_img = cv.resize(img, shape)
    return resized_img

def rle_decode(mask_rle, shape=(1400, 2100)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction

def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv.resize(img, (width, height))


def move_data(image_df, path, HEIGHT=384, WIDTH=480, test_train='train'):
    """
    Creates an image in a given directory
    Parameters:
        image_list: the list of images
        path: where the images are
        HEIGHT: the dimensions of the image
        WIDTH: the dimensions of the image
    """
    for i in range(image_df.shape[0]):
        item = image_df.iloc[i]
        img_name = item['image']
        image = get_resize_image(img_name,  (WIDTH, HEIGHT), test_train)
        cv.imwrite(path + img_name, image)

        
def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv.resize(img, (width, height))        


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    mask = np.zeros( width*height ).astype(np.uint8)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T


def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle2mask(rle, input_shape)
            else:
                mask = rle2mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask
    
    return masks

def dice_coefficient(y_true, y_pred):
    """The metrics.
    For further information refer to: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Parameters:
        y_true: true label
        y_pred: predicted label
    Returns
        double: the result
    """
    y_true = np.asarray(y_true).astype(np.bool)
    y_pred = np.asarray(y_pred).astype(np.bool)
    intersection = np.logical_and(y_true, y_pred)
    return (2. * intersection.sum()) / (y_true.sum() + y_pred.sum())


def dice_coef(y_true, y_pred, smooth=1):
    """The metrics.
    For further information refer to: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Parameters:
        y_true: true label
        y_pred: predicted label
        smooth: smooth of the metric
    Returns
        double: the result
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Model evaluation
def plot_metrics(history, metric_list=['loss', 'dice_coef'], figsize=(22, 14)):
    fig, axes = plt.subplots(len(metric_list), 1, sharex='col', figsize=(22, len(metric_list)*4))
    axes = axes.flatten()
    
    for index, metric in enumerate(metric_list):
        axes[index].plot(history[metric], label='Train %s' % metric)
        axes[index].plot(history['val_%s' % metric], label='Validation %s' % metric)
        axes[index].legend(loc='best')
        axes[index].set_title(metric)

    plt.xlabel('Epochs')
    sns.despine()
    plt.show()
    
# Model post process
def post_process(probability, threshold=0.5, min_size=10000):
    mask = cv.threshold(probability, threshold, 1, cv.THRESH_BINARY)[1]
    num_component, component = cv.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros(probability.shape, np.float32)
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return predictions
    
    
    # Prediction evaluation
def get_metrics(model, target_df, df, df_images_dest_path, tresholds, min_mask_sizes, N_CLASSES=4, seed=0, preprocessing=None, set_name='Complete set'):
    class_names = ['Fish', 'Flower', 'Gravel', 'Sugar']
    metrics = []

    for class_name in class_names:
        metrics.append([class_name, 0, 0])

    metrics_df = pd.DataFrame(metrics, columns=['Class', 'Dice', 'Dice Post'])
    
    for i in range(0, df.shape[0], 200):
        batch_idx = list(range(i, min(df.shape[0], i + 200)))
        batch_set = df[batch_idx[0]: batch_idx[-1]+1]
        ratio = len(batch_set) / len(df)

        generator = DataGenerator(
                      directory=df_images_dest_path,
                      dataframe=batch_set,
                      target_df=target_df,
                      batch_size=len(batch_set), 
                      target_size=model.input_shape[1:3],
                      n_channels=model.input_shape[3],
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      mode='fit',
                      shuffle=False)

        x, y = generator.__getitem__(0)
        preds = model.predict(x)
        
        for class_index in range(N_CLASSES):
            class_score = []
            class_score_post = []
            mask_class = y[..., class_index]
            pred_class = preds[..., class_index]
            for index in range(len(batch_idx)):
                sample_mask = mask_class[index, ]
                sample_pred = pred_class[index, ]
                sample_pred_post = post_process(sample_pred, threshold=tresholds[class_index], min_size=min_mask_sizes[class_index])
                if (sample_mask.sum() == 0) & (sample_pred.sum() == 0):
                    dice_score = 1.
                else:
                    dice_score = dice_coefficient(sample_pred, sample_mask)
                if (sample_mask.sum() == 0) & (sample_pred_post.sum() == 0):
                    dice_score_post = 1.
                else:
                    dice_score_post = dice_coefficient(sample_pred_post, sample_mask)
                class_score.append(dice_score)
                class_score_post.append(dice_score_post)
            metrics_df.loc[metrics_df['Class'] == class_names[class_index], 'Dice'] += np.mean(class_score) * ratio
            metrics_df.loc[metrics_df['Class'] == class_names[class_index], 'Dice Post'] += np.mean(class_score_post) * ratio

    metrics_df = metrics_df.append({'Class':set_name, 'Dice':np.mean(metrics_df['Dice'].values), 'Dice Post':np.mean(metrics_df['Dice Post'].values)}, ignore_index=True).set_index('Class')
    
    return metrics_df

def inspect_predictions(df, image_ids, images_dest_path, pred_col=None, label_col='EncodedPixels', title_col='Image_Label', img_shape=(525, 350), figsize=(22, 6)):
    if pred_col:
        for sample in image_ids:
            sample_df = df[df['image'] == sample]
            fig, axes = plt.subplots(2, 5, figsize=figsize)
            img = cv.imread(images_dest_path + sample_df['image'].values[0])
            img = cv.resize(img, img_shape)
            axes[0][0].imshow(img)
            axes[1][0].imshow(img)
            axes[0][0].set_title('Label', fontsize=16)
            axes[1][0].set_title('Predicted', fontsize=16)
            axes[0][0].axis('off')
            axes[1][0].axis('off')
            for i in range(4):
                mask = sample_df[label_col].values[i]
                try:
                    math.isnan(mask)
                    mask = np.zeros((img_shape[1], img_shape[0]))
                except:
                    mask = rle_decode(mask)
                axes[0][i+1].imshow(mask)
                axes[1][i+1].imshow(rle2mask(sample_df[pred_col].values[i], img.shape))
                axes[0][i+1].set_title(sample_df[title_col].values[i], fontsize=18)
                axes[1][i+1].set_title(sample_df[title_col].values[i], fontsize=18)
                axes[0][i+1].axis('off')
                axes[1][i+1].axis('off')
    else:
        for sample in image_ids:
            sample_df = df[df['image'] == sample]
            fig, axes = plt.subplots(1, 5, figsize=figsize)
            img = cv.imread(images_dest_path + sample_df['image'].values[0])
            img = cv.resize(img, img_shape)
            axes[0].imshow(img)
            axes[0].set_title('Original', fontsize=16)
            axes[0].axis('off')
            for i in range(4):
                axes[i+1].imshow(rle2mask(sample_df[label_col].values[i], img.shape))
                axes[i+1].set_title(sample_df[title_col].values[i], fontsize=18)
                axes[i+1].axis('off')

            
def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def build_rles(masks, reshape=None):
    width, height, depth = masks.shape
    rles = []
    
    for i in range(depth):
        mask = masks[:, :, i]
        
        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)
        
        rle = mask2rle(mask)
        rles.append(rle)
        
    return rles
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, directory, batch_size, n_channels, target_size,  n_classes, 
                 mode='fit', target_df=None, shuffle=True, preprocessing=None, augmentation=None, seed=0):
        
        self.batch_size = batch_size
        self.dataframe = dataframe
        self.mode = mode
        self.directory = directory
        self.target_df = target_df
        self.target_size = target_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.seed = seed
        self.mask_shape = (1400, 2100)
        self.list_IDs = self.dataframe.index
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        X = self.__generate_X(list_IDs_batch)
        
        if self.mode == 'fit':
            Y = self.__generate_Y(list_IDs_batch)
            
            if self.augmentation:
                X, Y = self.__augment_batch(X, Y)
            
            return X, Y
        
        elif self.mode == 'predict':
            return X
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __generate_X(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.target_size, self.n_channels))
        
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            img_path = self.directory + img_name
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if self.preprocessing:
                img = self.preprocessing(img)
                
            X[i,] = img

        return X
    
    def __generate_Y(self, list_IDs_batch):
        Y = np.empty((self.batch_size, *self.target_size, self.n_classes), dtype=int)
        
        for i, ID in enumerate(list_IDs_batch):
            img_name = self.dataframe['image'].loc[ID]
            image_df = self.target_df[self.target_df['image'] == img_name]
            rles = image_df['EncodedPixels'].values
            masks = build_masks(rles, input_shape=self.mask_shape, reshape=self.target_size)
            Y[i, ] = masks

        return Y
    
    def __augment_batch(self, X_batch, Y_batch):
        for i in range(X_batch.shape[0]):
            X_batch[i, ], Y_batch[i, ] = self.__random_transform(X_batch[i, ], Y_batch[i, ])
        
        return X_batch, Y_batch
    
    def __random_transform(self, X, Y):
        composed = self.augmentation(image=X, mask=Y)
        X_aug = composed['image']
        Y_aug = composed['mask']
        
        return X_aug, Y_aug
now = time.time()

# We create a copy from our already loaded and preprocessed DT
DF = copy.deepcopy(corrected_df)

# Why deep copy instead of assigment operator ?
# We will shuffle the list -->  https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/

# We split into train and validation sets, adding an additional column to distinguish them
X_train, X_val = train_test_split(DF, test_size=0.2, random_state=1997)

X_train.insert(0, 'Set', ['Train' for _ in range(len(X_train))], True)
X_train.reset_index()
random_select_train = np.random.choice(X_train['image'].values, 50)

X_val.insert(0, 'Set', ['Val' for _ in range(len(X_val))], True)
X_val.reset_index()
random_select_val = np.random.choice(X_val['image'].values, 50)

size_train = len(X_train)
size_val = len(X_val)

print('# train examples {}'.format(size_train))
print('# val examples {}'.format(size_val))
print(time.time() - now)

# We create a distinct folders for both train and validation
# TODO: put this in a different file
# TODO: do the same without creating the folders
train_images_path = '/kaggle/working/train_images_dict/'
validation_images_path = '/kaggle/working/val_images_dict/'

if not os.path.exists(validation_images_path):
    os.makedirs(validation_images_path)
    
if not check_if_exist(random_select_val, 'val_images_dict'):
    if os.path.exists(validation_images_path):
        shutil.rmtree(validation_images_path)

    os.makedirs(validation_images_path)
    try:
        move_data(X_val, validation_images_path)
        print('moved and created')
        print(str((time.time() - now)/60.0) + " minutes")    
    except IndexError:
        shutil.rmtree(validation_images_path)


# Since directories should be created only once
if not os.path.exists(train_images_path):
    os.makedirs(train_images_path)

if not check_if_exist(random_select_train, 'train_images_dict'):
    if os.path.exists(train_images_path):
        shutil.rmtree(train_images_path)

    os.makedirs(train_images_path)
    try:
        move_data(X_train, train_images_path)
        print('moved and created')
        print(str((time.time() - now)/60.0) + " minutes")    
    except IndexError:
        shutil.rmtree(train_images_path)
BACKBONE = 'resnet152'
BATCH_SIZE = 14
EPOCHS = 15
LEARNING_RATE = 1e-4
HEIGHT = 384
WIDTH = 480
CHANNELS = 3
N_CLASSES = 4
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5
model_path = 'uNet_%s_%sx%s.h5' % (BACKBONE, HEIGHT, WIDTH)
# We add augmentation and preprocessing
preprocessing = sm.get_preprocessing(BACKBONE)

augmentation = albu.Compose([albu.HorizontalFlip(),
                             albu.VerticalFlip(),
                             albu.ShiftScaleRotate(rotate_limit=30, shift_limit=0.1),
                             ])

train_generator = DataGenerator(
                  directory=train_images_path,
                  dataframe=X_train,
                  target_df=train,
                  batch_size=BATCH_SIZE,
                  target_size=(HEIGHT, WIDTH),
                  n_channels=CHANNELS,
                  n_classes=N_CLASSES,
                  augmentation=augmentation,
                  preprocessing=preprocessing)

valid_generator = DataGenerator(
                  directory=validation_images_path,
                  dataframe=X_val,
                  target_df=train,
                  batch_size=BATCH_SIZE, 
                  target_size=(HEIGHT, WIDTH),
                  n_channels=CHANNELS,
                  n_classes=N_CLASSES,
                  preprocessing=preprocessing)

model = sm.Unet(backbone_name=BACKBONE,
                encoder_weights='imagenet',
                classes=4,
                activation='sigmoid',
                input_shape=(HEIGHT, WIDTH, 3))
class OneCycleScheduler(tf.keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)
onecycle = OneCycleScheduler(len(X_train) // BATCH_SIZE * EPOCHS, max_rate=0.05)
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)

metric_list = [dice_coef, sm.metrics.iou_score]
callback_list = [checkpoint, es, onecycle]


optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9,nesterov=True)
model.compile(optimizer, loss=sm.losses.bce_dice_loss, metrics=metric_list)
model.summary()
STEP_SIZE_TRAIN = len(X_train) // BATCH_SIZE
STEP_SIZE_VALID = len(X_val) // BATCH_SIZE

history = model.fit(
    train_generator,  
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks=callback_list,
    epochs=EPOCHS,
    verbose=1)
import numpy as np
max(2,5)
print(str((time.time() - now)/60.0) + " minutes")    
plot_metrics(history.history, metric_list=['loss', 'dice_coef', 'iou_score'])
class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
best_tresholds = [.5, .5, .5, .35]
best_masks = [25000, 20000, 22500, 15000]

for index, name in enumerate(class_names):
    print('%s treshold=%.2f mask size=%d' % (name, best_tresholds[index], best_masks[index]))
train_metrics = get_metrics(model, train, X_train, train_images_path, best_tresholds, best_masks, preprocessing=preprocessing, set_name='Train')
display(train_metrics)

print(str((time.time() - now)/60.0) + " minutes")    

validation_metrics = get_metrics(model, train, X_val, validation_images_path, best_tresholds, best_masks, preprocessing=preprocessing, set_name='Validation')
display(validation_metrics)
print(str((time.time() - now)/60.0) + " minutes")    

model = segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')
submission = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')
submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
test = pd.DataFrame(submission['image'].unique(), columns=['image'])

random_select_test = np.random.choice(test['image'].values, 30)
test_images_path = '/kaggle/working/test_images_dict/'
if not os.path.exists(test_images_path):
    os.makedirs(test_images_path)
    
if not check_if_exist(random_select_test, 'test_images_dict'):
    if os.path.exists(test_images_path):
        shutil.rmtree(test_images_path)

    os.makedirs(test_images_path)
    try:
        move_data(test, test_images_path, test_train='test')
        print('created')
        print(str((time.time() - now)/60.0) + " minutes")
    except IndexError:
        shutil.rmtree(test_images_path)
test_df = []

for i in range(0, test.shape[0], 300):
    batch_idx = list(range(i, min(test.shape[0], i + 300)))
    batch_set = test[batch_idx[0]: batch_idx[-1]+1]
    
    test_generator = DataGenerator(
                      directory=test_images_path,
                      dataframe=batch_set,
                      target_df=submission,
                      batch_size=1, 
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      mode='predict',
                      shuffle=False)
    
    model.run_eagerly = True
    preds = model.predict_generator(test_generator)

    for index, b in enumerate(batch_idx):
        filename = test['image'].iloc[b]
        image_df = submission[submission['image'] == filename].copy()
        pred_masks = preds[index, ].round().astype(int)
        pred_rles = build_rles(pred_masks, reshape=(350, 525))
        image_df['EncodedPixels'] = pred_rles

        ### Post procecssing
        pred_masks_post = preds[index, ].astype('float32') 
        for class_index in range(N_CLASSES):
            pred_mask = pred_masks_post[...,class_index]
            pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
            pred_masks_post[...,class_index] = pred_mask

        pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
        image_df['EncodedPixels_post'] = pred_rles_post
        ###
        
        test_df.append(image_df)

sub_df = pd.concat(test_df)
print(str((time.time() - now)/60.0) + " minutes")    

# Choose 3 samples at random
images_to_inspect = np.random.choice(X_val['image'].unique(), 3, replace=False)
inspect_set = train[train['image'].isin(images_to_inspect)].copy()
inspect_set_temp = []

inspect_generator = DataGenerator(
                    directory=validation_images_path,
                    dataframe=inspect_set,
                    target_df=train,
                    batch_size=1, 
                    target_size=(HEIGHT, WIDTH),
                    n_channels=CHANNELS,
                    n_classes=N_CLASSES,
                    preprocessing=preprocessing,
                    mode='fit',
                    shuffle=False)

preds = model.predict_generator(inspect_generator)
print(str((time.time() - now)/60.0) + " minutes")    

for index, b in enumerate(range(len(preds))):
    filename = inspect_set['image'].iloc[b]
    image_df = inspect_set[inspect_set['image'] == filename].copy()
    pred_masks = preds[index, ].round().astype(int)
    pred_rles = build_rles(pred_masks, reshape=(350, 525))
    image_df['EncodedPixels_pred'] = pred_rles
    
    ### Post procecssing
    pred_masks_post = preds[index, ].astype('float32') 
    for class_index in range(N_CLASSES):
        pred_mask = pred_masks_post[...,class_index]
        pred_mask = post_process(pred_mask, threshold=best_tresholds[class_index], min_size=best_masks[class_index])
        pred_masks_post[...,class_index] = pred_mask

    pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
    image_df['EncodedPixels_pred_post'] = pred_rles_post
    ###
    inspect_set_temp.append(image_df)


inspect_set = pd.concat(inspect_set_temp)
inspect_predictions(inspect_set, images_to_inspect, validation_images_path, pred_col='EncodedPixels_pred')
print(str((time.time() - now)/60.0) + " minutes")    
inspect_predictions(inspect_set, images_to_inspect, validation_images_path, pred_col='EncodedPixels_pred_post')
# Choose 5 samples at random
images_to_inspect_test =  np.random.choice(sub_df['image'].unique(), 4, replace=False)
inspect_predictions(sub_df, images_to_inspect_test, test_images_path)
inspect_predictions(sub_df, images_to_inspect_test, test_images_path, label_col='EncodedPixels_post')
submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
submission_df_post.to_csv('submission_post.csv', index=False)
display(submission_df_post.head())
submission_df = sub_df[['Image_Label' ,'EncodedPixels']]
submission_df.to_csv('submission.csv', index=False)
display(submission_df.head())
print(str((time.time() - now)/60.0) + " minutes")    