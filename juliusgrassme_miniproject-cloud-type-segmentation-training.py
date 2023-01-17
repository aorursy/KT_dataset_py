# This notebook have trained the models for the miniproject, currently it is setup to run the baseline model 



# It uses U-Net++ with the pretrained EfficientNet B5 weights on to save kaggle training time, 

# I use the code from another notebook intended for the Serverstal competition but which i have adapted for this competition, i have forked the original notebook to show support.

# The notebook code is used as upposed to writing it from scratch as it have features which is otherwise quite time consuming to reimplement

# The main features of this notebook are:



# 1: It uses pretrained weights for the backbone in Unet which is preferable since GPU time is limited

# See: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet or the appropriate dataset here on kaggle



# 2: That it comes with an implementation of the accumulative optemizer which allows for the usage of larger batch sizes than what the GPU can handle in 1 go

# see for more info: https://translate.googleusercontent.com/translate_c?depth=1&rurl=translate.google.com&sl=zh-CN&sp=nmt4&tl=en&u=https://kexue.fm/archives/6794&xid=25657,15700021,15700186,15700190,15700256,15700259,15700262,15700265,15700271,15700283&usg=ALkJrhj73QF8gX6v1hUInBfN1JueIUFY9g 

# also: https://github.com/bojone/accum_optimizer_for_keras/blob/master/accum_optimizer.py



# 3: The dynamic decoding and of the dataset is set up which is needed for this dataset as having all the data in memory is infeseable here on kaggle

# this is just the standard generator https://keras.io/preprocessing/image/ (i first did it from scratch without in another notebook but discovered the issue there)





# The steps taken in this notebook are

# 1: Preprocessing which removes images with no lables so the net overfits less to empty images 

# 2: The model is built with the pretrained weights from the Efficientnet, here weights that was trained on imagenet 1000 are used

# 3: The generator for labels and images are defiend i downscale 1400, 2100 to 256, 384 as this preserves aspect ratio and speed up computation(but can be a problem for classes where the features are small in the images)

# 4: the model is trained using Adam for 21 epochs with the accumelation optimizer to allow grater batch sizes, here the dice coefficient is used as this is the evaluation metric for the competition

# 5: the model is saved and the training history is printed out





"""

# About this kernel(from starter code)



Please note that this training kernel can't be submitted directly, since it is trained for more than one hour. Please see [the inference kernel](https://www.kaggle.com/xhlulu/severstal-efficient-u-net-inference), with LB > 0.87.



* **Preprocessing**: Discard all of the training data that have all 4 masks missing. We will only train on images that have at least 1 mask, so that our model does not overfit on empty masks.

* **Utility Functions**: Utilities for encoding and decoding masks. Also create a `DataGenerator` class slightly modified from the other class, such that it can resize the encoded and decoded masks to the desired size.

* **Model: U-Net++ with EfficientNet Encoder**: Idea comes from [Siddartha's Efficient U-Net++](https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder). I modified the architecture by using simple convolution layers instead of residual blocks for decoding. Furthermore, my implementation is slightly different from the original paper since it performs a resolution reduction in the `H` and `U` functions, which was not exactly specified in the original paper. I also refactored most of the code in order to match the notation in the paper, and included the last efficientnet block (which was omitted by Siddartha).

* **Training the model**: We train the model for 30 epochs using an Adam with learning rate of 3e-4. The model with the best `val_dice_coef` is saved.

* **Evaluation**: Display the change in loss and Dice score throughout the epochs.



## Changelog

* **V28 [LB ?]**: Disabled GroupNormalization, added gradient accumulation.

* **V27 [LB 0.87328]**: Replaced all BatchNormalization with GroupNormalization, following [this discussion](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104686).

* **V18 [LB 0.87501]**: Added albumentations and many different types of transforms. Not yet evaluated.

* **V16 [LB 0.8766]**: Added random horizontal and vertical flip.

* **V13 [LB 0.8737]**: Changed loss from `binary_cross_entropy` to `bce_dice_loss`.



## References

* GroupNormalization: https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/104686

* UNet ++ Original Architecture: https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder

* Data generator: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

* RLE encoding and decoding: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

* Mask encoding: https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/data

* Original Kernel U-Net: https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate

* Original 2-step pipeline: https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline

* Missing mask predictor: https://www.kaggle.com/xhlulu/severstal-predict-missing-masks

* Source for `bce_dice_loss`: https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/



"""
import os

import json

import gc



import albumentations as albu

import cv2

import keras

from keras import backend as K

from keras.engine import Layer, InputSpec

from keras.utils.generic_utils import get_custom_objects

from keras import initializers, constraints, regularizers, layers

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model

from keras.layers import Input, Dropout, Conv2D, BatchNormalization, add

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras import backend as K

from keras.layers import LeakyReLU

from keras.layers import ELU

from keras.losses import binary_crossentropy

from keras.layers.merge import concatenate, Concatenate, Add

from keras.optimizers import Adam

from keras.callbacks import Callback, ModelCheckpoint

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import train_test_split



path = '../input/understanding_cloud_organization'

os.listdir(path)

train = pd.read_csv(f'{path}/train.csv')

sub = pd.read_csv(f'{path}/sample_submission.csv')

train.head()
print('the image dataset consists of:' )

train_num = len(os.listdir(f'{path}/train_images'))

test_num = len(os.listdir(f'{path}/test_images'))

print(f'{train_num} images in the training dataset')

print(f'{test_num} images in the testing dataset')

print('where the lables for the training is distributed like so')

train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()

print('the distribution of images with multiple classes looks like this')

train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()

print('there are a lot of images with multiple classes')
from PIL import Image



def rle_decode(mask_rle, shape=(350, 525)):

    

    try:

        s = mask_rle[0].split()



        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

        starts -= 1

        ends = starts + lengths

        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):

            img[lo:hi] = 1

            

        return img.reshape(shape, order='F')  # Needed to align to RLE direction

    except:

        #print("fail")

        return np.zeros((350, 525)) 





train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])

train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])



sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])

sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])



fig = plt.figure(figsize=(25, 16))

for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):

    for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):

        ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])

        im = Image.open(f"{path}/train_images/{row['Image_Label'].split('_')[0]}")

        plt.imshow(im)

        mask_rle = row['EncodedPixels']

        mask = rle_decode(mask_rle)

        plt.imshow(mask, alpha=0.5, cmap='gray')

        ax.set_title(f"im: {row['Image_Label'].split('_')[0]}. la: {row['label']}")
!pip install ../input/efficientnet-keras-source-code/repository/qubvel-efficientnet-c993591/ --quiet

import efficientnet.keras as efn
#train_df = pd.read_csv('../input/cloud-experiment-dataset/trainC1-4.csv') # the 4 pure classes of the 16 derived classes

#train_df = pd.read_csv('../input/cloud-experiment-dataset/trainC1-16.csv')# the 16 derived classes

train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv') # the original data



train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()



print(train_df.shape)

train_df.head()





mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

print(mask_count_df.shape)

mask_count_df.head()



sub_df = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')

sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

test_imgs.head()



non_missing_train_idx = mask_count_df[mask_count_df['hasMask'] > 0]

non_missing_train_idx.head()
def np_resize(img, input_shape):

    """Reshape a numpy array, which is input_shape=(height, width), as opposed to input_shape=(width, height) for cv2"""

    height, width = input_shape

    return cv2.resize(img, (width, height))

    

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



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

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



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 #base_path='../input/severstal-steel-defect-detection/train_images',

                 #batch_size=32, dim=(256, 1600), n_channels=3, reshape=None,

                 base_path='../input/understanding_cloud_organization/train_images',

                 batch_size=32, dim=(1400, 2100), n_channels=3, reshape=None,

                 augment=False, n_classes=4, random_state=2019, shuffle=True):

        self.dim = dim

        self.batch_size = batch_size

        self.df = df

        self.mode = mode

        self.base_path = base_path

        self.target_df = target_df

        self.list_IDs = list_IDs

        self.reshape = reshape

        self.n_channels = n_channels

        self.augment = augment

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()

        np.random.seed(self.random_state)



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.list_IDs) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        

        X = self.__generate_X(list_IDs_batch)

        

        if self.mode == 'fit':

            y = self.__generate_y(list_IDs_batch)

            

            if self.augment:

                X, y = self.__augment_batch(X, y)

            

            return X, y

        

        elif self.mode == 'predict':

            return X



        else:

            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

        

    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:

            np.random.seed(self.random_state)

            np.random.shuffle(self.indexes)

    

    def __generate_X(self, list_IDs_batch):

        'Generates data containing batch_size samples'

        # Initialization

        if self.reshape is None:

            X = np.empty((self.batch_size, *self.dim, self.n_channels))

        else:

            X = np.empty((self.batch_size, *self.reshape, self.n_channels))

        

        # Generate data

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            img_path = f"{self.base_path}/{im_name}"

            img = self.__load_rgb(img_path)

            

            if self.reshape is not None:

                img = np_resize(img, self.reshape)

            

            # Store samples

            X[i,] = img



        return X

    

    def __generate_y(self, list_IDs_batch):

        if self.reshape is None:

            y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        else:

            y = np.empty((self.batch_size, *self.reshape, self.n_classes), dtype=int)

        

        for i, ID in enumerate(list_IDs_batch):

            im_name = self.df['ImageId'].iloc[ID]

            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            

            rles = image_df['EncodedPixels'].values

            

            if self.reshape is not None:

                masks = build_masks(rles, input_shape=self.dim, reshape=self.reshape)

            else:

                masks = build_masks(rles, input_shape=self.dim)

            

            y[i, ] = masks



        return y

    

    def __load_grayscale(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.

        img = np.expand_dims(img, axis=-1)



        return img

    

    def __load_rgb(self, img_path):

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.



        return img

    

    def __random_transform(self, img, masks):

        composition = albu.Compose([

            albu.HorizontalFlip(),

            albu.VerticalFlip(),

            albu.ShiftScaleRotate(rotate_limit=30),

#             albu.OpticalDistortion(),

#             albu.GridDistortion(),

#             albu.ElasticTransform()

        ])

        

        composed = composition(image=img, mask=masks)

        aug_img = composed['image']

        aug_masks = composed['mask']

        

        return aug_img, aug_masks

    

    def __augment_batch(self, img_batch, masks_batch):

        for i in range(img_batch.shape[0]):

            img_batch[i, ], masks_batch[i, ] = self.__random_transform(img_batch[i, ], masks_batch[i, ])

        

        return img_batch, masks_batch
class GroupNormalization(keras.layers.Layer):

    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group

    the mean and variance for normalization. GN's computation is independent of batch sizes,

    and its accuracy is stable in a wide range of batch sizes

    # Arguments

        groups: Integer, the number of groups for Group Normalization.

        axis: Integer, the axis that should be normalized

            (typically the features axis).

            For instance, after a `Conv2D` layer with

            `data_format="channels_first"`,

            set `axis=1` in `BatchNormalization`.

        epsilon: Small float added to variance to avoid dividing by zero.

        center: If True, add offset of `beta` to normalized tensor.

            If False, `beta` is ignored.

        scale: If True, multiply by `gamma`.

            If False, `gamma` is not used.

            When the next layer is linear (also e.g. `nn.relu`),

            this can be disabled since the scaling

            will be done by the next layer.

        beta_initializer: Initializer for the beta weight.

        gamma_initializer: Initializer for the gamma weight.

        beta_regularizer: Optional regularizer for the beta weight.

        gamma_regularizer: Optional regularizer for the gamma weight.

        beta_constraint: Optional constraint for the beta weight.

        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape

        Arbitrary. Use the keyword argument `input_shape`

        (tuple of integers, does not include the samples axis)

        when using this layer as the first layer in a model.

    # Output shape

        Same shape as input.

    # References

        - [Group Normalization](https://arxiv.org/abs/1803.08494)

    """



    def __init__(self,

                 groups=32,

                 axis=-1,

                 epsilon=1e-5,

                 center=True,

                 scale=True,

                 beta_initializer='zeros',

                 gamma_initializer='ones',

                 beta_regularizer=None,

                 gamma_regularizer=None,

                 beta_constraint=None,

                 gamma_constraint=None,

                 **kwargs):

        super(GroupNormalization, self).__init__(**kwargs)

        self.supports_masking = True

        self.groups = groups

        self.axis = axis

        self.epsilon = epsilon

        self.center = center

        self.scale = scale

        self.beta_initializer = initializers.get(beta_initializer)

        self.gamma_initializer = initializers.get(gamma_initializer)

        self.beta_regularizer = regularizers.get(beta_regularizer)

        self.gamma_regularizer = regularizers.get(gamma_regularizer)

        self.beta_constraint = constraints.get(beta_constraint)

        self.gamma_constraint = constraints.get(gamma_constraint)



    def build(self, input_shape):

        dim = input_shape[self.axis]



        if dim is None:

            raise ValueError('Axis ' + str(self.axis) + ' of '

                             'input tensor should have a defined dimension '

                             'but the layer received an input with shape ' +

                             str(input_shape) + '.')



        if dim < self.groups:

            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '

                             'more than the number of channels (' +

                             str(dim) + ').')



        if dim % self.groups != 0:

            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '

                             'multiple of the number of channels (' +

                             str(dim) + ').')



        self.input_spec = InputSpec(ndim=len(input_shape),

                                    axes={self.axis: dim})

        shape = (dim,)



        if self.scale:

            self.gamma = self.add_weight(shape=shape,

                                         name='gamma',

                                         initializer=self.gamma_initializer,

                                         regularizer=self.gamma_regularizer,

                                         constraint=self.gamma_constraint)

        else:

            self.gamma = None

        if self.center:

            self.beta = self.add_weight(shape=shape,

                                        name='beta',

                                        initializer=self.beta_initializer,

                                        regularizer=self.beta_regularizer,

                                        constraint=self.beta_constraint)

        else:

            self.beta = None

        self.built = True



    def call(self, inputs, **kwargs):

        input_shape = K.int_shape(inputs)

        tensor_input_shape = K.shape(inputs)



        # Prepare broadcasting shape.

        reduction_axes = list(range(len(input_shape)))

        del reduction_axes[self.axis]

        broadcast_shape = [1] * len(input_shape)

        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups

        broadcast_shape.insert(1, self.groups)



        reshape_group_shape = K.shape(inputs)

        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]

        group_axes[self.axis] = input_shape[self.axis] // self.groups

        group_axes.insert(1, self.groups)



        # reshape inputs to new group shape

        group_shape = [group_axes[0], self.groups] + group_axes[2:]

        group_shape = K.stack(group_shape)

        inputs = K.reshape(inputs, group_shape)



        group_reduction_axes = list(range(len(group_axes)))

        group_reduction_axes = group_reduction_axes[2:]



        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)

        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)



        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))



        # prepare broadcast shape

        inputs = K.reshape(inputs, group_shape)

        outputs = inputs



        # In this case we must explicitly broadcast all parameters.

        if self.scale:

            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)

            outputs = outputs * broadcast_gamma



        if self.center:

            broadcast_beta = K.reshape(self.beta, broadcast_shape)

            outputs = outputs + broadcast_beta



        outputs = K.reshape(outputs, tensor_input_shape)



        return outputs



    def get_config(self):

        config = {

            'groups': self.groups,

            'axis': self.axis,

            'epsilon': self.epsilon,

            'center': self.center,

            'scale': self.scale,

            'beta_initializer': initializers.serialize(self.beta_initializer),

            'gamma_initializer': initializers.serialize(self.gamma_initializer),

            'beta_regularizer': regularizers.serialize(self.beta_regularizer),

            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),

            'beta_constraint': constraints.serialize(self.beta_constraint),

            'gamma_constraint': constraints.serialize(self.gamma_constraint)

        }

        base_config = super(GroupNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))



    def compute_output_shape(self, input_shape):

        return input_shape
class AccumOptimizer(keras.optimizers.Optimizer):

    """继承Optimizer类，包装原有优化器，实现梯度累积。

    # 参数

        optimizer：优化器实例，支持目前所有的keras优化器；

        steps_per_update：累积的步数。

    # 返回

        一个新的keras优化器

    Inheriting Optimizer class, wrapping the original optimizer

    to achieve a new corresponding optimizer of gradient accumulation.

    # Arguments

        optimizer: an instance of keras optimizer (supporting

                    all keras optimizers currently available);

        steps_per_update: the steps of gradient accumulation

    # Returns

        a new keras optimizer.

    """

    def __init__(self, optimizer, steps_per_update=1, **kwargs):

        super(AccumOptimizer, self).__init__(**kwargs)

        self.optimizer = optimizer

        with K.name_scope(self.__class__.__name__):

            self.steps_per_update = steps_per_update

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.cond = K.equal(self.iterations % self.steps_per_update, 0)

            self.lr = self.optimizer.lr

            self.optimizer.lr = K.switch(self.cond, self.optimizer.lr, 0.)

            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:

                if hasattr(self.optimizer, attr):

                    value = getattr(self.optimizer, attr)

                    setattr(self, attr, value)

                    setattr(self.optimizer, attr, K.switch(self.cond, value, 1 - 1e-7))

            for attr in self.optimizer.get_config():

                if not hasattr(self, attr):

                    value = getattr(self.optimizer, attr)

                    setattr(self, attr, value)

            # 覆盖原有的获取梯度方法，指向累积梯度

            # Cover the original get_gradients method with accumulative gradients.

            def get_gradients(loss, params):

                return [ag / self.steps_per_update for ag in self.accum_grads]

            self.optimizer.get_gradients = get_gradients

    def get_updates(self, loss, params):

        self.updates = [

            K.update_add(self.iterations, 1),

            K.update_add(self.optimizer.iterations, K.cast(self.cond, 'int64')),

        ]

        # 累积梯度 (gradient accumulation)

        self.accum_grads = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        grads = self.get_gradients(loss, params)

        for g, ag in zip(grads, self.accum_grads):

            self.updates.append(K.update(ag, K.switch(self.cond, ag * 0, ag + g)))

        # 继承optimizer的更新 (inheriting updates of original optimizer)

        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])

        self.weights.extend(self.optimizer.weights)

        return self.updates

    def get_config(self):

        iterations = K.eval(self.iterations)

        K.set_value(self.iterations, 0)

        config = self.optimizer.get_config()

        K.set_value(self.iterations, iterations)

        return config
## Losses and Metrics



def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
def H(lst, name, use_gn=False):

    if use_gn:

        norm = GroupNormalization(groups=1, name=name+'_gn')

    else:

        norm = BatchNormalization(name=name+'_bn')

    

    x = concatenate(lst)

    num_filters = int(x.shape.as_list()[-1]/2)

    

    x = Conv2D(num_filters, (2, 2), padding='same', name=name)(x)

    x = norm(x)

    x = LeakyReLU(alpha=1.0, name=name+'_activation')(x)

    

    return x



def U(x, use_gn=False):

    if use_gn:

        norm = GroupNormalization(groups=1)

    else:

        norm = BatchNormalization()

    

    num_filters = int(x.shape.as_list()[-1]/2)

    

    x = Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(x)

    x = norm(x)

    x = LeakyReLU(alpha=1.0)(x)

    

    return x
def EfficientUNet(input_shape):

    backbone = efn.EfficientNetB4(

        weights=None,

        include_top=False,

        input_shape=input_shape

    )

    

    backbone.load_weights(('../input/efficientnet-keras-weights-b0b5/'

                           'efficientnet-b4_imagenet_1000_notop.h5'))

    

    # Skipping block 4f and block 6h since they have the same output dim as 5f and 7b

    x00 = backbone.input  # (256, 384, 3) 

    x10 = backbone.get_layer('stem_activation').output  # (128, 192‬, 4) 

    x20 = backbone.get_layer('block2d_add').output  # (64, 96, 32)

    x30 = backbone.get_layer('block3d_add').output  # (32, 48, 56)

    x40 = backbone.get_layer('block5f_add').output  # (16, 24, 160)

    x50 = backbone.get_layer('block7b_add').output  # (8, 12, 448)

    

    x01 = H([x00, U(x10)], 'X01')

    x11 = H([x10, U(x20)], 'X11')

    x21 = H([x20, U(x30)], 'X21')

    x31 = H([x30, U(x40)], 'X31')

    x41 = H([x40, U(x50)], 'X41')

    

    x02 = H([x00, x01, U(x11)], 'X02')

    x12 = H([x11, U(x21)], 'X12')

    x22 = H([x21, U(x31)], 'X22')

    x32 = H([x31, U(x41)], 'X32')

    

    x03 = H([x00, x01, x02, U(x12)], 'X03')

    x13 = H([x12, U(x22)], 'X13')

    x23 = H([x22, U(x32)], 'X23')

    

    x04 = H([x00, x01, x02, x03, U(x13)], 'X04')

    x14 = H([x13, U(x23)], 'X14')

    

    x05 = H([x00, x01, x02, x03, x04, U(x14)], 'X05')

    

    x_out = Concatenate(name='bridge')([x01, x02, x03, x04, x05])

    x_out = Conv2D(4, (3,3), padding="same", name='final_output', activation="sigmoid")(x_out)

    

    return Model(inputs=x00, outputs=x_out)
#model = EfficientUNet((256, 512, 3))

model = EfficientUNet((256, 384 ,3))

model.compile(optimizer=AccumOptimizer(Adam(2e-3), 4), loss=bce_dice_loss, metrics=[dice_coef])

model.summary()
BATCH_SIZE = 8



train_idx, val_idx = train_test_split(

    non_missing_train_idx.index,  

    random_state=2019, 

    test_size=0.2

)



train_generator = DataGenerator(

    train_idx, 

    reshape=(256, 384),

    df=mask_count_df,

    target_df=train_df,

    augment=True,

    batch_size=BATCH_SIZE, 

    n_classes=4

)



val_generator = DataGenerator(

    val_idx, 

    reshape=(256, 384), 

    df=mask_count_df,

    target_df=train_df,

    augment=False,

    batch_size=BATCH_SIZE, 

    n_classes=4

)
checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)



history = model.fit_generator(

    train_generator,

    validation_data=val_generator, 

    callbacks=[checkpoint],

    verbose=1,

    epochs=1

)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['dice_coef', 'val_dice_coef']].plot()