#this is the acompanying 
"""

This is the inference kernel for [U-Net++ with EfficientNetB4](https://www.kaggle.com/xhlulu/severstal-u-net-with-efficientnetb4) kernel, which needed to be trained separately due to the running time exceeding 60 mins. This follows the same workflow as my [Severstal: Simple 2-step pipeline](https://www.kaggle.com/xhlulu/severstal-simple-2-step-pipeline), except it loads a trained model instead of training it from scratch. Below are the relevant sections:



* **Load Models**: Load the U-Net++ with some custom objects (ie. functions/classes not built-in Keras). Also load the [DenseNet for predicting missing masks](https://www.kaggle.com/xhlulu/severstal-predict-missing-masks) (i.e. samples without any defect). Those models were trained separately.

* **Step 1: Remove test images without defects**: Basically the same task as step 1 in the *Simple 2-step pipeline*.

* **Step 2: Predict masks using U-Net++**: Closely follows the step 2 of the same kernel, except this time we are resizing the decoded mask from 256x1600 to 256x512 before feeding it to the model, and resize the output of the model from 256x512 to 256x1600 before encoding it.



"""
import cv2

import keras

import keras.backend as K

from keras.optimizers import Adam

from keras.losses import binary_crossentropy

from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.engine import Layer, InputSpec

from keras.utils.generic_utils import get_custom_objects

from keras import initializers, constraints, regularizers, layers

import numpy as np

import pandas as pd

import tensorflow as tf

from tqdm import tqdm



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import cv2

import collections

import time 

import tqdm

from PIL import Image

from functools import partial

train_on_gpu = True

from skimage.data import imread

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import skimage

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from matplotlib.colors import LinearSegmentedColormap

from skimage.util import img_as_float

#Custom objects needed to load U-Net:



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

    def __init__(self, optimizer=Adam(2e-3), steps_per_update=4, **kwargs):

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
def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



class FixedDropout(keras.layers.Dropout):

    def _get_noise_shape(self, inputs):

        if self.noise_shape is None:

            return self.noise_shape



        symbolic_shape = K.shape(inputs)

        noise_shape = [symbolic_shape[axis] if shape is None else shape

                       for axis, shape in enumerate(self.noise_shape)]

        return tuple(noise_shape)

def dice_coef_rounded(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred = K.cast(y_pred, 'float32')

    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')

    intersection = y_true_f * y_pred_f

    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

    return score

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#densenet = load_model('../input/model.h5')

##densenet = load_model('../input/trained-cloud-model/model.h5')

#densenet.summary()



sub_df = pd.read_csv('../input/understanding_cloud_organization/sample_submission.csv')

sub_df['ImageId'] = sub_df['Image_Label'].apply(lambda x: x.split('_')[0])



test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])

test_imgs.head()







#train_df = pd.read_csv('../input/cloud-experiment-dataset/trainC1-16.csv')

#train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

#train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

#train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()



#`filtered_sub_df` contains all of the images with at least one mask. `null_sub_df` contains all the images with exactly 4 missing masks.



filtered_mask = sub_df['ImageId'].isin(test_imgs["ImageId"].values)

filtered_sub_df = sub_df[filtered_mask].copy()

null_sub_df = sub_df[~filtered_mask].copy()

null_sub_df['EncodedPixels'] = null_sub_df['EncodedPixels'].apply(lambda x: ' ')



filtered_sub_df.reset_index(drop=True, inplace=True)

test_imgs.reset_index(drop=True, inplace=True)



print(filtered_sub_df.shape)

print(null_sub_df.shape)



filtered_sub_df.head()

def np_resize(img, input_shape):

    """

    Reshape a numpy array, which is input_shape=(height, width), 

    as opposed to input_shape=(width, height) for cv2

    """

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
class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',

                 #base_path='../input/severstal-steel-defect-detection/train_images',

                 #base_path='../input/severstal-steel-defect-detection/train_images',

                 #batch_size=32, dim=(256, 1600), n_channels=3, reshape=None,

                 #n_classes=4, random_state=2019, shuffle=True):

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

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.random_state = random_state

        

        self.on_epoch_end()



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
# evaluate what models 

# activateelu_data_4 = True

# batch4_data_4 = True

# lesslayers_data_4 = True

# groupnorm_data_4 = True

# augmented_class_16 = True

# augmented_pure_4 = True



"""

unet_model_path = '../input/trained-models-cloud-mini-project/baseline_data_4.h5'

h_classes = 4

csv_file_name = 'baseline_data_4.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
"""

unet_model_path = '../input/trained-models-cloud-mini-project/activateelu_data_4.h5'

h_classes = 4

csv_file_name = 'activateelu_data_4.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
"""

unet_model_path = '../input/trained-models-cloud-mini-project/batch4_data_4.h5'

h_classes = 4

csv_file_name = 'batch4_data_4.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
"""

unet_model_path = '../input/trained-models-cloud-mini-project/lesslayers_data_4.h5'

h_classes = 4

csv_file_name = 'lesslayers_data_4.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
"""

unet_model_path = '../input/trained-models-cloud-mini-project/groupnorm_data_4.h5'

h_classes = 4

csv_file_name = 'groupnorm_data_4.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
"""

unet_model_path = '../input/trained-models-cloud-mini-project/augmented_pure_4.h5'

h_classes = 4

csv_file_name = 'augmented_pure_4.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
"""

unet_model_path = '../input/trained-models-cloud-mini-project/augmented_class_16.h5'

h_classes = 16

csv_file_name = 'augmented_class_16.csv'



from keras.optimizers import Adam



custom_objects = custom_objects={

    'swish': tf.nn.swish,

    'FixedDropout': FixedDropout,

    'dice_coef': dice_coef,

    'bce_dice_loss': bce_dice_loss,

    'GroupNormalization': GroupNormalization,

    'AccumOptimizer': Adam, # Placeholder, does not matter since we are not modifying the model

    'dice_coef_rounded': dice_coef_rounded



}



unet = load_model(unet_model_path, custom_objects=custom_objects)



unet.summary()



test_df = []



#for all images in test set iterate over thme in increments of 300

for i in range(0, test_imgs.shape[0], 300):

    #print("loop : ", i)



    # create a batch of 300 elements that needs to be processed

    batch_idx = list( range(i, min(test_imgs.shape[0], i + 300)) )



    # make the test generator for that batch 

    test_generator = DataGenerator(

        batch_idx,

        df=test_imgs,

        shuffle=False,

        mode='predict',

        base_path='../input/understanding_cloud_organization/test_images',        

        target_df=filtered_sub_df,

        reshape=(256, 384),

        batch_size=1,

        n_classes=h_classes

    )



    # predict masks for each of those sampels

    batch_pred_masks = unet.predict_generator(

        test_generator, 

        workers=1,

        verbose=1,

        use_multiprocessing=False

    )



    for j, b in tqdm(enumerate(batch_idx)):



        filename = test_imgs['ImageId'].iloc[b]



        image_df = sub_df[sub_df['ImageId'] == filename].copy()



        pred_masks = batch_pred_masks[j, ].round().astype(int)

        pred_rles = build_rles(pred_masks,reshape=(350, 525))



        image_df = pd.DataFrame(  np.array( [ 

            [image_df.iloc[0, 0], pred_rles[0], image_df.iloc[0, 2]], 

            [image_df.iloc[1, 0], pred_rles[1], image_df.iloc[1, 2]], 

            [image_df.iloc[2, 0], pred_rles[2], image_df.iloc[2, 2]], 

            [image_df.iloc[3, 0], pred_rles[3], image_df.iloc[3, 2]] ] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )



        test_df.append(image_df)











l1 = pred_rles[0]

l2 = pred_rles[1]

l3 = pred_rles[2]

l4 = pred_rles[3]





df = pd.DataFrame(  np.array( [ 

    [image_df.iloc[0, 0], l1, image_df.iloc[0, 2]], 

    [image_df.iloc[1, 0], l2, image_df.iloc[1, 2]], 

    [image_df.iloc[2, 0], l3, image_df.iloc[2, 2]], 

    [image_df.iloc[3, 0], l4, image_df.iloc[3, 2]] 

] ), columns=[ 'Image_Label', 'EncodedPixels', 'ImageId' ] )





print(df)



test_df = pd.concat(test_df)

final_submission_df = pd.concat([test_df, null_sub_df])



print(test_df.shape)

print(final_submission_df.shape)



final_submission_df.head()



final_submission_df[['Image_Label', 'EncodedPixels']].to_csv(csv_file_name, index=False)#



print(final_submission_df)

"""
def plotter_funk(my_plot, index, mask, label,image_name):

    my_plot_arr = my_plot.flat[index]

    my_plot_arr.axis('off') 

    imgcv2 = plt.imread('../input/understanding_cloud_organization/test_images/' + image_name)

    my_plot_arr.imshow(imgcv2)

    my_plot_arr.imshow(mask, alpha=0.5)

    my_plot_arr.set_title(label, fontsize=24)





def rle_decode(mask_rle, shape=(350, 525)):

   





    try: # label might not be there!

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

    



    

    





def plot_results( result_csv,image_name1,image_name2,image_name3,image_name4 ):



    train_df = pd.read_csv(result_csv)

    train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

    train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

    #train_df['EncodedPixels'] = train_df['EncodedPixels'].apply(lambda x: x.split(' ')[0])

    train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()



    print(result_csv)









    labels = [] # sorted(list(set(train_df['Image_Label'].apply(lambda x: x.split('_')[1]))))



    labels.append("Fish")

    labels.append("Flower")

    labels.append("Gravel")

    labels.append("Sugar")

    labels.append("Fish")

    labels.append("Flower")

    labels.append("Gravel")

    labels.append("Sugar")

    labels.append("Fish")

    labels.append("Flower")

    labels.append("Gravel")

    labels.append("Sugar")

    labels.append("Fish")

    labels.append("Flower")

    labels.append("Gravel")

    labels.append("Sugar")





    label = 'Fish'

    image_label = image_name1 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label , 'EncodedPixels'].values

    Fish1 = rle_decode(mask_rle)

    label = 'Flower'

    image_label = image_name1 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Flower1 = rle_decode(mask_rle)

    label = 'Gravel'

    image_label = image_name1 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Gravel1 = rle_decode(mask_rle)

    label = 'Sugar'

    image_label = image_name1 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Sugar1 = rle_decode(mask_rle)





    label = 'Fish'

    image_label = image_name2 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label , 'EncodedPixels'].values

    Fish2 = rle_decode(mask_rle)

    label = 'Flower'

    image_label = image_name2 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Flower2 = rle_decode(mask_rle)

    label = 'Gravel'

    image_label = image_name2 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Gravel2 = rle_decode(mask_rle)

    label = 'Sugar'

    image_label = image_name2 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Sugar2 = rle_decode(mask_rle)



    label = 'Fish'

    image_label = image_name3 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label , 'EncodedPixels'].values

    Fish3 = rle_decode(mask_rle)

    label = 'Flower'

    image_label = image_name3 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Flower3 = rle_decode(mask_rle)

    label = 'Gravel'

    image_label = image_name3 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Gravel3 = rle_decode(mask_rle)

    label = 'Sugar'

    image_label = image_name3 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Sugar3 = rle_decode(mask_rle)



    label = 'Fish'

    image_label = image_name4 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label , 'EncodedPixels'].values

    Fish4 = rle_decode(mask_rle)

    label = 'Flower'

    image_label = image_name4 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Flower4 = rle_decode(mask_rle)

    label = 'Gravel'

    image_label = image_name4 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Gravel4 = rle_decode(mask_rle)

    label = 'Sugar'

    image_label = image_name4 + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values

    Sugar4 = rle_decode(mask_rle)











    fig, my_plot = plt.subplots(4, 4, figsize=(20, 20)) 





    plotter_funk(my_plot, 0, Fish1, labels[0],image_name1)

    plotter_funk(my_plot, 1, Flower1, labels[1],image_name1)

    plotter_funk(my_plot, 2, Gravel1, labels[2],image_name1)

    plotter_funk(my_plot, 3, Sugar1, labels[3],image_name1)

    plotter_funk(my_plot, 4, Fish2, labels[0],image_name2)

    plotter_funk(my_plot, 5, Flower2, labels[1],image_name2)

    plotter_funk(my_plot, 6, Gravel2, labels[2],image_name2)

    plotter_funk(my_plot, 7, Sugar2, labels[3],image_name2)

    plotter_funk(my_plot, 8, Fish3, labels[0],image_name3)

    plotter_funk(my_plot, 9, Flower3, labels[1],image_name3)

    plotter_funk(my_plot, 10, Gravel3, labels[2],image_name3)

    plotter_funk(my_plot, 11, Sugar3, labels[3],image_name3)

    plotter_funk(my_plot, 12, Fish4, labels[0],image_name4)

    plotter_funk(my_plot, 13, Flower4, labels[1],image_name4)

    plotter_funk(my_plot, 14, Gravel4, labels[2],image_name4)

    plotter_funk(my_plot, 15, Sugar4, labels[3],image_name4)



    plt.tight_layout(h_pad=0.1, w_pad=0.1)

    plt.show()

    

    

    

    

plot_results('../input/trained-models-cloud-mini-project/baseline_data_4.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')

plot_results('../input/trained-models-cloud-mini-project/activateelu_data_4.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')

plot_results('../input/trained-models-cloud-mini-project/batch4_data_4.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')

plot_results('../input/trained-models-cloud-mini-project/lesslayers_data_4.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')

plot_results('../input/trained-models-cloud-mini-project/groupnorm_data_4.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')

plot_results('../input/trained-models-cloud-mini-project/augmented_class_16.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')

plot_results('../input/trained-models-cloud-mini-project/augmented_pure_4.csv' , '0a3d2d8.jpg', '006440a.jpg','00a47f4.jpg' ,'0111e06.jpg')
