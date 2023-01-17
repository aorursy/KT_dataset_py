# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.





import seaborn as sns

from sklearn.utils import shuffle

from os.path import isfile, join

import keras



# Standard dependencies

import cv2

import time

import scipy as sp

import numpy as np

import pandas as pd

from tqdm import tqdm

from PIL import Image

from functools import partial

import matplotlib.pyplot as plt



# Machine Learning

import tensorflow as tf

import keras

from keras import initializers

from keras import regularizers

from keras import constraints

from keras import backend as K

from keras.activations import elu

from keras.optimizers import Adam

from keras.models import Sequential

from keras.engine import Layer, InputSpec

from keras.utils.generic_utils import get_custom_objects

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout

from sklearn.metrics import cohen_kappa_score

import pydicom



import json



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom

from keras import layers

from keras.callbacks import Callback, ModelCheckpoint

from keras.models import Sequential

from keras.optimizers import Adam

from tqdm import tqdm



  

from keras import backend as K

import tensorflow as tf
!ls ../input

os.listdir('/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection')
BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = 'stage_2_train/'

TEST_DIR = 'stage_2_test/'

train_df = pd.read_csv(BASE_PATH + 'stage_2_train.csv')
train_df
train_df.ID == 'ID_6431af929_intraparenchymal'
sns.countplot(train_df.Label)
train_df.Label.value_counts()
sub_df = pd.read_csv(BASE_PATH + 'stage_2_sample_submission.csv')



train_df['filename'] = train_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

train_df['type'] = train_df['ID'].apply(lambda st: st.split('_')[2])

sub_df['filename'] = sub_df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

sub_df['type'] = sub_df['ID'].apply(lambda st: st.split('_')[2])



print(train_df.shape)

train_df.head()
test_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])

print(test_df.shape)

test_df.head()
train_df.shape
np.random.seed(2019)

sample_files = np.random.choice(os.listdir(BASE_PATH + TRAIN_DIR), 400000)

sample_df = train_df[train_df.filename.apply(lambda x: x.replace('.png', '.dcm')).isin(sample_files)]
pivot_df = sample_df[['Label', 'filename', 'type']].drop_duplicates().pivot(

    index='filename', columns='type', values='Label').reset_index()

print(pivot_df.shape)

pivot_df
def window_image(img, window_center,window_width, intercept, slope, rescale=True):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    

    if rescale:

        # Extra rescaling to 0-1, not in the original notebook

        img = (img - img_min) / (img_max - img_min)

    

    return img

    

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
def save_and_resize(filenames, load_dir):    

    save_dir = '/kaggle/tmp/'

    if not os.path.exists(save_dir):

        os.makedirs(save_dir)



    for filename in tqdm(filenames):

        try:

            path = load_dir + filename

            new_path = save_dir + filename.replace('.dcm', '.png')

            dcm = pydicom.dcmread(path)

            window_center , window_width, intercept, slope = get_windowing(dcm)

            img = dcm.pixel_array

            img = window_image(img, window_center, window_width, intercept, slope)



            resized = cv2.resize(img, (224, 224))

            res = cv2.imwrite(new_path, resized)

            

        except ValueError:

            continue

            
save_and_resize(filenames=sample_files, load_dir=BASE_PATH + TRAIN_DIR)

save_and_resize(filenames=os.listdir(BASE_PATH + TEST_DIR), load_dir=BASE_PATH + TEST_DIR)
!pip install -U efficientnet==0.0.4
from efficientnet import EfficientNetB4



size = 224
#print(os.listdir("../input/efficientnetb0b7-keras-weights/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5"))
effnet = EfficientNetB4(weights=None, include_top=False, input_shape=(size, size, 3))



effnet.load_weights('../input/efficientnetb0b7-keras-weights/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')
class GroupNormalization(Layer):

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


for i, layer in enumerate(effnet.layers):

    if "batch_normalization" in layer.name:

        effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
from keras.losses import binary_crossentropy

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
!pip install keras-rectified-adam

from keras_radam import RAdam
def build_model():

    """

    A custom implementation of EfficientNetB4

    """

    model = Sequential()

    model.add(effnet)

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))

    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss=bce_dice_loss,

                  optimizer=RAdam(warmup_proportion=0.1,lr=0.00005), 

                  metrics=['accuracy'])

    print(model.summary())

    return model
# Initialize model

model = build_model()
def preprocess_image(image, sigmaX=10):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (size, size))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

    return image
!pip install git+https://github.com/keras-team/keras-preprocessing.git
from keras_preprocessing.image import ImageDataGenerator
BATCH_SIZE = 32



def create_datagen():

    return ImageDataGenerator(validation_split=0.15)



def create_test_gen():

    return ImageDataGenerator().flow_from_dataframe(

        sub_df,

        directory=  '/kaggle/tmp/',

        x_col='filename',

        class_mode=None,

        target_size=(size, size),

        batch_size=BATCH_SIZE,

        shuffle=False

    )



def create_flow(datagen, subset):

    return datagen.flow_from_dataframe(

        pivot_df, 

        directory='/kaggle/tmp/',

        

        x_col='filename', 

        y_col=['any', 'epidural', 'intraparenchymal', 

               'intraventricular', 'subarachnoid', 'subdural'],

        class_mode='other',

        target_size=(size, size),

        batch_size=BATCH_SIZE,

        preprocessing_function=preprocess_image,

        rotation_range=360,

        horizontal_flip=True,

        validation_split=0.15,

        rescale=1 / 128.,

        subset=subset

    )



# Using original generator

data_generator = create_datagen()

train_gen = create_flow(data_generator, 'training')

val_gen = create_flow(data_generator, 'validation')

test_gen = create_test_gen()
checkpoint = ModelCheckpoint(

    'effnetb4.h5', 

    monitor='val_loss', 

    verbose=0, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)

#train_length = len(train_df)

total_steps = sample_files.shape[0] // BATCH_SIZE

total_steps = total_steps // 4

history = model.fit_generator(

    train_gen,

    steps_per_epoch = total_steps,

    validation_data=val_gen,

    validation_steps=total_steps * 0.15,

    callbacks=[checkpoint],

    epochs=1

)
model.load_weights('effnetb4.h5')

y_test = model.predict_generator(

    test_gen,

    steps=len(test_gen),

    verbose=1

)
# Append the output predicts in the wide format to the y_test

test_df = test_df.join(pd.DataFrame(y_test, columns=[

    'any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'

]))



# Unpivot table, i.e. wide (N x 6) to long format (6N x 1)

test_df = test_df.melt(id_vars=['filename'])



# Combine the filename column with the variable column

test_df['ID'] = test_df.filename.apply(lambda x: x.replace('.png', '')) + '_' + test_df.variable

test_df['Label'] = test_df['value']



test_df[['ID', 'Label']].to_csv('submission.csv', index=False)