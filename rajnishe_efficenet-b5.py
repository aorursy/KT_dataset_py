# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in ok



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

 #   for filename in filenames:

  #      print(os.path.join(dirname, filename))

#for dirname, _, filenames in os.walk('/kaggle/input/effnet'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

base_dir = "../input/aptos2019-blindness-detection/"



train_csv = base_dir+"train.csv"

test_csv = base_dir+"test.csv"

test_dir = base_dir+"test_images/"



#test_dir_processed = base_dir+'test_dir_processed'

#train_dir = base_dir+"train_data_cropped"



test_dir_processed = 'test_dir_processed'

train_dir = "train_data_cropped"



IMG_SIZE = 224



SEED = 72

# Any results you write to the current directory are saved as output.
#pip install -U git+http://github.com/qubvel/efficientnet

#!pip install git+https://github.com/qubvel/efficientnet
#https://github.com/qubvel/efficientnet#installation
import sys

# Repository source: https://github.com/qubvel/efficientnet

sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))

from efficientnet import EfficientNetB5

#from efficientnet import EfficientNetB4

#from efficientnet import EfficientNetB3

#from efficientnet import EfficientNetB2

#from efficientnet import EfficientNetB1

#from efficientnet import EfficientNetB0

import cv2 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight, shuffle
image = cv2.imread('../input/aptos2019-blindness-detection/train_images/295fdc964f6e.png')

plt.imshow(image)
import shutil

# commenting for kernel run else there will be error

#shutil.rmtree('train_data_cropped')

#shutil.rmtree('test_dir_processed')

os.mkdir('train_data_cropped')

os.mkdir('test_dir_processed')
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img         # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
image_path = '../input/aptos2019-blindness-detection/train_images/'



for fileName in os.listdir(image_path):

    

    #Ignore the file which are not png

    # some file with .DS_Store will be there

    # created by jupyter and caused issue as not images

    if fileName.endswith('png'):

        #image = cv2.imread('train_data'+'/'+fileName)

        image = cv2.imread(image_path+fileName)

        

        #convert into gray images

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        

        #Crop image so that there is less black around retina image

        image = crop_image_from_gray(image)

        

        # resize image ,default started with 512

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        

        #This line of code enhance image 

        #image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        #Please refer to about Gaussian

        #https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/ .

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)

        

        # save image on disk

        cv2.imwrite('train_data_cropped/'+fileName,image)
image_path = '../input/aptos2019-blindness-detection/test_images/'



for fileName in os.listdir(image_path):

    

    #Ignore the file which are not png

    # some file with .DS_Store will be there

    # created by jupyter and caused issue as not images

    if fileName.endswith('png'):

        #image = cv2.imread('train_data'+'/'+fileName)

        image = cv2.imread(image_path+fileName)

        

        #convert into gray images

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        

        #Crop image so that there is less black around retina image

        image = crop_image_from_gray(image)

        

        # resize image ,default started with 512

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        

        #This line of code enhance image 

        #image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

        #Please refer to about Gaussian

        #https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/ .

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)

        

        # save image on disk

        cv2.imwrite('test_dir_processed/'+fileName,image)
# Let's view processed images

i = 0

for fileName in os.listdir("train_data_cropped/"):

    i = i + 1



print(i)
# Let's view processed images

i = 0

for fileName in os.listdir("test_dir_processed/"):

    i = i + 1



print(i)
image1 = cv2.imread('../input/aptos2019-blindness-detection/train_images/295fdc964f6e.png')

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

plt.imshow(image1)
import matplotlib.pyplot as plt

#295fdc964f6e.png  c8905b8d5cf1.png

fig = plt.figure(figsize=(25, 16))

ax = fig.add_subplot(5, 5, 5, xticks=[], yticks=[])

image2 = cv2.imread('train_data_cropped/295fdc964f6e.png')

image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

plt.imshow(image2,cmap='gray')

image2.shape
# constants for 

WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")





NUM_CLASSES = 5

SEED = 72

TRAIN_NUM = 1000
def df_train_test_split_preprocess(df):

    

    image_ids = df["id_code"].values.tolist()

    labels = df["diagnosis"].values.tolist()

    

    for i in range(len(image_ids)):

        imgname = image_ids[i]

        newname = str(imgname) + ".png"

        image_ids[i] = newname

    

    xtrain, xval, ytrain, yval = train_test_split(image_ids, labels, test_size = 0.15)

    

    df_train = pd.DataFrame({"id_code":xtrain, "diagnosis":ytrain})

    df_val = pd.DataFrame({"id_code":xval, "diagnosis":yval})

    

    df_train["diagnosis"] = df_train["diagnosis"].astype('str')

    df_val["diagnosis"] = df_val["diagnosis"].astype('str')

    

    print("Length of Training Data :",len(df_train))

    print("Length of Validation Data :",len(df_val))

    

    return df_train, df_val
df = pd.read_csv(train_csv)

df_train, df_val = df_train_test_split_preprocess(df)
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score
from keras.preprocessing.image import ImageDataGenerator



#   --- TO DO ----

#   zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.

#   zca_whitening: Boolean. Apply ZCA whitening.

#

train_aug = ImageDataGenerator(rescale=1./255,

                               horizontal_flip = True,

                               zoom_range = 0.15,

                               vertical_flip = True,

                               shear_range=0.1,

                               rotation_range = 90

                               )
train_generator = train_aug.flow_from_dataframe(dataframe = df_train,

                                               directory = train_dir,

                                               x_col = "id_code",

                                               y_col = "diagnosis",

                                               batch_size = 16,

                                               target_size =  (IMG_SIZE, IMG_SIZE),

                                               #color_mode = 'grayscale',

                                               class_mode = "categorical")
# Using same as for training

validation_generator = train_aug.flow_from_dataframe(dataframe = df_val,

                                                    directory = train_dir,

                                                    x_col = "id_code",

                                                    y_col = "diagnosis",

                                                    batch_size = 16, 

                                                    target_size = (IMG_SIZE, IMG_SIZE),

                                                    #color_mode = 'grayscale',

                                                    class_mode = "categorical")
import keras

import tensorflow as tf

from keras import backend as K

from sklearn.metrics import cohen_kappa_score

# build model...(not shown)



# custom metric with TF

def cohens_kappa(y_true, y_pred):

    y_true_classes = tf.argmax(y_true, 1)

    y_pred_classes = tf.argmax(y_pred, 1)

    ck_val = tf.contrib.metrics.cohen_kappa(y_true_classes, y_pred_classes, 5)[1]

    print(ck_val)

    return ck_val



# custom metric with TF

#def quad_cohens_kappa(y_true, y_pred):

#    y_true_classes = tf.argmax(y_true, 1)

#    y_pred_classes = tf.argmax(y_pred, 1)

#    print(y_true_classes)

    #ck_val = cohen_kappa_score(y_true_classes, y_pred_classes, weights='quadratic')

#    ck_val = 0

#    print(ck_val)

#    return ck_val

test_df_orig = pd.read_csv(test_csv)



def process_test_df(test_df):

    test_ids = test_df["id_code"].values.tolist()

    for i in range(len(test_ids)):

        imgname = test_ids[i]

        newname = str(imgname) + ".png"

        test_ids[i] = newname

    test_df["id_code"] = test_ids

    return test_df



test_df = process_test_df(test_df_orig)
# No need to augment only rescale pixel values

test_aug = ImageDataGenerator(rescale = 1./255 )



test_generator = test_aug.flow_from_dataframe(dataframe = test_df, 

                                              directory = test_dir_processed,

                                              x_col = "id_code",

                                              batch_size = 1,

                                              target_size =  (IMG_SIZE, IMG_SIZE), # to be changed as ???

                                              shuffle = False,

                                              class_mode = None)
# Code Source: https://github.com/CyberZHG/keras-radam/blob/master/keras_radam/optimizers.py

class RAdam(keras.optimizers.Optimizer):

    """RAdam optimizer.

    # Arguments

        lr: float >= 0. Learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: float >= 0. Weight decay for each param.

        amsgrad: boolean. Whether to apply the AMSGrad variant of this

            algorithm from the paper "On the Convergence of Adam and

            Beyond".

        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.

        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.

        min_lr: float >= 0. Minimum learning rate after warmup.

    # References

        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)

        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,

                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):

        super(RAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

            self.weight_decay = K.variable(weight_decay, name='weight_decay')

            self.total_steps = K.variable(total_steps, name='total_steps')

            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')

            self.min_lr = K.variable(lr, name='min_lr')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.initial_weight_decay = weight_decay

        self.initial_total_steps = total_steps

        self.amsgrad = amsgrad



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations, K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1



        if self.initial_total_steps > 0:

            warmup_steps = self.total_steps * self.warmup_proportion

            decay_steps = self.total_steps - warmup_steps

            lr = K.switch(

                t <= warmup_steps,

                lr * (t / warmup_steps),

                lr * (1.0 - K.minimum(t, decay_steps) / decay_steps),

            )



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]

        else:

            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]



        self.weights = [self.iterations] + ms + vs + vhats



        beta_1_t = K.pow(self.beta_1, t)

        beta_2_t = K.pow(self.beta_2, t)



        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0

        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)



        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)



            m_corr_t = m_t / (1.0 - beta_1_t)

            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t) + self.epsilon)

                self.updates.append(K.update(vhat, vhat_t))

            else:

                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t) + self.epsilon)



            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *

                         (sma_t - 2.0) / (sma_inf - 2.0) *

                         sma_inf / sma_t)



            p_t = K.switch(sma_t > 5, r_t * m_corr_t / v_corr_t, m_corr_t)



            if self.initial_weight_decay > 0:

                p_t += self.weight_decay * p



            p_t = p - lr * p_t



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {

            'lr': float(K.get_value(self.lr)),

            'beta_1': float(K.get_value(self.beta_1)),

            'beta_2': float(K.get_value(self.beta_2)),

            'decay': float(K.get_value(self.decay)),

            'weight_decay': float(K.get_value(self.weight_decay)),

            'epsilon': self.epsilon,

            'amsgrad': self.amsgrad,

            'total_steps': float(K.get_value(self.total_steps)),

            'warmup_proportion': float(K.get_value(self.warmup_proportion)),

            'min_lr': float(K.get_value(self.min_lr)),

        }

        base_config = super(RAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
from keras.applications import ResNet50

from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import keras

from keras.engine import Layer,InputSpec

#from keras.applications import DenseNet121,DenseNet169,DenseNet201
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
print(IMG_SIZE)
input_layer = Input(shape = (IMG_SIZE,IMG_SIZE,3))



base_model = EfficientNetB5(weights = None,

                       include_top = False,

                       input_tensor = input_layer)



base_model.load_weights('../input/effnet/efficientnetb5notop.h5')

# all are false

#for layer in base_model.layers:

#    layer.trainable = False

# top 5 are fasle    

#for layer in base_model.layers[:180]:

#    layer.trainable = False

#--- v1 with 90 , it was CH = .7435 & ACC = .89 start with .43   

#--- v2 with 71 , it was CH = .7435 & ACC = .89 start with .43   



# last 4 are false

#for layer in vgg_conv.layers[:-4]:

#    layer.trainable = False

    

x = GlobalAveragePooling2D()(base_model.output)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.40)(x)

x = Dense(512, activation='relu')(x)

x = Dropout(0.3)(x)

out = Dense(5, activation = 'softmax')(x)



model = Model(inputs = input_layer, outputs = out)
#model.summary()
#for layer in model.layers:

#    print(layer.name,layer.trainable)
#optimizer = keras.optimizers.Adam(lr=2e-4)

#optimizer = keras.optimizers.Adam(lr=0.0005)



optimizer = RAdam(lr=0.0005)

#optimizer = RAdam(lr=2e-4)

#es = EarlyStopping(monitor='val_loss', mode='min', patience = 9, restore_best_weights=True)

#rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience = 3, factor = 0.5, min_lr=1e-6)



es = EarlyStopping(monitor='cohens_kappa', mode='auto', verbose=1, patience=3,restore_best_weights=True)

rlrop = ReduceLROnPlateau(monitor='cohens_kappa', 

                        factor=0.2, 

                        patience=5, 

                        verbose=1, 

                        mode='auto', 

                        min_lr=1e-6)

#kappa_metrics = Metrics()

callback_list = [ rlrop ]



model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy",cohens_kappa]) 
K.get_session().run(tf.local_variables_initializer())
#import gc

#gc.collect()
history = model.fit_generator(generator = train_generator, 

                    steps_per_epoch = len(train_generator), 

                    epochs = 18, 

                    validation_data = validation_generator, 

                    validation_steps = len(validation_generator),

                    callbacks =  callback_list  )
acc = history.history['acc']

val_acc = history.history['val_acc']



loss = history.history['loss']

val_loss = history.history['val_loss']



cohens_kappa = history.history['cohens_kappa']

val_cohens_kappa = history.history['val_cohens_kappa']



epochs = range(len(acc))

 

plt.plot(epochs, acc, 'b', label='Training acc')

plt.plot(epochs, val_acc, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

 

plt.figure()

 

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.figure()

 

plt.plot(epochs, cohens_kappa, 'b', label='Training Cohen-kappa')

plt.plot(epochs, val_cohens_kappa, 'r', label='Validation Cohen-kappa')

plt.title('Cohen Kappa - Training and validation score')

plt.legend()





plt.show()
predprobs = model.predict_generator(test_generator, steps=len(test_generator))
#Cleaning all processed file

# Else you will get "TOO MANY FILES" Error

shutil.rmtree('train_data_cropped')

shutil.rmtree('test_dir_processed')
# select prediction of highest probability

predictions = []

for i in predprobs:

    predictions.append(np.argmax(i))
test_df_orig.info()
# create new column and assign prediction class

test_df_orig["diagnosis"] = predictions
test_df_orig.head(1)
test_ids = test_df_orig["id_code"].values.tolist()

for i in range(len(test_ids)):

    imgname = test_ids[i]

    newname = imgname.split('.')[0]

    test_ids[i] = newname

    test_df_orig["id_code"] = test_ids
test_df_orig.head(5)
test_df_orig.to_csv('submission.csv',index=False)
subfile = pd.read_csv('submission.csv')

subfile.head(3)