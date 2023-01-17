# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.





import cv2

import json

import matplotlib.pyplot as plt



import seaborn as sns

from collections import Counter

from PIL import Image

import math

import seaborn as sns

from collections import defaultdict

from pathlib import Path

import cv2

from tqdm import tqdm



import albumentations as A



from keras.utils import Sequence



from IPython.display import clear_output

pd.set_option("display.max_rows", 101)

plt.style.use('ggplot')



PATH = "../input/severstal-steel-defect-detection/"
train_df = pd.read_csv(f"{PATH}/train.csv")

test_df = pd.read_csv(f"{PATH}/sample_submission.csv")

train_df.shape, test_df.shape
# RESTRUCTURE TRAIN DATAFRAME

train_df['ImageId'] = train_df['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')

train2 = pd.DataFrame({'ImageId':train_df['ImageId'][::4]})

train2['e1'] = train_df['EncodedPixels'][::4].values

train2['e2'] = train_df['EncodedPixels'][1::4].values

train2['e3'] = train_df['EncodedPixels'][2::4].values

train2['e4'] = train_df['EncodedPixels'][3::4].values

train2.reset_index(inplace=True,drop=True)

train2.fillna('',inplace=True); 

train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values

train2.head()
# RESTRUCTURE TRAIN DATAFRAME

test_df['ImageId'] = test_df['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
train2 = train2[train2['count'] > 0]

train2.shape
# https://www.kaggle.com/titericz/building-and-visualizing-masks

def rle2maskResize(rle):

    # CONVERT RLE TO MASK 

    if (pd.isnull(rle))|(rle==''): 

        return np.zeros((256,1600) ,dtype=np.uint8)

    

    height= 256

    width = 1600

    mask= np.zeros(width*height ,dtype=np.uint8)



    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]-1

    lengths = array[1::2]    

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

    

    return mask.reshape((height,width), order='F')#[::2,::2]



def mask2contour(mask, width=3):

    # CONVERT MASK TO ITS CONTOUR

    w = mask.shape[1]

    h = mask.shape[0]

    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)

    mask2 = np.logical_xor(mask,mask2)

    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)

    mask3 = np.logical_xor(mask,mask3)

    return np.logical_or(mask2,mask3) 



def mask2pad(mask, pad=2):

    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT

    w = mask.shape[1]

    h = mask.shape[0]

    

    # MASK UP

    for k in range(1,pad,2):

        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK DOWN

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)

        mask = np.logical_or(mask,temp)

    # MASK LEFT

    for k in range(1,pad,2):

        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)

        mask = np.logical_or(mask,temp)

    # MASK RIGHT

    for k in range(1,pad,2):

        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)

        mask = np.logical_or(mask,temp)

    

    return mask 



def mask2rle(img):

    tmp = np.rot90( np.flipud( img ), k=3 )

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(startpos)+' '+str(endpos-startpos+1) )

    return " ".join(rle)



def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )

# helper function for data visualization

def visualize(gray=False, **images):

    """PLot images in one row."""

    n = len(images)

    plt.figure(figsize=(12, 12))

    for i, (name, image) in enumerate(images.items()):

        plt.subplot(n, 1, i+1)

        plt.xticks([])

        plt.yticks([])

        plt.title(name)

        if gray: 

            plt.imshow(image.squeeze())

        else:

            plt.imshow(image)

    plt.show()

    

# helper function for data visualization    

def denormalize(x):

    """Scale image to range 0..1 for correct plot"""

    x_max = np.percentile(x, 98)

    x_min = np.percentile(x, 2)    

    x = (x - x_min) / (x_max - x_min)

    x = x.clip(0, 1)

    return x

    

class Dataset:

    """

    Args:

        images_dir (str): path to images folder

        masks_dir (str): path to segmentation masks folder

        class_values (list): values of classes to extract from segmentation mask

        augmentation (albumentations.Compose): data transfromation pipeline 

            (e.g. flip, scale, etc.)

        preprocessing (albumentations.Compose): data preprocessing 

            (e.g. noralization, shape manipulation, etc.)

    

    """

    

    

    def __init__(self, df, subset='train', augmentation=None, preprocessing=None):

        

        self.CLASSES = ['e1', 'e2', 'e3', 'e4']

        self.df = df

        self.subset = subset

        

        if self.subset == "train":

            self.data_path = PATH + 'train_images/'

            self.images_fps = [os.path.join(self.data_path, image_id) for image_id in self.df['ImageId']]

        elif self.subset == "test":

            self.data_path = PATH + 'test_images/'

            self.images_fps = [os.path.join(self.data_path, image_id) for image_id in self.df['ImageId'].unique()]

            

        self.masks_fps = self.df.drop('ImageId', 1)

        

        

        # convert str names to class values on masks

#         self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        

        self.augmentation = augmentation

        self.preprocessing = preprocessing

    

    def __getitem__(self, i):

        

        # read data

        image = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)

#         print(image.shape)

#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        

        if len(image.shape) == 2:

            image = image[..., np.newaxis]

#             print(image.shape)

        if self.subset == 'train':

            masks = np.zeros((256, 1600, len(self.CLASSES)), dtype=np.uint8)



            # extract certain classes from mask (e.g. cars)

            for j in range(len(self.CLASSES)):

                masks[..., j] = rle2maskResize(self.df[f'e{j+1}'].iloc[i])



    #         # add background if mask is not binary

    #         if masks.shape[-1] != 1:

    #             background = 1 - masks.sum(axis=-1, keepdims=True)

    #             masks = np.concatenate((masks, background), axis=-1)



            # apply augmentations

            if self.augmentation:

                sample = self.augmentation(image=image, mask=masks)

                image, masks = sample['image'], sample['mask']



            # apply preprocessing

            if self.preprocessing:

                sample = self.preprocessing(image=image, mask=masks)

                image, masks = sample['image'], sample['mask']



            return image/255., masks

        

        else:

            return image/255.

        

    def __len__(self):

        return len(self.images_fps)

    

    

class Dataloder(Sequence):

    """Load data from dataset and form batches

    

    Args:

        dataset: instance of Dataset class for image loading and preprocessing.

        batch_size: Integet number of images in batch.

        shuffle: Boolean, if `True` shuffle image indexes each epoch.

    """

    

    def __init__(self, dataset, batch_size=16, shuffle=False, subset='train'):

        self.dataset = dataset

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.indexes = np.arange(len(dataset))

        self.subset = subset

        self.on_epoch_end()



    def __getitem__(self, i):

#         if self.__len__()-1 == i:

#             self.indexes = np.random.permutation(self.indexes)

        # collect batch data

        start = i * self.batch_size

        stop = (i + 1) * self.batch_size

        data = []

        for j in range(start, stop):

            data.append(self.dataset[j]) #(self.dataset[j][0], list(self.dataset[j][1].transpose(2,0,1))))

        

        if self.subset == 'test':

        # transpose list of lists

            clear_output()

            print(i)

            return np.array(data)

        

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        #batch = list(map(lambda x: list(x[1].squeeze()), batch))

        return batch

    

    def __len__(self):

        """Denotes the number of batches per epoch"""

        return len(self.indexes) // self.batch_size

    

    def on_epoch_end(self):

        """Callback function to shuffle indexes each epoch"""

        if self.shuffle:

            self.indexes = np.random.permutation(self.indexes)
def round_clip_0_1(x, **kwargs):

    return x.round().clip(0, 1)



# define heavy augmentations

def get_training_augmentation():

    train_transform = [



        A.VerticalFlip(p=0.5),

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),

        A.GridDistortion(p=0.5),

        A.OpticalDistortion(p=0.4, distort_limit=2, shift_limit=0.5)

    ]

    return A.Compose(train_transform)





def get_validation_augmentation():

    """Add paddings to make image shape divisible by 32"""

    test_transform = [

        A.PadIfNeeded(256, 1600)

    ]

    return A.Compose(test_transform)



def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        A.Lambda(image=preprocessing_fn),

    ]

    

    return A.Compose(_transform)
# Lets look at data we have

dataset = Dataset(train2, augmentation=get_training_augmentation())

print(len(dataset))

image, mask = dataset[0] # get some sample

visualize(gray=True,

    image=image, 

    e1=mask[..., 0].squeeze(),

    e2=mask[..., 1].squeeze(),

    e3=mask[..., 2].squeeze(),

    e4=mask[..., 3].squeeze(),

)

from sklearn.model_selection import train_test_split



train_idx, valid_idx = train_test_split(np.arange(len(train2)), test_size=0.05)



train_dataset = Dataset(train2.iloc[train_idx], augmentation=get_training_augmentation())

valid_dataset = Dataset(train2.iloc[valid_idx])





train_dataloader = Dataloder(train_dataset, batch_size=8, shuffle=True)

valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)
import h5py



from keras.models import Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Reshape

from keras.layers import Activation, add, multiply, Lambda, concatenate

from keras.layers import AveragePooling2D, average, UpSampling2D, Dropout, Flatten, Add, Maximum

from keras.optimizers import Adam, SGD, RMSprop

from keras.initializers import glorot_normal, random_normal, random_uniform

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

from keras.losses import binary_crossentropy

from keras import optimizers

from keras import backend as K

from keras.layers.normalization import BatchNormalization 

from keras.applications import VGG19, densenet

from keras.models import load_model

from keras.utils.generic_utils import get_custom_objects



import tensorflow as tf 



import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools





K.set_image_data_format('channels_last')  # TF dimension ordering in this code

kinit = 'glorot_normal'



epsilon = 1e-5

smooth = 1



class Swish(Activation):

    

    def __init__(self, activation, **kwargs):

        super(Swish, self).__init__(activation, **kwargs)

        self.__name__ = 'SWISH'



def swish(x):

    return (K.sigmoid(x) * x)



get_custom_objects().update({'swish': Swish(swish)})





def dsc_np(y_true, y_pred):

    smooth = 1.

    y_true_f = y_true.flatten()

    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f * y_pred_f)

    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

    return 1 - score





def dsc(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score



def dice_loss(y_true, y_pred):

    loss = 1 - dsc(y_true, y_pred)

    return loss



def bce_dice_loss(y_true, y_pred):

    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss



def confusion(y_true, y_pred):

    smooth=1

    y_pred_pos = K.clip(y_pred, 0, 1)

    y_pred_neg = 1 - y_pred_pos

    y_pos = K.clip(y_true, 0, 1)

    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)

    fp = K.sum(y_neg * y_pred_pos)

    fn = K.sum(y_pos * y_pred_neg) 

    prec = (tp + smooth)/(tp+fp+smooth)

    recall = (tp+smooth)/(tp+fn+smooth)

    return prec, recall



def tp(y_true, y_pred):

    smooth = 1

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pos = K.round(K.clip(y_true, 0, 1))

    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 

    return tp 



def tn(y_true, y_pred):

    smooth = 1

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))

    y_neg = 1 - y_pos 

    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )

    return tn 



def tversky(y_true, y_pred):

    y_true_pos = K.flatten(y_true)

    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)

    false_neg = K.sum(y_true_pos * (1-y_pred_pos))

    false_pos = K.sum((1-y_true_pos)*y_pred_pos)

    alpha = 0.7

    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)



def tversky_loss(y_true, y_pred):

    return 1 - tversky(y_true,y_pred)



def focal_tversky(y_true,y_pred):

    pt_1 = tversky(y_true, y_pred)

    gamma = 0.75

    return K.pow((1-pt_1), gamma)





class RAdam(optimizers.Optimizer):

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

            self.min_lr = K.variable(min_lr, name='min_lr')

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

            lr = K.switch(

                t <= warmup_steps,

                lr * (t / warmup_steps),

                self.min_lr + (lr - self.min_lr) * (1.0 - K.minimum(t, self.total_steps) / self.total_steps),

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



            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / v_corr_t, m_corr_t)



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
ACT = 'swish'



def expend_as(tensor, rep,name):

    my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep},  name='psi_up'+name)(tensor)

    return my_repeat





def AttnGatingBlock(x, g, inter_shape, name):

    ''' take g which is the spatially smaller signal, do a conv to get the same

    number of feature channels as x (bigger spatially)

    do a conv on x to also get same geature channels (theta_x)

    then, upsample g to be same size as x 

    add x and g (concat_xg)

    relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''

    

    shape_x = K.int_shape(x)  # 32

    shape_g = K.int_shape(g)  # 16



    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)  # 16

    shape_theta_x = K.int_shape(theta_x)



    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)

    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same', name='g_up'+name)(phi_g)  # 16



    concat_xg = add([upsample_g, theta_x])

    act_xg = Activation(ACT)(concat_xg)

    psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)

    sigmoid_xg = Activation('sigmoid')(psi)

    shape_sigmoid = K.int_shape(sigmoid_xg)

    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32



    upsample_psi = expend_as(upsample_psi, shape_x[3],  name)

    y = multiply([upsample_psi, x], name='q_attn'+name)



    result = Conv2D(shape_x[3], (1, 1), padding='same',name='q_attn_conv'+name)(y)

    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)

    return result_bn



def UnetConv2D(input, outdim, is_batchnorm, name):

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)

    if is_batchnorm:

        x =BatchNormalization(name=name + '_1_bn')(x)

    x = Activation(ACT,name=name + '_1_act')(x)



    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)

    if is_batchnorm:

        x = BatchNormalization(name=name + '_2_bn')(x)

    x = Activation(ACT, name=name + '_2_act')(x)

    return x





def UpConv(x, n_exp, name, is_batchnorm=True):

    for i in range(n_exp-2):

        x = Conv2DTranspose(4, (4,4), strides=(2,2), padding='same', kernel_initializer=kinit)(x)

        if is_batchnorm:

            x = BatchNormalization(name=name + '_bn' + str(i))(x)

        x = Activation(ACT)(x)

    x = Conv2DTranspose(4, (4,4), strides=(2,2), padding='same', activation=ACT, kernel_initializer=kinit)(x)

    return x



def UnetGatingSignal(input, is_batchnorm, name):

    ''' this is simply 1x1 convolution, bn, activation '''

    shape = K.int_shape(input)

    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)

    if is_batchnorm:

        x = BatchNormalization(name=name + '_bn')(x)

    x = Activation(ACT, name = name + '_act')(x)

    return x





def attn_reg(opt,input_size, lossfxn):

    

    img_input = Input(shape=input_size, name='input_scale1')

    scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)

    scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)

    scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)



    conv1 = UnetConv2D(img_input, 16, is_batchnorm=True, name='conv1')

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    

    input2 = Conv2D(32, (3, 3), padding='same', activation=ACT, name='conv_scale2')(scale_img_2)

    input2 = concatenate([input2, pool1], axis=3)

    conv2 = UnetConv2D(input2, 32, is_batchnorm=True, name='conv2')

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    

    input3 = Conv2D(64, (3, 3), padding='same', activation=ACT, name='conv_scale3')(scale_img_3)

    input3 = concatenate([input3, pool2], axis=3)

    conv3 = UnetConv2D(input3, 64, is_batchnorm=True, name='conv3')

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    

    input4 = Conv2D(128, (3, 3), padding='same', activation=ACT, name='conv_scale4')(scale_img_4)

    input4 = concatenate([input4, pool3], axis=3)

    conv4 = UnetConv2D(input4, 32, is_batchnorm=True, name='conv4')

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        

    center = UnetConv2D(pool4, 256, is_batchnorm=True, name='center')

    

    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')

    attn1 = AttnGatingBlock(conv4, g1, 64, '_1')

    up1 = concatenate([Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', activation=ACT, kernel_initializer=kinit)(center), attn1], name='up1')



    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')

    attn2 = AttnGatingBlock(conv3, g2, 32, '_2')

    up2 = concatenate([Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', activation=ACT, kernel_initializer=kinit)(up1), attn2], name='up2')



    g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')

    attn3 = AttnGatingBlock(conv2, g3, 16, '_3')

    up3 = concatenate([Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', activation=ACT, kernel_initializer=kinit)(up2), attn3], name='up3')



    up4 = concatenate([Conv2DTranspose(16, (3,3), strides=(2,2), padding='same', activation=ACT, kernel_initializer=kinit)(up3), conv1], name='up4')

    

    conv6 = UnetConv2D(up1, 32, is_batchnorm=True, name='conv6')

    conv7 = UnetConv2D(up2, 32, is_batchnorm=True, name='conv7')

    conv8 = UnetConv2D(up3, 32, is_batchnorm=True, name='conv8')

    conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')



    out6 = Conv2D(1, (1, 1), activation=ACT, name='pred1')(conv6)

    out7 = Conv2D(1, (1, 1), activation=ACT, name='pred2')(conv7)

    out8 = Conv2D(1, (1, 1), activation=ACT, name='pred3')(conv8)

    out9 = Conv2D(4, (1, 1), activation=ACT, name='final')(conv9)

    

    out6 = UpConv(out6, 4, name='total_final6')

    out7 = UpConv(out7, 3, name='total_final7')

    out8 = UpConv(out8, 2, name='total_final8')

    

    out = Add()([out6,out7,out8,out9])

    

    out = Activation('softmax')(out)

#     out = Lambda(lambda x: K.cast(x > 0.5, dtype='float16'), output_shape=(256, 1600, 4))(out)

    

    model = Model(inputs=[img_input], outputs=[out])

 

    loss = {'pred1':lossfxn,

            'pred2':lossfxn,

            'pred3':lossfxn,

            'final': tversky_loss}

    

#     loss_weights = {'pred1':1,

#                     'pred2':1,

#                     'pred3':1,

#                     'final':1}

#     model.compile(optimizer=opt, loss=tversky_loss, loss_weights=loss_weights, metrics=[dsc])

    model.compile(optimizer=opt, loss=focal_tversky, metrics=[dice_loss])

    

    return model
input_size = (256, 1600, 1)

#adam = RAdam(3e-4, decay=0.99, min_lr=1e-6)

adam = Adam(3e-4)

es_callback = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

checkpointer = ModelCheckpoint(filepath='w_tversky_gray.hdf5', verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-6)

callbacks = [es_callback, checkpointer, reduce_lr]



model = attn_reg(adam, input_size, focal_tversky)

model.summary()
#train model

history = model.fit_generator(

    train_dataloader, 

    steps_per_epoch=len(train_dataloader), 

    epochs=15, 

    callbacks=callbacks, 

    validation_data=valid_dataloader, 

    validation_steps=len(valid_dataloader),

)

# model.load_weights("../input/weights-gray/w_tversky_gray.hdf5")

        
# test_dataset = Dataset(test_df, 'test')

# test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, subset='test')
# from tqdm import tqdm_notebook as tqdm



# idx = test_df.ImageId.unique()

# for pct in tqdm(range(len(test_dataset)), total=len(test_dataset)):

    

#     X = test_dataset[pct]

#     msk = model.predict(X[np.newaxis, ...]).squeeze()

#     msk = (msk > 0.5).astype(int)

#     suma = np.sum(msk.reshape((-1, 4)), axis=0)

#     #print(suma)

#     results = [mask2rle(msk[..., m]) if suma[m] > 1000 else ' ' for m in range(4)]

#     test_df.loc[test_df.ImageId == idx[pct], 'EncodedPixels'] = results

    

# test_df.drop('ImageId', 1, inplace=True)

# test_df.to_csv('submission.csv', index=False)
# thrs = [0.46, 0.5]

# results = {}

# for thr in thrs:

#     print(thr)

#     preds = []

#     for pct in range(100):

#         X, y = test_dataset[pct]

#         pred = model.predict(X[np.newaxis, ...])

#         score = dsc_np(y[np.newaxis, ...], (pred > thr).astype(int))

#         preds.append(score)

        

#     med = np.median(np.array(preds))

#     mean = np.mean(np.array(preds))

#     std = np.std(np.array(preds))

#     results[str(thr)] = (mean, med, std)