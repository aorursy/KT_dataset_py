# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy

from keras.applications.densenet import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score

from keras.utils import Sequence

WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

beta_f2=2
nb_classes = 1103

#batch_size = 100

img_size = 224

nb_epochs = 25
imet_path = '../input/imet-2019-fgvc6/'

#imet_path = 'F:/ChromeDownload/IMET/'



model_weights_path = '../input/densenet121weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'

#model_weights_path = 'F:/Edge Downloads/keras_model_weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'

#model_weights_path = 'F:/Edge Downloads/keras_model_weights/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'

#model_weights_path = 'F:/Edge Downloads/keras_model_weights/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'



train_df_path = imet_path + 'train.csv'

label_df_path = imet_path + 'labels.csv'

sub_df_path = imet_path + 'sample_submission.csv'

train_img_path = imet_path + 'train/'

test_img_path = imet_path + 'test/'
from keras.legacy import interfaces

from keras.optimizers import Optimizer

from keras import backend as K



class AdamAccumulate_v1(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=20, **kwargs):

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.effective_iterations = K.variable(0, dtype='int64', name='effective_iterations')



            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, dtype='int64')



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)



        self.updates = [K.update(self.iterations, self.iterations + 1)]



        flag = K.equal(self.iterations % self.accum_iters, self.accum_iters - 1)

        flag = K.cast(flag, K.floatx())



        self.updates.append(K.update(self.effective_iterations,

                                     self.effective_iterations + K.cast(flag, 'int64')))



        lr = self.lr

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.effective_iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.effective_iterations, K.floatx()) + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t)))



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, gg in zip(params, grads, ms, vs, vhats, gs):



            gg_t = (1 - flag) * (gg + g)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * (gg + flag * g) / K.cast(self.accum_iters, K.floatx())

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(

                (gg + flag * g) / K.cast(self.accum_iters, K.floatx()))



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - flag * lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, vhat_t))

            else:

                p_t = p - flag * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append((m, flag * m_t + (1 - flag) * m))

            self.updates.append((v, flag * v_t + (1 - flag) * v))

            self.updates.append((gg, gg_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))





class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., amsgrad=False, accum_iters=1, **kwargs):

        if accum_iters < 1:

            raise ValueError('accum_iters must be >= 1')

        super(AdamAccumulate, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.lr = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')

        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay

        self.amsgrad = amsgrad

        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))

        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())



    @interfaces.legacy_get_updates_support

    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.lr



        completed_updates = K.cast(K.tf.floor(self.iterations / self.accum_iters), K.floatx())



        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * completed_updates))



        t = completed_updates + 1



        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))



        # self.iterations incremented after processing a batch

        # batch:              1 2 3 4 5 6 7 8 9

        # self.iterations:    0 1 2 3 4 5 6 7 8

        # update_switch = 1:        x       x    (if accum_iters=4)

        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)

        update_switch = K.cast(update_switch, K.floatx())



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]



        if self.amsgrad:

            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        else:

            vhats = [K.zeros(1) for _ in params]



        self.weights = [self.iterations] + ms + vs + vhats



        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):



            sum_grad = tg + g

            avg_grad = sum_grad / self.accum_iters_float



            m_t = (self.beta_1 * m) + (1. - self.beta_1) * avg_grad

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(avg_grad)



            if self.amsgrad:

                vhat_t = K.maximum(vhat, v_t)

                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)

                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))

            else:

                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)



            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))

            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))

            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'amsgrad': self.amsgrad}

        base_config = super(AdamAccumulate, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
# Load dataset info

path_to_train = imet_path+'train/'

data = pd.read_csv(imet_path+'train.csv')



train_dataset_info = []

for name, labels in zip(data['id'], data['attribute_ids'].str.split(' ')):

    train_dataset_info.append({

        'path':os.path.join(path_to_train, name),

        'labels':np.array([int(label) for label in labels])})

train_dataset_info = np.array(train_dataset_info)
gamma = 2.0

epsilon = K.epsilon()

def old_focal_loss(y_true, y_pred):

    pt = y_pred * y_true + (1-y_pred) * (1-y_true)

    pt = K.clip(pt, epsilon, 1-epsilon)

    CE = -K.log(pt)

    FL = K.pow(1-pt, gamma) * CE

    loss = K.sum(FL, axis=1)

    return loss
train_df = pd.read_csv('../input/imet-2019-fgvc6/train.csv')

train_labels = np.zeros((train_df.shape[0], 1103))



for row_index, row in enumerate(train_df['attribute_ids']):

    for label in row.split():

        train_labels[row_index, int(label)] = 1

labels_sum = np.sum(train_labels,axis=0)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)



train_index = None

valid_index = None



class data_generator(Sequence):



    def mix_up(x, y):

        x = np.array(x, np.float32)

        lam = np.random.beta(0.4, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y

    

    def create_train(dataset_info, batch_size, shape, augument=True, mix=False):

        assert shape[2] == 3

        while True:

            dataset_info = shuffle(dataset_info)

            for start in range(0, len(dataset_info), batch_size):

                end = min(start + batch_size, len(dataset_info))

                batch_images = []

                X_train_batch = dataset_info[start:end]

                batch_labels = np.zeros((len(X_train_batch), nb_classes))

                for i in range(len(X_train_batch)):

                    image = data_generator.load_image(

                        X_train_batch[i]['path'], shape)   

                    if augument:

                        image = data_generator.augment(image)

                    #batch_images.append(preprocess_input(image))

                    batch_images.append((image)/255.0)

                    batch_labels[i][X_train_batch[i]['labels']] = 1

                if(mix):

                    batch_images, batch_labels = data_generator.mix_up(batch_images, batch_labels)    

                yield np.array(batch_images, np.float32), batch_labels

                

    def create_valid(valid_dataset_info, batch_size, shape, augument=False):

        global valid_index

        assert shape[2] == 3

        while True:

            #valid_dataset_info = dataset_info[valid_index]

            for start in range(0, len(valid_dataset_info), batch_size):

                end = min(start + batch_size, len(valid_dataset_info))

                batch_images = []

                X_train_batch = valid_dataset_info[start:end]

                batch_labels = np.zeros((len(X_train_batch), nb_classes))

                for i in range(len(X_train_batch)):

                    image = data_generator.load_image(

                        X_train_batch[i]['path'], shape)   

                    if augument:

                        image = data_generator.augment(image)

                    #batch_images.append(preprocess_input(image))

                    batch_images.append((image)/255.0)

                    batch_labels[i][X_train_batch[i]['labels']] = 1

                yield np.array(batch_images, np.float32), batch_labels



    def load_image(path, shape):

        #img_pil = Image.open(path+'.png')

        #img_cv = np.array(img_pil.filter(PIL.ImageFilter.FIND_EDGES))

        img_cv = cv2.imread(path+'.png')

        img_cv = cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB)

        if img_cv.shape[1]==300 and img_cv.shape[0]>=450:

            over_high = img_cv.shape[0]-300

            start = np.random.randint(over_high//3,over_high,1)[0]

            img_cv = img_cv[start:start+300,:,:]

        elif img_cv.shape[0]==300 and img_cv.shape[1]>450:

            over_width = img_cv.shape[1]-300

            start = np.random.randint(over_width//3,over_width,1)[0]

            img_cv = img_cv[:,start:start+300,:]

        img_cv = cv2.resize(img_cv,(img_size,img_size))

        return img_cv



    def augment(image):

        augment_img = iaa.Sequential([

            iaa.SomeOf((0,4),[

                #iaa.Crop(percent=(0, 0.1)),

                iaa.ContrastNormalization((0.5, 1.5)),

                iaa.Multiply((0.9, 1.1), per_channel=0.2),

                iaa.Fliplr(0.5),

                iaa.GaussianBlur(sigma=(0, 0.6)),

                #iaa.Affine(

                #        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},

                #        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},

                #        rotate=(-10, 10),

                #    )

            ])], random_order=True)



        image_aug = augment_img.augment_image(image)

        return image_aug
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,Concatenate)

from keras import applications

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
# reference link: https://gist.github.com/drscotthawley/d1818aabce8d1bf082a6fb37137473ae

from keras.callbacks import Callback



def get_1cycle_schedule(lr_max=1e-3, n_data_points=8000, epochs=200, batch_size=40, verbose=0):          

    """

    Creates a look-up table of learning rates for 1cycle schedule with cosine annealing

    See @sgugger's & @jeremyhoward's code in fastai library: https://github.com/fastai/fastai/blob/master/fastai/train.py

    Wrote this to use with my Keras and (non-fastai-)PyTorch codes.

    Note that in Keras, the LearningRateScheduler callback (https://keras.io/callbacks/#learningratescheduler) only operates once per epoch, not per batch

      So see below for Keras callback



    Keyword arguments:

    lr_max            chosen by user after lr_finder

    n_data_points     data points per epoch (e.g. size of training set)

    epochs            number of epochs

    batch_size        batch size

    Output:  

    lrs               look-up table of LR's, with length equal to total # of iterations

    Then you can use this in your PyTorch code by counting iteration number and setting

          optimizer.param_groups[0]['lr'] = lrs[iter_count]

    """

    if verbose > 0:

        print("Setting up 1Cycle LR schedule...")

    pct_start, div_factor = 0.3, 25.        # @sgugger's parameters in fastai code

    lr_start = lr_max/div_factor

    lr_end = lr_start/1e4

    n_iter = (n_data_points * epochs // batch_size) + 1    # number of iterations

    a1 = int(n_iter * pct_start)

    a2 = n_iter - a1



    # make look-up table

    lrs_first = np.linspace(lr_start, lr_max, a1)            # linear growth

    lrs_second = (lr_max-lr_end)*(1+np.cos(np.linspace(0,np.pi,a2)))/2 + lr_end  # cosine annealing

    lrs = np.concatenate((lrs_first, lrs_second))

    return lrs





class OneCycleScheduler(Callback):

    """My modification of Keras' Learning rate scheduler to do 1Cycle learning

       which increments per BATCH, not per epoch

    Keyword arguments

        **kwargs:  keyword arguments to pass to get_1cycle_schedule()

        Also, verbose: int. 0: quiet, 1: update messages.



    Sample usage (from my train.py):

        lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=X_train.shape[0],

        epochs=epochs, batch_size=batch_size, verbose=1)

    """

    def __init__(self, **kwargs):

        super(OneCycleScheduler, self).__init__()

        self.verbose = kwargs.get('verbose', 0)

        self.lrs = get_1cycle_schedule(**kwargs)

        self.iteration = 0



    def on_batch_begin(self, batch, logs=None):

        lr = self.lrs[self.iteration]

        K.set_value(self.model.optimizer.lr, lr)         # here's where the assignment takes place

        if self.verbose > 0:

            print('\nIteration %06d: OneCycleScheduler setting learning '

                  'rate to %s.' % (self.iteration, lr))

        self.iteration += 1



    def on_epoch_end(self, epoch, logs=None):  # this is unchanged from Keras LearningRateScheduler

        logs = logs or {}

        logs['lr'] = K.get_value(self.model.optimizer.lr)

        self.iteration = 0



from keras.callbacks import Callback

import keras.backend as K

import numpy as np

import matplotlib.pyplot as plt





class CyclicLR(Callback):

    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with

    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).

    The amplitude of the cycle can be scaled on a per-iteration or 

    per-cycle basis.

    This class has three built-in policies, as put forth in the paper.

    "triangular":

        A basic triangular cycle w/ no amplitude scaling.

    "triangular2":

        A basic triangular cycle that scales initial amplitude by half each cycle.

    "exp_range":

        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 

        cycle iteration.

    For more detail, please see paper.    

    # Example

        ```python

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., mode='triangular')

            model.fit(X_train, Y_train, callbacks=[clr])

        ```    

    Class also supports custom scaling functions:

        ```python

            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))

            clr = CyclicLR(base_lr=0.001, max_lr=0.006,

                                step_size=2000., scale_fn=clr_fn,

                                scale_mode='cycle')

            model.fit(X_train, Y_train, callbacks=[clr])

            ```    

    # Arguments

        base_lr: initial learning rate which is the

            lower boundary in the cycle.

        max_lr: upper boundary in the cycle. Functionally,

            it defines the cycle amplitude (max_lr - base_lr).

            The lr at any cycle is the sum of base_lr

            and some scaling of the amplitude; therefore 

            max_lr may not actually be reached depending on

            scaling function.

        step_size: number of training iterations per

            half cycle. Authors suggest setting step_size

            2-8 x training iterations in epoch.

        mode: one of {triangular, triangular2, exp_range}.

            Default 'triangular'.

            Values correspond to policies detailed above.

            If scale_fn is not None, this argument is ignored.

        gamma: constant in 'exp_range' scaling function:

            gamma**(cycle iterations)

        scale_fn: Custom scaling policy defined by a single

            argument lambda function, where 

            0 <= scale_fn(x) <= 1 for all x >= 0.

            mode paramater is ignored 

        scale_mode: {'cycle', 'iterations'}.

            Defines whether scale_fn is evaluated on 

            cycle number or cycle iterations (training

            iterations since start of cycle). Default is 'cycle'.

    """

    def __init__(self, base_lr=1e-5, max_lr=1e-2, step_size=1000., mode='triangular',

                 gamma=1., scale_fn=None, scale_mode='cycle'):

        super(CyclicLR, self).__init__()

        self.base_lr = base_lr

        self.max_lr = max_lr

        self.step_size = step_size

        self.mode = mode

        self.gamma = gamma

        if scale_fn == None:

            if self.mode == 'triangular':

                self.scale_fn = lambda x: 1.

                self.scale_mode = 'cycle'

            elif self.mode == 'triangular2':

                self.scale_fn = lambda x: 1/(2.**(x-1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma**(x)

                self.scale_mode = 'iterations'

        else:

            self.scale_fn = scale_fn

            self.scale_mode = scale_mode

        self.clr_iterations = 0.

        self.trn_iterations = 0.

        self.history = {}

        self._reset()

        

    def _reset(self, new_base_lr=None, new_max_lr=None,

               new_step_size=None):

        """Resets cycle iterations.

        Optional boundary/step size adjustment.

        """

        if new_base_lr != None:

            self.base_lr = new_base_lr

        if new_max_lr != None:

            self.max_lr = new_max_lr

        if new_step_size != None:

            self.step_size = new_step_size

        self.clr_iterations = 0.        

        

    def clr(self):

        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))

        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)        



    def on_train_begin(self, logs={}):

        logs = logs or {}

        if self.clr_iterations == 0:

            K.set_value(self.model.optimizer.lr, self.base_lr)

        else:

            K.set_value(self.model.optimizer.lr, self.clr())        

       

    def on_batch_end(self, epoch, logs=None):        

        logs = logs or {}

        self.trn_iterations += 1

        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)        

        K.set_value(self.model.optimizer.lr, self.clr())        
from keras.regularizers import l2



def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    #base_model = ResNet50V2(include_top=False,weights=None,input_tensor=input_tensor)

    #base_model.load_weights('../input/keras-pretrain-model-weights/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

#     x = Conv2D(32, kernel_size=(1,1), activation='relu')(base_model.output)

#     x = Flatten()(x)

    model = applications.DenseNet121(weights=None,include_top=False,input_tensor=input_tensor)

    model.load_weights(model_weights_path)

    x0 = model.output

    x1 = GlobalAveragePooling2D()(x0)

    x2 = GlobalMaxPooling2D()(x0)

    x = Concatenate()([x1,x2])

    

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    

    #x = GlobalAveragePooling2D()(base_model.output)

    #x = Dropout(0.5)(x)

    #x = Dense(1024, activation='relu', kernel_regularizer=l2(5e-4))(x)

    #x = Dropout(0.5)(x)

    final_output = Dense(n_out, activation='sigmoid', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)

                             

from sklearn.model_selection import train_test_split



epochs = 17; warmup_batch_size = 128;mixup_batch_size = 64

batch_size = 64

checkpoint = ModelCheckpoint('denseNet121_focal.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=7)



csv_logger = CSVLogger(filename='training_log.csv',

                       separator=',',

                       append=True)





# split data into train, valid

indexes = np.arange(train_dataset_info.shape[0])

train_indexes, valid_indexes = train_test_split(indexes, test_size=0.15, random_state=8)



# create train and valid datagens

train_generator = data_generator.create_train(

    train_dataset_info[train_indexes], batch_size, (img_size,img_size,3), augument=True, mix=False)

train_mixup = data_generator.create_train(

    train_dataset_info[train_indexes], mixup_batch_size, (img_size,img_size,3), augument=False, mix=True)

train_generator_warmup = data_generator.create_train(

    train_dataset_info[train_indexes], warmup_batch_size, (img_size,img_size,3), augument=False)

validation_generator = data_generator.create_valid(

    train_dataset_info[valid_indexes], batch_size, (img_size,img_size,3), augument=False)



lrsched = OneCycleScheduler(lr_max=1e-4, n_data_points=len(train_indexes),

        epochs=1, batch_size=warmup_batch_size, verbose=0)

# callbacks_list = [checkpoint, csv_logger, lrsched]

# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]
model = create_model(

    input_shape=(img_size,img_size,3), 

    n_out=nb_classes)
%%time

# warm up model

for layer in model.layers:

    layer.trainable = False



for i in range(-6,0):

    model.layers[i].trainable = True



model.compile(

    loss='binary_crossentropy',

    #loss='categorical_crossentropy',

    #loss = [focal_loss(labels_sum)],

    optimizer=Adam(1e-3))

    # optimizer=AdamAccumulate(lr=1e-3, accum_iters=2))



model.fit_generator(

    train_generator_warmup,

    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(warmup_batch_size)),

    epochs=2,

    max_queue_size=16, 

    workers=2, 

    use_multiprocessing=True,

    verbose=1,

    class_weight = 'auto'

    #callbacks = lrsched

)
%%time

# train all layers

for layer in model.layers:

    layer.trainable = True



# callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]

model.compile(

            loss='binary_crossentropy',

            #loss='categorical_crossentropy',

            #loss = [focal_loss(labels_sum)],

             #loss=focal_loss,

            #optimizer=Adam(lr=1e-4))

            optimizer=AdamAccumulate(lr=1e-4, accum_iters=4))



model.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(mixup_batch_size)),

    # validation_data=validation_generator,

    # validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),

    epochs=7,

    verbose=1,

    class_weight = 'auto',

    max_queue_size=16, 

    workers=WORKERS, 

    use_multiprocessing=True)

    # callbacks=callbacks_list)
%%time

clr = CyclicLR(base_lr=1e-5,max_lr=6e-5,mode='triangular2', step_size=np.ceil(len(train_indexes)/batch_size))



model.compile(

    loss='binary_crossentropy',

    #loss='categorical_crossentropy',

    #loss = [focal_loss(labels_sum)],

     #loss=focal_loss,

    optimizer=Adam(lr=3e-5))

    # optimizer=AdamAccumulate(lr=1e-4, accum_iters=4))



#callbacks_list = [checkpoint, csv_logger, clr]

callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]

model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_indexes)) / float(batch_size)),

    validation_data=validation_generator,

    validation_steps=np.ceil(float(len(valid_indexes)) / float(batch_size)),

    epochs=epochs,

    verbose=1,

    class_weight = 'auto',

    max_queue_size=16, workers=WORKERS, use_multiprocessing=True,

    callbacks=callbacks_list)
print(os.listdir('../working/'))

model.save('../input/denseNet121.h5')
#model.summary()
submit = pd.read_csv('../input/imet-2019-fgvc6/sample_submission.csv')

#model = keras.models.load_model('../input/denseNet121.h5',custom_objects={'focal_loss':focal_loss})

predicted = []
'''Search for the best threshold regarding the validation set'''



BATCH = 512

fullValGen = data_generator.create_valid(

    train_dataset_info[valid_indexes], BATCH, (img_size,img_size,3))



n_val = round(train_dataset_info.shape[0]*0.15)//BATCH

print(n_val)



lastFullValPred = np.empty((0, nb_classes))

lastFullValLabels = np.empty((0, nb_classes))

for i in tqdm(range(n_val+1)): 

    im, lbl = next(fullValGen)

    scores = model.predict(im)

    lastFullValPred = np.append(lastFullValPred, scores, axis=0)

    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)

print(lastFullValPred.shape, lastFullValLabels.shape)
def my_f2(y_true, y_pred):

    assert y_true.shape[0] == y_pred.shape[0]



    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1)

    tn = np.sum((y_true == 0) & (y_pred == 0), axis=1)

    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1)

    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f2 = (1+beta_f2**2)*p*r / (p*beta_f2**2 + r + 1e-15)



    return np.mean(f2)



def find_best_fixed_threshold(preds, targs, do_plot=True):

    score = []

    thrs = np.arange(0, 0.5, 0.01)

    for thr in tqdm(thrs):

        score.append(my_f2(targs, (preds > thr).astype(int) ))

    score = np.array(score)

    pm = score.argmax()

    best_thr, best_score = thrs[pm], score[pm].item()

    print(f'thr={best_thr:.3f}', f'F2={best_score:.3f}')

    if do_plot:

        plt.plot(thrs, score)

        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())

        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);

        plt.show()

    return best_thr, best_score
best_thr, best_score = find_best_fixed_threshold(lastFullValPred, lastFullValLabels, do_plot=True)
for i, name in tqdm(enumerate(submit['id'])):

    path = os.path.join('../input/imet-2019-fgvc6/test/', name)

    image = data_generator.load_image(path, (img_size,img_size,3))

    #score_predict = model.predict(preprocess_input(image[np.newaxis]))

    score_predict = model.predict((image[np.newaxis])/255.0)

    # print(score_predict)

    label_predict = np.arange(nb_classes)[score_predict[0]>=best_thr]

    # print(label_predict)

    str_predict_label = ' '.join(str(l) for l in label_predict)

    predicted.append(str_predict_label)
submit['attribute_ids'] = predicted

submit.to_csv('submission.csv', index=False)