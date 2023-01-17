# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



!pip install segmentation-models --quiet

!pip install git+https://github.com/HelioStrike/pathomics-research-utils.git



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import segmentation_models as sm



# Any results you write to the current directory are saved as output.
#!wget http://andrewjanowczyk.com/wp-static/nuclei.tgz

#!tar -zxvf nuclei.tgz

#!mkdir nuclei

#!mv *.tif nuclei

#!mv *.png nuclei

#os.listdir('.')
class RAdam(tf.keras.optimizers.Optimizer):

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
def vanilla_unet(input_shape):

    """

    This is the old model. Best LB is ~0.5

    """

    inputs = Input(input_shape)



    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (inputs)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same') (c1)

    p1 = MaxPooling2D((2, 2), padding='same') (c1)



    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (p1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same') (c2)

    p2 = MaxPooling2D((2, 2), padding='same') (c2)



    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (p2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same') (c3)

    p3 = MaxPooling2D((2, 2), padding='same') (c3)



    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (p3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same') (c4)

    p4 = MaxPooling2D((2, 2), padding='same') (c4)



    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (p4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same') (c5)

    p5 = MaxPooling2D((2, 2), padding='same') (c5)



    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (p5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same') (c55)



    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)

    u6 = concatenate([u6, c5])

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (u6)

    c6 = Conv2D(64, (3, 3), activation='elu', padding='same') (c6)



    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)

    u71 = concatenate([u71, c4])

    c71 = Conv2D(32, (3, 3), activation='elu', padding='same') (u71)

    c61 = Conv2D(32, (3, 3), activation='elu', padding='same') (c71)



    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (u7)

    c7 = Conv2D(32, (3, 3), activation='elu', padding='same') (c7)



    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (u8)

    c8 = Conv2D(16, (3, 3), activation='elu', padding='same') (c8)



    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (u9)

    c9 = Conv2D(8, (3, 3), activation='elu', padding='same') (c9)



    outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)



    model = Model(inputs=[inputs], outputs=[outputs])

    

    return model
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