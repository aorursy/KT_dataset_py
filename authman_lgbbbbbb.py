# new submission??

# - how much free space exists on the side (Y) of the field the risher is more on. this can be dist to the nearest defender from the y-side

# - [building] LIGHTGBM

# - [todo] divisor=1, timing...

# - [todo] stack ridge,lasso,survival model linear models?



# - AUG: sample features from columns where the yards match to create pseudo samples

# - FE: rusher_dist_td or other rusher_X type inputs....

# - FE: rusherA; typically moving close to max speed in intended direction





# FINAL SUB

# n_splits = 10

# 2019 as new season or as same season...

# KFold looks like it won

# divisor=1 is worse but trains faster, maybe use divisor==3?



RUN_KAGGLE = True

DO_LGBM = True



DIVISOR = 2

n_runs = 4 # n-1 will be kept.

n_splits = 10 # Use 10 for final sub!!!!!!!





# Cols we use to bound predictions:

bounding_cols = ['rusher_X','rusher_SX','YardLine']



# Derived by: X.Yards.value_counts().sort_index()[0:50]

MIN_CLIP = -9

MAX_CLIP = 70



N_OUTPUTS = MAX_CLIP - MIN_CLIP + 1

N_OUTPUTS
import logging

import numpy as np

import pandas as pd

import lightgbm as lgb

import sklearn.metrics as mtr

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from keras.layers import Dense

from keras.models import Sequential

from sklearn.utils import class_weight

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LambdaCallback

from keras.models import Model

from keras.optimizers import Nadam

from keras import losses

from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, merge, Add, BatchNormalization, Lambda, GaussianNoise, Layer

from keras.layers.embeddings import Embedding

from keras.models import load_model

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import KFold,GroupKFold

import warnings, math, numba

import random as rn

from time import time



import tensorflow as tf

import keras.backend as K



from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from tensorflow.python import ops, math_ops, state_ops, control_flow_ops



from scipy.spatial import Voronoi, ConvexHull, voronoi_plot_2d



import os, gc

warnings.filterwarnings("ignore")



import seaborn as sns

import matplotlib.patches as patches

import matplotlib.pyplot as plt



# from kaggle.competitions import nflrush

# env = nflrush.make_env()

# iter_test = env.iter_test()
if RUN_KAGGLE:

    train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', parse_dates=['TimeHandoff','TimeSnap','PlayerBirthDate'], dtype={'WindSpeed': 'object'})

else:

    train = pd.read_csv("../train.csv",  low_memory=False, parse_dates=['TimeHandoff','TimeSnap','PlayerBirthDate'], dtype={'WindSpeed': 'object'})

    test = pd.read_csv("../df_test.csv", low_memory=False, parse_dates=['TimeHandoff','TimeSnap','PlayerBirthDate'], dtype={'WindSpeed': 'object'})
num_rounds = 1000

params = {

#     'objective':'multiclass',

#     'num_class': N_OUTPUTS,

#     'learning_rate': 0.01,

#     "boosting": "gbdt",

#     "metric": "multi_logloss",

#     "verbosity": -1,

#     "seed":1234

    

    

    'num_leaves': 50, #Original 50

    'min_data_in_leaf': 30, #Original 30

    'objective':'multiclass',

    'num_class': N_OUTPUTS,

    'max_depth': -1,

    'learning_rate': 0.01,

    "min_child_samples": 20,

    "boosting": "gbdt",

    "feature_fraction": 0.7, #0.9

    "bagging_freq": 1,

    "bagging_fraction": 0.9,

    "bagging_seed": 11,

    

    "lambda_l1": 0.1,

    "verbosity": -1,

    "seed":1234,

    "metric": "multi_logloss",

}



cat_feats = [

    'Season','InfluenceRusherX_flip','def_triCon',

    'oline_num_def_inbox','defense_Y','Congestion',

]



# reg_cols

lgb_cols = [

    'Season', 'rusher_A', 'rusher_SX', 'avg_def9_dist_to_rusher', 'rusher_dist_scrimmage',

    'rusher_S', 'avg_off5_dist_to_rusher', 'min_off5_dist_to_rusher',

    'dist_next_point_time', 'min_def9_dist_to_rusher', 'std_off5_dist_to_rusher',

    'defense_scrimmage_Y_std', 'std_def9_dist_to_rusher', 'Dis', 'closest_def_S',

    'closest_def_A', 'defense_Y', 'InfluenceRusherX_flip', 'Congestion', 'def_triCon',

    'vArea', 'avg_def9_A', 'avg_def9_S', 'std_def9_S', 'min_def9_S',

    'RusherMaxObservedS', 'Mean_SX','Mean_AX','Mean_WX',

    'oline_length', 'oline_bline_area', 'oline_num_def_inbox', 'oline_avg_ol_dist',



#     # For diversity:

#     'def9_dist_to_rusher_0',

#     'def9_dist_to_rusher_1', 'def9_dist_to_rusher_2',

#     'def9_dist_to_rusher_3', 'def9_dist_to_rusher_4',

#     'def9_dist_to_rusher_5', 'def9_dist_to_rusher_6',

#     'def9_dist_to_rusher_7', 'def9_dist_to_rusher_8',

]
class RAdam(OptimizerV2):

    """RAdam optimizer.

    According to the paper

    [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).

    """



    def __init__(self,

                 learning_rate=0.001,

                 beta_1=0.9,

                 beta_2=0.999,

                 epsilon=1e-7,

                 weight_decay=0.,

                 amsgrad=False,

                 total_steps=0,

                 warmup_proportion=0.1,

                 min_lr=0.,

                 name='RAdam',

                 **kwargs):

        r"""Construct a new Adam optimizer.

        Args:

            learning_rate: A Tensor or a floating point value.    The learning rate.

            beta_1: A float value or a constant float tensor. The exponential decay

                rate for the 1st moment estimates.

            beta_2: A float value or a constant float tensor. The exponential decay

                rate for the 2nd moment estimates.

            epsilon: A small constant for numerical stability. This epsilon is

                "epsilon hat" in the Kingma and Ba paper (in the formula just before

                Section 2.1), not the epsilon in Algorithm 1 of the paper.

            weight_decay: A floating point value. Weight decay for each param.

            amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from

                the paper "On the Convergence of Adam and beyond".

            total_steps: An integer. Total number of training steps.

                Enable warmup by setting a positive value.

            warmup_proportion: A floating point value. The proportion of increasing steps.

            min_lr: A floating point value. Minimum learning rate after warmup.

            name: Optional name for the operations created when applying gradients.

                Defaults to "Adam".    @compatibility(eager) When eager execution is

                enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each be

                a callable that takes no arguments and returns the actual value to use.

                This can be useful for changing these values across different

                invocations of optimizer functions. @end_compatibility

            **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,

                `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip

                gradients by value, `decay` is included for backward compatibility to

                allow time inverse decay of learning rate. `lr` is included for backward

                compatibility, recommended to use `learning_rate` instead.

        """



        super(RAdam, self).__init__(name, **kwargs)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

        self._set_hyper('beta_1', beta_1)

        self._set_hyper('beta_2', beta_2)

        self._set_hyper('decay', self._initial_decay)

        self._set_hyper('weight_decay', weight_decay)

        self._set_hyper('total_steps', float(total_steps))

        self._set_hyper('warmup_proportion', warmup_proportion)

        self._set_hyper('min_lr', min_lr)

        self.epsilon = epsilon or K.epsilon()

        self.amsgrad = amsgrad

        self._initial_weight_decay = weight_decay

        self._initial_total_steps = total_steps



    def _create_slots(self, var_list):

        for var in var_list:

            self.add_slot(var, 'm')

        for var in var_list:

            self.add_slot(var, 'v')

        if self.amsgrad:

            for var in var_list:

                self.add_slot(var, 'vhat')



    def set_weights(self, weights):

        params = self.weights

        num_vars = int((len(params) - 1) / 2)

        if len(weights) == 3 * num_vars + 1:

            weights = weights[:len(params)]

        super(RAdam, self).set_weights(weights)



    def _resource_apply_dense(self, grad, var):

        var_dtype = var.dtype.base_dtype

        lr_t = self._decayed_lr(var_dtype)

        m = self.get_slot(var, 'm')

        v = self.get_slot(var, 'v')

        beta_1_t = self._get_hyper('beta_1', var_dtype)

        beta_2_t = self._get_hyper('beta_2', var_dtype)

        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)

        beta_1_power = math_ops.pow(beta_1_t, local_step)

        beta_2_power = math_ops.pow(beta_2_t, local_step)



        if self._initial_total_steps > 0:

            total_steps = self._get_hyper('total_steps', var_dtype)

            warmup_steps = total_steps * self._get_hyper('warmup_proportion', var_dtype)

            min_lr = self._get_hyper('min_lr', var_dtype)

            decay_steps = K.maximum(total_steps - warmup_steps, 1)

            decay_rate = (min_lr - lr_t) / decay_steps

            lr_t = tf.where(

                local_step <= warmup_steps,

                lr_t * (local_step / warmup_steps),

                lr_t + decay_rate * K.minimum(local_step - warmup_steps, decay_steps),

            )



        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0

        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)



        m_t = state_ops.assign(m,

                               beta_1_t * m + (1.0 - beta_1_t) * grad,

                               use_locking=self._use_locking)

        m_corr_t = m_t / (1.0 - beta_1_power)



        v_t = state_ops.assign(v,

                               beta_2_t * v + (1.0 - beta_2_t) * math_ops.square(grad),

                               use_locking=self._use_locking)

        if self.amsgrad:

            vhat = self.get_slot(var, 'vhat')

            vhat_t = state_ops.assign(vhat,

                                      math_ops.maximum(vhat, v_t),

                                      use_locking=self._use_locking)

            v_corr_t = math_ops.sqrt(vhat_t / (1.0 - beta_2_power))

        else:

            vhat_t = None

            v_corr_t = math_ops.sqrt(v_t / (1.0 - beta_2_power))



        r_t = math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *

                            (sma_t - 2.0) / (sma_inf - 2.0) *

                            sma_inf / sma_t)



        var_t = tf.where(sma_t >= 5.0, r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t)



        if self._initial_weight_decay > 0.0:

            var_t += self._get_hyper('weight_decay', var_dtype) * var



        var_update = state_ops.assign_sub(var,

                                          lr_t * var_t,

                                          use_locking=self._use_locking)



        updates = [var_update, m_t, v_t]

        if self.amsgrad:

            updates.append(vhat_t)

        return control_flow_ops.group(*updates)



    def _resource_apply_sparse(self, grad, var, indices):

        var_dtype = var.dtype.base_dtype

        lr_t = self._decayed_lr(var_dtype)

        beta_1_t = self._get_hyper('beta_1', var_dtype)

        beta_2_t = self._get_hyper('beta_2', var_dtype)

        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)

        beta_1_power = math_ops.pow(beta_1_t, local_step)

        beta_2_power = math_ops.pow(beta_2_t, local_step)



        if self._initial_total_steps > 0:

            total_steps = self._get_hyper('total_steps', var_dtype)

            warmup_steps = total_steps * self._get_hyper('warmup_proportion', var_dtype)

            min_lr = self._get_hyper('min_lr', var_dtype)

            decay_steps = K.maximum(total_steps - warmup_steps, 1)

            decay_rate = (min_lr - lr_t) / decay_steps

            lr_t = tf.where(

                local_step <= warmup_steps,

                lr_t * (local_step / warmup_steps),

                lr_t + decay_rate * K.minimum(local_step - warmup_steps, decay_steps),

            )



        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0

        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)



        m = self.get_slot(var, 'm')

        m_scaled_g_values = grad * (1 - beta_1_t)

        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)

        with ops.control_dependencies([m_t]):

            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        m_corr_t = m_t / (1.0 - beta_1_power)



        v = self.get_slot(var, 'v')

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)

        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)

        with ops.control_dependencies([v_t]):

            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)



        if self.amsgrad:

            vhat = self.get_slot(var, 'vhat')

            vhat_t = state_ops.assign(vhat,

                                      math_ops.maximum(vhat, v_t),

                                      use_locking=self._use_locking)

            v_corr_t = math_ops.sqrt(vhat_t / (1.0 - beta_2_power))

        else:

            vhat_t = None

            v_corr_t = math_ops.sqrt(v_t / (1.0 - beta_2_power))



        r_t = math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *

                            (sma_t - 2.0) / (sma_inf - 2.0) *

                            sma_inf / sma_t)



        var_t = tf.where(sma_t >= 5.0, r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t)



        if self._initial_weight_decay > 0.0:

            var_t += self._get_hyper('weight_decay', var_dtype) * var



        var_update = self._resource_scatter_add(var, indices, tf.gather(-lr_t * var_t, indices))



        updates = [var_update, m_t, v_t]

        if self.amsgrad:

            updates.append(vhat_t)

        return control_flow_ops.group(*updates)



    def get_config(self):

        config = super(RAdam, self).get_config()

        config.update({

            'learning_rate': self._serialize_hyperparameter('learning_rate'),

            'beta_1': self._serialize_hyperparameter('beta_1'),

            'beta_2': self._serialize_hyperparameter('beta_2'),

            'decay': self._serialize_hyperparameter('decay'),

            'weight_decay': self._serialize_hyperparameter('weight_decay'),

            'epsilon': self.epsilon,

            'amsgrad': self.amsgrad,

            'total_steps': self._serialize_hyperparameter('total_steps'),

            'warmup_proportion': self._serialize_hyperparameter('warmup_proportion'),

            'min_lr': self._serialize_hyperparameter('min_lr'),

        })

        return config
def build_pidf(pis):

    if len(pis)==0:

        return None

    

    keys = list(pis[0].keys())

    pidf = pd.DataFrame({

        'cols':keys,

        'means':[np.nanmean([pi[key] for pi in pis]) for key in keys],

        'medians':[np.nanmedian([pi[key] for pi in pis]) for key in keys],

        'stds':[np.nanstd([pi[key] for pi in pis]) for key in keys]

    })

    pidf['std_mean_ratio'] = np.round(pidf.stds / pidf.means * 100, 2)

    pidf.sort_values(['means', 'std_mean_ratio'], inplace=True, ascending=False)

    return pidf
def plot_feature_importance(perms, cutoff=None):

    pidf_cvs = pd.DataFrame({

        'cols': sum([list(fold.keys()) for fold in perms], []),

        'vals': sum([list(fold.values()) for fold in perms], [])

    }).sort_values('vals', ascending=False)

    

    if cutoff is not None:

        cutoff_df = pidf_cvs.groupby('cols').vals.mean().sort_values().reset_index()

        pidf_cvs = pidf_cvs[pidf_cvs.cols.isin(cutoff_df.iloc[:-cutoff].cols)]

        cutoff_df.rename(columns={'vals':'sortorder'}, inplace=True)

        pidf_cvs = pidf_cvs.merge(cutoff_df, how='left', on='cols')

        pidf_cvs.sort_values('sortorder', ascending=False)

        

    fig, ax = plt.subplots(figsize=(8, 8))

    #pidf_cvs.plot.barh(ax=ax)

    #fig.show()

    sns.barplot(y="cols", x="vals", data=pidf_cvs)
# evaluation metric



@numba.jit(numba.f8[:](numba.f8[:]))

def norm(x):

    return x / x.sum(axis=1).reshape(-1,1)

    

@numba.jit(numba.f8[:](numba.f8[:]))

def cumsum(x):

    return np.clip(np.cumsum(x, axis=1), 0, 1)



@numba.jit(numba.f8[:](numba.f8[:], numba.f8[:]))

def crps(y_pred, y_true):

    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (y_true.shape[1] * y_true.shape[0]) 
# author : nlgn

# Link : https://www.kaggle.com/kingychiu/keras-nn-starter-crps-early-stopping

class Metric(Callback):

    def __init__(self, model, callbacks, data):

        super().__init__()

        self.model = model

        self.callbacks = callbacks

        self.data = data

        self.epochnum = 0



    def on_train_begin(self, logs=None):

        self.epochnum = 0

        for callback in self.callbacks:

            callback.on_train_begin(logs)



    def on_train_end(self, logs=None):

        for callback in self.callbacks:

            callback.on_train_end(logs)



    def set_original_data(self, orig_train, orig_valid):

        global bounding_cols

        self.orig_train = orig_train[bounding_cols].copy()

        self.orig_valid = orig_valid[bounding_cols].copy()

        

    def bound_prediction(self, y_pred, non_scaled_data, ACCEPTABLE_DIST_FROM_ARGMAX=30):

        """

        Zero out values we cannot get to



        Look for the Yard Line Value corresponding to the position we believe

        we will not go behind

        """

        

        global N_OUTPUTS

        

        # Zero out any y_preds less than this value

        indices = np.repeat(np.arange(N_OUTPUTS).reshape(1,N_OUTPUTS), y_pred.shape[0], axis=0)



        # NOTE: `rusher_dist_scrimmage = YardLine - rusher_X`

        

        # We cannot lose more than YardLine yards. Furthermore, we can't lose more

        # than 3 yards less than where the rusher is currently positioned.

        rusher_X = non_scaled_data.rusher_X.astype(int).values

        YardLine = non_scaled_data.YardLine.values

        Slippage = (non_scaled_data.rusher_SX.values > 0).astype(int)



        # We can lose up to our rusher position yards and 3 extra

        # However if rusher is currently moving in the +x direction

        # Then that 3 extra turns into 2

        # We also bound to the stadium (e.g. 1 yard line)

        MaxLossYardLine = np.maximum(0, rusher_X - 3 + Slippage)



        # Convert MaxLossYardLine to frame of reference of the actual YardLine

        MaxLossYards = (MaxLossYardLine - YardLine - MIN_CLIP).reshape(-1,1)

        y_pred[indices < MaxLossYards] = 0

        



        # We cannot gain more yards than 100-YardLine, e.g. dist_to_td

        MaxGainYards = (100 - YardLine - MIN_CLIP).reshape(-1,1)

        y_pred[indices > MaxGainYards] = 0

        

        # TODO: Revisit:

        # Any value > ACCEPTABLE_DIST_FROM_ARGMAX units from our argmax is zero'd

        # maxes = np.argmax(y_pred, axis=1)

        #y_pred[indices > (maxes+ACCEPTABLE_DIST_FROM_ARGMAX).reshape(-1,1)] = 0

        #y_pred[indices < (maxes-ACCEPTABLE_DIST_FROM_ARGMAX).reshape(-1,1)] = 0

        # NOTE: We don't have to use amax..... we can use mu, which is one of our predicted values...

        

        # TODO: Expderiment doing this @ convert stage

        return norm(y_pred)

        #return y_pred#norm(y_pred)

        

    def convert_to_y199_cumsum(self, y_pred):

        output = np.zeros((y_pred.shape[0], 199))

        output[:,99+MIN_CLIP:99+MAX_CLIP+1] = y_pred

        

        #csum = np.clip(np.cumsum(output, axis=1), 0, 1)

        csum = cumsum(output)

        

        # This is our best guess for the value (better than argmax)

        # We might have to +1

        #cross90 = np.argmax(csum>0.95, axis=1)

#         cross10 = np.argmax(csum>0.05, axis=1)

        

#         # All eggs in one basket:

#         mask = np.repeat(np.arange(199).reshape(1,-1), y_pred.shape[0], axis=0)

#         output[mask < cross50.reshape(-1,1) - 4] = 0

#         output[mask > cross50.reshape(-1,1) + 8] = 0

#         csum = cumsum(output)

        

#         mask = np.repeat(np.arange(199).reshape(1,-1), y_pred.shape[0], axis=0)

#         output[mask < cross10.reshape(-1,1)] /= 2

        #output[mask > cross90.reshape(-1,1)] /= 2

        

#         csum = cumsum(output)

        # Alternatively try it again, looks like our indices were wrong!!!

        

        return csum

        

            

        '''

        We've discovered that if we look at where we cross the 50% boundary, it seems that

        value is typically 1 unit shy of the actual real answer, on average.

        

        How to leverage this?

        - Increase the amount predicted at this 50% crossover?

        - shift the crossover?

        - shift the value after the crossover?

        - other?

            err_ent = np.argmax(oof>0.5, axis=1) - 99

            current_err = (X.Yards.values - err_ent).sum()

            patch_err = (X.Yards.values - (err_ent+1)).sum()



            current_err, patch_err

        '''

        return csum

    def on_epoch_end(self, batch, logs=None):

        self.epochnum += 1

        X_train, y_train = self.data[0][0], self.data[0][1]['ent']

        y_pred = self.model.predict(X_train, batch_size=1024)[0]

        y_pred = self.bound_prediction(y_pred, self.orig_train)

        y_pred = self.convert_to_y199_cumsum(y_pred)

        y_train = self.convert_to_y199_cumsum(y_train)

        tr_s = crps(y_pred, y_train)

        tr_s = np.round(tr_s, 6)

        logs['tr_CRPS'] = tr_s



        

        X_valid, y_valid = self.data[1][0], self.data[1][1]['ent']

        y_pred = self.model.predict(X_valid, batch_size=1024)[0]

        y_pred = self.bound_prediction(y_pred, self.orig_valid)

        y_pred = self.convert_to_y199_cumsum(y_pred)

        y_valid = self.convert_to_y199_cumsum(y_valid)

        val_s = crps(y_pred, y_valid)

        val_s = np.round(val_s, 6)

        logs['val_CRPS'] = val_s

        

        print(f'{self.epochnum}\ttCRPS: {tr_s}\tvCRPS: {val_s}')



        for callback in self.callbacks:

            callback.on_epoch_end(batch, logs)
# TODO: Note - looks like this one is returning something different from the one in show_voroni!!!

def calcvs(play):

    xy = play[['X', 'Y']].values

    rusher = play.IsRusher.values

    vor = Voronoi(xy)



    # PROBLEMATIC - doesnt seem like we're getting the right index always.

    rx_idx = np.argmax(rusher)

    region = vor.regions[vor.point_region[rx_idx]]

    rx_poly = np.array([vor.vertices[i] for i in region])



    try:

        ch = ConvexHull(rx_poly)

        return ch.area

    except:

        return np.nan

        

def vfeats(df):

    # We use this to cut the area behind the rusher and so we can select his area

    # Since he's usually the guy on the far left:

    rusher = df[df.IsRusher] #it's already a .copy()

    rusher.IsRusher = False

    rusher.X -= 1; df = df.append(rusher.copy(), sort=False)

    rusher.X += 99; df = df.append(rusher.copy(), sort=False)

    rusher.X -= 99

    rusher.Y -= 15; df = df.append(rusher.copy(), sort=False)

    rusher.Y += 30; df = df.append(rusher.copy(), sort=False)



    results = df.groupby('PlayId').apply(calcvs).reset_index()

    return results
@numba.jit

def euclidean_distance(x1,y1,x2,y2):

    x_diff = x1-x2

    y_diff = y1-y2

    return math.sqrt(x_diff*x_diff + y_diff*y_diff)

    

@numba.jit

def quad(a,b,c):

    # Return solutions for quadratic

    sol = None

    if abs(a) < 1e-6:

        if abs(b) < 1e-6:

            sol = (0,0) if abs(c) < 1e-6 else None

        else:

            sol = (-c/b, -c/b)

    else:

        disc = b*b - 4*a*c

        if disc >= 0:

            disc = np.sqrt(disc)

            a = 2*a

            sol = ((-b-disc)/a, (-b+disc)/a)

    return sol



@numba.jit

def intercept(def_x, def_y, def_v, run_x, run_y, run_vx, run_vy, direct=False, get_intercept=False):

    if direct:

        # Don't extrapolate

        dir_x = run_x - def_x

        dir_y = run_y - def_y

        

        # Normalize

        dir_len = np.sqrt(dir_x*dir_x + dir_y*dir_y)

        return dir_x/dir_len, dir_y/dir_len



    # Courtesy https://stackoverflow.com/questions/2248876/2d-game-fire-at-a-moving-target-by-predicting-intersection-of-projectile-and-u

    # Calculate the x,y direction the defender needs to be moving at to intercept the player

    # If no real solution is possible, then just return x,y values

    # that move us directly to runners current pos

    tx = run_x - def_x

    ty = run_y - def_y

    tvx = run_vx

    tvy = run_vy

    

    #Get quadratic equation components

    a = tvx*tvx + tvy*tvy - def_v*def_v

    b = 2 * (tvx * tx + tvy * ty);

    c = tx*tx + ty*ty



    ts = quad(a, b, c)



    # Find smallest positive solution

    sol = None

    if ts is not None:

        t0, t1 = ts

        t = min(t0, t1)

        if t < 0:

            t = max(t0, t1)

        if t > 0:

            sol = {

                'x': run_x + run_vx*t,

                'y': run_y + run_vy*t,

            }



    if get_intercept:

        if sol is None:

            return MAX_CLIP

        return sol['x']

            

    # Sol is the intercept point...

    # Now, calculate desired x,y accel directions accordingly....

    if sol is not None:

        dir_x = sol['x'] - def_x

        dir_y = sol['y'] - def_y

    else:

        dir_x = run_x - def_x

        dir_y = run_y - def_y

        

    # Normalize

    dir_len = np.sqrt(dir_x*dir_x + dir_y*dir_y)

    return dir_x/dir_len, dir_y/dir_len
@numba.jit

def euclidean_flat(p1,p2):

    x_diff = p2[:,0]-p1[:,0]

    y_diff = p2[:,1]-p1[:,1]

    return np.sqrt(x_diff*x_diff + y_diff*y_diff)



def PlayerInfluence(grp):

    ball_coords = grp.ball_coords.values[0]

    offp_coords = grp.offp_coords.values[0]

    defp_coords = grp.defp_coords.values[0]

    

    MAXV = 16

    MAXV_log1p = np.log1p(MAXV)

    

    results = []

    for x in range(int(ball_coords[0]+1), int(ball_coords[0]+1+MAXV)):

        point = ball_coords.copy()

        point[0] = x

        point = np.repeat(point.reshape(1,-1), defp_coords.shape[0], axis=0)

        

        # Calculate distance from each offender/defender to the points on front of the ball/runner (+x)

        offp_point_dist = euclidean_flat(offp_coords, point).clip(0,MAXV)

        defp_point_dist = euclidean_flat(defp_coords, point).clip(0,MAXV)

        

        # Compute influence

        offp_point_inf = MAXV_log1p - np.log1p(offp_point_dist)

        defp_point_inf = MAXV_log1p - np.log1p(defp_point_dist)



        results.append(

            offp_point_inf.sum() - defp_point_inf.sum()

        )

        

    return results



@numba.jit

def PointInTriangle(p, p0, p1, p2):

    s = p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]

    t = p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]



    if (s < 0) != (t < 0): return False



    A = -p1[1] * p2[0] + p0[1] * (p2[0] - p1[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]



    return s <= 0 and s + t >= A if A < 0 else s >= 0 and s + t <= A



@numba.jit

def PointInTriangleFlat(p, p0x, p0y, p1x, p1y, p2x, p2y):

    return PointInTriangle(p, [p0x,p0y], [p1x,p1y], [p2x,p2y])



# @numba.jit

def build_oline_features(grp):

    IDX_X = 0

    IDX_Y = 1

    IDX_ROLE = 2

    IDX_OFFENSE = 3



    grp = grp[['X','Y','role','IsOnOffense']].values

    oliners = grp[grp[:,IDX_ROLE]==1]

    defense = grp[grp[:,IDX_OFFENSE]==0, :2] # only interested in X,Y

    TE = grp[grp[:,IDX_ROLE]==2]

    BC = grp[grp[:,IDX_ROLE]==3][0]

    BCx = BC[IDX_X]



    oA = oliners[0]

    oB = oliners[-1]



    if TE.shape[0] == 0:

        # Extend both sides:

        temp = oB.copy(); temp[IDX_Y] += 2; grp = np.concatenate([grp,temp.reshape(1,-1)], axis=0)

        temp = oA.copy(); temp[IDX_Y] -= 2; grp = np.concatenate([temp.reshape(1,-1), grp], axis=0)



    else:

        TE = TE[0]

        d = euclidean_distance(TE[0], TE[1], oA[0], oA[1])

        if d<4:

            # TE is on the low side, extend the high side

            TE[IDX_ROLE] = 1

            temp = oB.copy(); temp[IDX_Y] += 2;

            grp = np.concatenate([TE.reshape(1,-1),grp,temp.reshape(1,-1)], axis=0)



        else:

            d = euclidean_distance(TE[0], TE[1], oB[0], oB[1])



            if d<4:

                # TE is on the hi side, extend the low side

                TE[IDX_ROLE] = 1

                temp = oA.copy(); temp[IDX_Y] -= 2;

                grp = np.concatenate([temp.reshape(1,-1),grp,TE.reshape(1,-1)], axis=0)



    # Update:

    oliners = grp[grp[:,IDX_ROLE]==1]



    oline_length = np.ptp(oliners[:, IDX_Y])

    oline_bline_area, oline_bline_num_def, oline_num_def_inbox = 0,0,0

    oline_bline_num_def_engaged = 0

    oline_num_def_safety = (

        (defense[:, IDX_X] >= BCx+10) &

        (defense[:, IDX_X] <= BCx+20) &

        (defense[:, IDX_Y] >= oliners[0, IDX_Y]) &

        (defense[:, IDX_Y] <= oliners[oliners.shape[0]-1, IDX_Y])

    ).sum()



    # show_play_std(play_id=pid, train=pre, displayit=False)



    oline_avg_ol_dist = []

    for idx, olA in enumerate(oliners):

        if idx == oliners.shape[0] - 2: break

        olB = oliners[idx+1]

        oline_avg_ol_dist.append(euclidean_distance(olA[0], olA[1], olB[0], olB[1]))

    oline_avg_ol_dist = np.array(oline_avg_ol_dist).mean()



    # These are sorted ascending

    for idx in range(oliners.shape[0]-1):

        man1 = oliners[idx]

        man2 = oliners[idx+1]



        man1x, man1y = man1[IDX_X], man1[IDX_Y]

        man2x, man2y = man2[IDX_X], man2[IDX_Y]



        # Area has two components; the square piece, and the triangle piece

        if man2x<man1x:

            MIN_MAN_X = man2x

            MAX_MAN_X = man1x

            min_man = man2

            max_man = man1

        else:

            MIN_MAN_X = man1x

            MAX_MAN_X = man2x

            min_man = man1

            max_man = man2



        h = man2y - man1y

        oline_bline_area += h * max(0, MIN_MAN_X - BCx) # square piece

        oline_bline_area += h * abs(man2x - man1x) / 2  # triangle piece



        square_piece = defense[

            (defense[:, IDX_X] >= BCx) &

            (defense[:, IDX_X] <= MIN_MAN_X) &

            (defense[:, IDX_Y] >= man1y) &

            (defense[:, IDX_Y] <= man2y)

        ]

        oline_bline_num_def += square_piece.shape[0] # Square piece



        for defplayer in defense:

            # Triangle piece

            oline_bline_num_def += int(PointInTriangleFlat(

                BC,

                MIN_MAN_X, min_man[IDX_Y],

                MIN_MAN_X, max_man[IDX_Y],

                MAX_MAN_X, max_man[IDX_Y]

            ))



        oline_num_def_inbox += (

            # Only Square piece

            (defense[:, IDX_X] >= BCx) &

            (defense[:, IDX_X] <= BCx + 10) &

            (defense[:, IDX_Y] >= man1y) &

            (defense[:, IDX_Y] <= man2y)

        ).sum()



        # For engaged part, we only check the square portion

        # For the triangle portion, we hope dist<4 to the closer guy

        for defplayer in square_piece:

            d = euclidean_distance(

                defplayer[0], defplayer[1],

                man1x, man1y

            )

            if d < 1.5:

                oline_bline_num_def_engaged += 1

                continue



            d = euclidean_distance(

                defplayer[0], defplayer[1],

                man2x, man2y

            )

            if d < 1.5:

                oline_bline_num_def_engaged += 1



    return [

        oline_length,

        oline_bline_area,

        oline_bline_num_def,

        oline_num_def_inbox,

        oline_bline_num_def_engaged,

        oline_num_def_safety,

        oline_avg_ol_dist

    ]
def fast_preprocess(df, means={}, isTrain=False, no_fe=False):

    global MIN_CLIP,MAX_CLIP

    seconds_in_year = 60*60*24*365.25



    t = time()

    

    dirs = {'ARI':'ARZ', 'BAL':'BLT', 'CLE':'CLV', 'HOU':'HST'}

    for bad,good in dirs.items():

        df.loc[df.VisitorTeamAbbr==bad, 'VisitorTeamAbbr'] = good

        df.loc[df.HomeTeamAbbr==bad, 'HomeTeamAbbr'] = good

        

    df['IsToLeft'] = df.PlayDirection == 'left'

    df['IsRusher'] = df.NflId == df.NflIdRusher

    

    df['TeamOnOffense'] = "home"

    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = 'away'

    df['IsOnOffense'] = df.Team == df.TeamOnOffense

    del df['TeamOnOffense']

    

    # Used in some downstream calcs

    df.FieldPosition.replace({np.nan:''}, inplace=True)

    mask = df.FieldPosition == df.PossessionTeam

    df['YardLine_std'] = 110 - df.YardLine

    df.loc[mask, 'YardLine_std'] = 10 + df.loc[mask,'YardLine']

    df.YardLine = df.YardLine_std - 10

    del df['YardLine_std']

    

    df['X_std'] = df.X

    df['Y_std'] = df.Y

    df.loc[df.IsToLeft, 'X_std'] = 120 - df.loc[df.IsToLeft, 'X']

    df.loc[df.IsToLeft, 'Y_std'] = 160/3 - df.loc[df.IsToLeft, 'Y'] 

    df.X = df.X_std - 10

    df.Y = df.Y_std

    del df['X_std'], df['Y_std']



    # ROTATE THIS 90 to the right

    df.loc[df.IsToLeft, 'Dir'] = np.mod(180 + df[df.IsToLeft].Dir, 360)

    df.Dir = np.mod(df.Dir/180*math.pi + 3*math.pi/2, math.pi*2)



    # Either do this or keep season as a feature:

    # https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/113277#latest-662092

    df.loc[df.Season==2019,'Season'] = 2018

    mask = df.Season == 2017

    df.loc[mask, 'Orientation'] = np.mod(math.pi/2 + df.loc[mask].Orientation, math.pi*2)

    

    # Correct 2017 Distributions for A and S separately...

    rusher2017_A = [2.54, 0.8112192479993628, 1.0391018448524394, 2.7]

    rusher2017_S = [3.84, 1.074601007663326, 1.1184611205650106, 4.54]

    no_rusher2017_A = [1.43, 0.8586214845778048, 1.034906128189814, 1.56]

    no_rusher2017_S = [2.24, 1.258396412289438, 1.412108400649438, 2.54]



    df.loc[mask & df.IsRusher, 'S'] = (df.S[mask & df.IsRusher] - rusher2017_S[0]) / rusher2017_S[1] * rusher2017_S[2] + rusher2017_S[3]

    df.loc[mask & df.IsRusher, 'A'] = (df.A[mask & df.IsRusher] - rusher2017_A[0]) / rusher2017_A[1] * rusher2017_A[2] + rusher2017_A[3]    

    df.loc[mask & ~df.IsRusher, 'S'] = (df.S[mask & ~df.IsRusher] - no_rusher2017_S[0]) / no_rusher2017_S[1] * no_rusher2017_S[2] + no_rusher2017_S[3]

    df.loc[mask & ~df.IsRusher, 'A'] = (df.A[mask & ~df.IsRusher] - no_rusher2017_A[0]) / no_rusher2017_A[1] * no_rusher2017_A[2] + no_rusher2017_A[3]

    

    if no_fe:

        return df

    

    # Rusher Features:

    features = [

        'GameId','PlayId','NflIdRusher','X','Y','Dir',

        'YardLine','Season','S','A','Dis',

    ]

    if isTrain: features += ['Yards']

    rushers = df[df.IsRusher][features].copy()

    

    ##########################################################################################

    # Voroni features of Rusher VS Offense

    vf = vfeats(df[df.IsRusher | ~df.IsOnOffense][['PlayId','IsRusher','X','Y']].copy())

    vf.columns = ['PlayId','vArea']

    rushers = rushers.merge(vf, how='left',on='PlayId')

    if isTrain: print(time()-t, 'Done Veroni'); t=time()

 

    ##########################################################################################

    

    # Influence (Distance) Features:

    df['XY'] = df[['X','Y']].apply(lambda x: x.tolist(), axis=1)

    

    offp_coords = df[df.IsOnOffense==True][['PlayId','XY']].groupby('PlayId').agg(list).reset_index()

    defp_coords = df[df.IsOnOffense==False][['PlayId','XY']].groupby('PlayId').agg(list).reset_index()

    ball_coords = df[df.IsRusher==True][['PlayId','XY','A']].groupby(['PlayId','A']).agg(list).reset_index()



    offp_coords.columns = ['PlayId', 'offp_coords']

    defp_coords.columns = ['PlayId', 'defp_coords']

    ball_coords.columns = ['PlayId', 'A', 'ball_coords']



    ball_coords.ball_coords = ball_coords.ball_coords.apply(lambda x: np.array(x[0]))

    defp_coords.defp_coords = defp_coords.defp_coords.apply(np.array)

    offp_coords.offp_coords = offp_coords.offp_coords.apply(np.array)



    ball_coords = ball_coords.merge(defp_coords, how='left', on='PlayId')

    ball_coords = ball_coords.merge(offp_coords, how='left', on='PlayId')



    InfluenceRusherX = ball_coords.groupby('PlayId').apply(PlayerInfluence).reset_index()

    InfluenceRusherX.columns = ['PlayId', 'InfluenceRusherX']

    InfluenceRusherX['InfluenceRusherX_flip'] = InfluenceRusherX.InfluenceRusherX.apply(lambda x: np.argmax(-np.sign(x)))

    rushers = rushers.merge(InfluenceRusherX[['PlayId','InfluenceRusherX_flip']], how='left', on='PlayId')



    def triangleCongestion(grp):

        p0 = grp.ball_coords.values[0]



        A = 10 - 5 * (min(8, grp.A.values[0]) / 8)

        offp_coords = grp.offp_coords.values[0]

        defp_coords = grp.defp_coords.values[0]



        p1, p2 = p0.copy(), p0.copy()

        p1[0] += 20; p1[1] -= A

        p2[0] += 20; p2[1] += A

        

        def_conjestion = 0

            

        for p in defp_coords:

            if not PointInTriangle(p, p0, p1, p2): continue

            def_conjestion += 1

            

        return def_conjestion

        

    triCon = ball_coords.groupby('PlayId').apply(triangleCongestion).reset_index()

    triCon.columns = ['PlayId', 'def_triCon']

    rushers = rushers.merge(triCon[['PlayId','def_triCon']], how='left', on='PlayId')

    

    del offp_coords, defp_coords, ball_coords, df['XY']

    if isTrain: print(time()-t, 'Done Influence'); t=time()

    #########################################################################################



    

    # TODO: Consider this one...

    rushers['rusher_dist_td'] = 99 - rushers.YardLine # We use YardLine so we can update yards...

    rushers['rusher_dist_scrimmage'] = rushers.YardLine - rushers.X

    rushers['rusher_moving_back'] = rushers.Dir.between(np.pi/2, 3*np.pi/2).astype(np.int)

    rushers = rushers.rename(columns={

        'X':'rusher_X',

        'Y':'rusher_Y',

    })

    features = [        

        'GameId','PlayId','NflIdRusher','rusher_X','rusher_Y','Dir',

        'S','A','Dis','YardLine','Season',

        'rusher_dist_scrimmage','rusher_moving_back','rusher_dist_td',

        'vArea','InfluenceRusherX_flip','def_triCon',

    ]

    

    if isTrain:

        features += ['Yards','TYards','Orig_Yards_NOTRAIN',]

        rushers['TYards'] = (

            # How far I'm going to move from my current position:

            rushers.Yards + rushers.rusher_dist_scrimmage.astype(int)

        ).clip(MIN_CLIP, MAX_CLIP)

        rushers['Orig_Yards_NOTRAIN'] = rushers.Yards.copy()

        rushers.Yards = rushers.Yards.clip(MIN_CLIP,MAX_CLIP)

        

    rushers = rushers[features]

    

    ##########################################################################################

    # OLine Features: https://www.kaggle.com/sherkt1/final-hole-metric-features-v5-0

    '''

        oline_length, y length of the offensive line

        oline_bline_area, area of the line under the curve

        oline_bline_num_def, number of defensive players who've penetrated the curve

        oline_num_def_inbox, number of defensive players within the box (rusher.x+10)

        oline_bline_num_def_engaged, number of defensive players who've penetrated the curve and are within 1.5 yards of OLman

        oline_num_def_safety, number of defensers in backfield rusher.x+[10,20]

        oline_avg_ol_dist, mean distance between consecutive OLmen

    '''

    if isTrain: print(time()-t, 'Preparing OLine'); t=time()

    players = df[[

        # select oline features here

        'PlayId','IsOnOffense','X','Y','Position','IsRusher'

    ]].copy()

    players['role'] = players.Position.isin('T,G,C,OT,OG'.split(',')).astype(np.uint8)

    players.loc[players.Position=='TE','role'] = 2

    players.loc[players.IsRusher,'role'] = 3

    del players['Position'], players['IsRusher'] # role = 1 for OLine, 2 for TE, 0 for other

    players.IsOnOffense = players.IsOnOffense.astype(np.uint8)

    players.sort_values(['IsOnOffense','Y'], inplace=True)

    oline_feats = players.groupby('PlayId').apply(build_oline_features).reset_index()

    del players

    oline_feats.columns = ['PlayId', 'olinef']



    # Merge relevant features

    oline_feats['oline_length'] = oline_feats.olinef.apply(lambda x: x[0])

    oline_feats['oline_bline_area'] = oline_feats.olinef.apply(lambda x: x[1])

    oline_feats['oline_bline_num_def'] = oline_feats.olinef.apply(lambda x: x[2])

    oline_feats['oline_num_def_inbox'] = oline_feats.olinef.apply(lambda x: x[3])

    oline_feats['oline_bline_num_def_engaged'] = oline_feats.olinef.apply(lambda x: x[4])

    oline_feats['oline_num_def_safety'] = oline_feats.olinef.apply(lambda x: x[5])

    oline_feats['oline_avg_ol_dist'] = oline_feats.olinef.apply(lambda x: x[6])

    rushers = rushers.merge(

        oline_feats[[

            'PlayId', 'oline_length',

            'oline_bline_area', 'oline_bline_num_def', 'oline_num_def_inbox',

            'oline_bline_num_def_engaged', 'oline_num_def_safety', 'oline_avg_ol_dist',

        ]],

        how='left', on='PlayId'

    )

    if isTrain: print(time()-t, 'Done OLine'); t=time()

    ##########################################################################################

    

    # Max Observed rusher speed, truncated at median

    if 'RusherMaxObservedS' not in means:

        medianS = rushers.S.median()

        RusherMaxObservedS = rushers.groupby('NflIdRusher').S.median().reset_index()

        RusherMaxObservedS = {NflIdRusher:S for NflIdRusher, S in RusherMaxObservedS.values}

        means['RusherMaxObservedS'] = RusherMaxObservedS

        means['RusherMaxObservedS_Median'] = medianS

    rushers['RusherMaxObservedS'] = rushers.NflIdRusher.map(means['RusherMaxObservedS']).fillna(means['RusherMaxObservedS_Median'])

    rushers.loc[rushers.RusherMaxObservedS<means['RusherMaxObservedS_Median'], 'RusherMaxObservedS'] = means['RusherMaxObservedS_Median']

    

    

    rushers['dist_next_point_time'] = np.square(rushers.S) + 2 * rushers.A * rushers.Dis

    rushers['rusher_SX'] = rushers.S * np.cos(rushers.Dir)

    rushers['rusher_SY'] = rushers.S * np.sin(rushers.Dir)

    rushers.rename(columns={'S':'rusher_S','A':'rusher_A'}, inplace=True)

    if isTrain: print(time()-t, 'Advanced Rusher'); t=time()

    

    

    # Player distances relative to the RB's

    player_dist = df[~df.IsRusher][['PlayId','NflId','X','Y','S','A','IsOnOffense','PlayerWeight','Dir']]

    player_dist['SX'] = player_dist.S * np.cos(player_dist.Dir)

    player_dist['AX'] = player_dist.A * np.cos(player_dist.Dir)

    player_dist['WX'] = player_dist.PlayerWeight * np.cos(player_dist.Dir)

    player_dist = player_dist.merge(rushers, on='PlayId', how='inner')

    player_dist['dist_to_rusher'] = player_dist[['X','Y','rusher_X','rusher_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    

    defensivePlayers = player_dist[player_dist.IsOnOffense==False]

    offensivePlayers = player_dist[player_dist.IsOnOffense==True]

    

    # For the rest of the analysis, we dont count the QB or any other offensive player behind the RB

    offensivePlayers = offensivePlayers[offensivePlayers.X>offensivePlayers.rusher_X]

    

    SX1 = offensivePlayers.groupby('PlayId').SX.mean().sort_index().reset_index(); SX1.columns=['PlayId','SX1']

    SX2 = defensivePlayers.groupby('PlayId').SX.mean().sort_index().reset_index(); SX2.columns=['PlayId','SX2']

    SX1 = SX1.merge(SX2, how='left', on='PlayId'); SX1.SX1 += SX1.SX2; del SX1['SX2']; SX1.columns=['PlayId','Mean_SX']

    rushers = rushers.merge(SX1, how='left', on='PlayId')

    

    AX1 = offensivePlayers.groupby('PlayId').AX.mean().sort_index().reset_index(); AX1.columns=['PlayId','AX1']

    AX2 = defensivePlayers.groupby('PlayId').AX.mean().sort_index().reset_index(); AX2.columns=['PlayId','AX2']

    AX1 = AX1.merge(AX2, how='left', on='PlayId'); AX1.AX1 += AX1.AX2; del AX1['AX2']; AX1.columns=['PlayId','Mean_AX']

    rushers = rushers.merge(AX1, how='left', on='PlayId')



    WX1 = offensivePlayers.groupby('PlayId').WX.mean().sort_index().reset_index(); WX1.columns=['PlayId','WX1']

    WX2 = defensivePlayers.groupby('PlayId').WX.mean().sort_index().reset_index(); WX2.columns=['PlayId','WX2']

    WX1 = WX1.merge(WX2, how='left', on='PlayId'); WX1.WX1 += WX1.WX2; del WX1['WX2']; WX1.columns=['PlayId','Mean_WX']

    rushers = rushers.merge(WX1, how='left', on='PlayId')

    

    del AX1, AX2, SX1, SX2, WX1, WX2

    ###############################################################

    

    # Count how many defenders are on the side of the field the rusher is moving towards

    # relative to the rusher:

    defense = defensivePlayers.groupby('PlayId').Y.agg(list).reset_index()

    defense.columns = ['PlayId', 'defense_Y']



    rushers = rushers.merge(defense, on='PlayId', how='left')

    rushers.defense_Y = rushers[['defense_Y','rusher_Y','rusher_SY']].apply(lambda row: sum([1 for dy in row.defense_Y if np.sign(row.rusher_SY)==np.sign(dy - row.rusher_Y)]), axis=1)

    

    if isTrain: print(time()-t, 'Done OffDef'); t=time()

    ###############################################################

    

    if isTrain: print(time()-t, 'Done IX'); t=time()



    closest_def = defensivePlayers[['PlayId','dist_to_rusher','S','A','X']].sort_values(['PlayId','dist_to_rusher'])



    def cdefagg_S(row):

        return row.sort_values('dist_to_rusher').S.values[:3].mean()

    def cdefagg_A(row):

        return row.sort_values('dist_to_rusher').A.values[:3].mean()

    def cdefagg_X(row):

        return row.sort_values('dist_to_rusher').X.values[:3].mean()

    

    cdef = defensivePlayers[['PlayId','dist_to_rusher','S']].groupby('PlayId').apply(cdefagg_S).reset_index()

    cdef.columns = ['PlayId', 'closest_def_S']

    rushers = rushers.merge(cdef, how='left', on='PlayId')

    

    cdef = defensivePlayers[['PlayId','dist_to_rusher','A']].groupby('PlayId').apply(cdefagg_A).reset_index()

    cdef.columns = ['PlayId', 'closest_def_A']

    rushers = rushers.merge(cdef, how='left', on='PlayId')

    

    # Stats on the location and spread of the defense along the scrimmage line

    grp = defensivePlayers.groupby('PlayId').Y.agg(['mean','std']).reset_index()

    grp.columns = ['PlayId','defense_scrimmage_Y_mean','defense_scrimmage_Y_std']

    rushers = rushers.merge(grp, how='left', on='PlayId')

    rushers['rusher_defense_scrimmage_Y_dist'] = rushers.rusher_Y - rushers.defense_scrimmage_Y_mean

    if isTrain: print(time()-t, 'Done closest def'); t=time()





    def offense_mid6(x):

        # For offensive players, we don't care about the QB and players very far from us

        # So we sort by distance, skip the closest guy (QB) then take the next 5 closest team mates

        # We already filter to include only ppl with X>rusherX

        subset = np.sort(x)[:8]

        return [subset.min(), subset.mean(), subset.max(), subset.std()]



    

    grp = player_dist[player_dist.IsOnOffense==True].groupby('PlayId')

    offensive_dist = grp.dist_to_rusher.agg(offense_mid6).reset_index()

    offensive_dist['min_off5_dist_to_rusher'] = offensive_dist.dist_to_rusher.apply(lambda x: x[0])

    offensive_dist['avg_off5_dist_to_rusher'] = offensive_dist.dist_to_rusher.apply(lambda x: x[1])

    offensive_dist['max_off5_dist_to_rusher'] = offensive_dist.dist_to_rusher.apply(lambda x: x[2])

    offensive_dist['std_off5_dist_to_rusher'] = offensive_dist.dist_to_rusher.apply(lambda x: x[3])

    del offensive_dist['dist_to_rusher']

    rushers = rushers.merge(offensive_dist, on='PlayId', how='inner')

    del offensive_dist



    

    def defense_closest9(x):

        # For defensive players, we don't care about the Defensive Backs.

        # Just the ppl closest to the Rusher

        subset = np.sort(x)[:9]

        subset = x

        return [subset.min(), subset.mean(), subset.max(), subset.std()]

    

    grp = player_dist[player_dist.IsOnOffense==False].groupby('PlayId')

    defensive_dist = grp.dist_to_rusher.agg(defense_closest9).reset_index()

    defensive_dist['min_def9_dist_to_rusher'] = defensive_dist.dist_to_rusher.apply(lambda x: x[0])

    defensive_dist['avg_def9_dist_to_rusher'] = defensive_dist.dist_to_rusher.apply(lambda x: x[1])

    defensive_dist['max_def9_dist_to_rusher'] = defensive_dist.dist_to_rusher.apply(lambda x: x[2])

    defensive_dist['std_def9_dist_to_rusher'] = defensive_dist.dist_to_rusher.apply(lambda x: x[3])

    

    def odist(x):

        return np.sort(x).tolist()

    temp = grp.dist_to_rusher.agg(odist).reset_index()

    for i in range(9):

        defensive_dist[f'def9_dist_to_rusher_{i}'] = temp.dist_to_rusher.apply(lambda x: x[i])

    

    del defensive_dist['dist_to_rusher']

    rushers = rushers.merge(defensive_dist, on='PlayId', how='inner')

    del defensive_dist    

    

    if isTrain: print(time()-t, 'dist_to_rusher'); t=time()

    

    ######    ######    ######    ######    ######    ######    ######    ######    ######    ######    ######

    #TODO

    # - fastest accelerating defenders acceleration

    # 	- dist to this person

    def defense_fA(row):

        # For defensive players, we don't care about the Defensive Backs.

        # Just the ppl closest to the Rusher

        subset = row.sort_values('A', ascending=False).iloc[0]

        return [subset.A, subset.dist_to_rusher]

    defensive_fA = grp[['dist_to_rusher','A']].apply(defense_fA).reset_index()

    defensive_fA.rename(columns={0:'temp'}, inplace=True)

    defensive_fA['defensive_fA_maxA'] = defensive_fA.temp.apply(lambda x: x[0])

    defensive_fA['defensive_fA_dist'] = defensive_fA.temp.apply(lambda x: x[1])

    del defensive_fA['temp']

    rushers = rushers.merge(defensive_fA, on='PlayId', how='inner')

    del defensive_fA

    if isTrain: print(time()-t, 'Done A'); t=time()

    

    # - fastest speeding defenders speed

    # - dist to this person

    def defense_fS(row):

        # For defensive players, we don't care about the Defensive Backs.

        # Just the ppl closest to the Rusher

        subset = row.sort_values('S', ascending=False).iloc[0]

        return [subset.S, subset.dist_to_rusher]

    defensive_fS = grp[['dist_to_rusher','S']].apply(defense_fS).reset_index()

    defensive_fS.rename(columns={0:'temp'}, inplace=True)

    defensive_fS['defensive_fS_maxS'] = defensive_fS.temp.apply(lambda x: x[0])

    defensive_fS['defensive_fS_dist'] = defensive_fS.temp.apply(lambda x: x[1])

    del defensive_fS['temp']

    rushers = rushers.merge(defensive_fS, on='PlayId', how='inner')

    del defensive_fS

    if isTrain: print(time()-t, 'Done B'); t=time()

    

    

    # - closest 9 defenders stats on A, and stats on S

    def defense_closest9_A(row):

        # For defensive players, we don't care about the Defensive Backs.

        # Just the ppl closest to the Rusher

        subset = row.sort_values('dist_to_rusher').A.iloc[:9]

        return [subset.min(), subset.mean(), subset.max(), subset.std()]

    defensive_A = grp[['dist_to_rusher','A']].apply(defense_closest9_A).reset_index()

    defensive_A.rename(columns={0:'temp'}, inplace=True)

    defensive_A['min_def9_A'] = defensive_A.temp.apply(lambda x: x[0])

    defensive_A['avg_def9_A'] = defensive_A.temp.apply(lambda x: x[1])

    defensive_A['max_def9_A'] = defensive_A.temp.apply(lambda x: x[2])

    defensive_A['std_def9_A'] = defensive_A.temp.apply(lambda x: x[3])

    del defensive_A['temp']

    rushers = rushers.merge(defensive_A, on='PlayId', how='inner')

    del defensive_A

    if isTrain: print(time()-t, 'Done C'); t=time()

    

    def defense_closest9_S(row):

        # For defensive players, we don't care about the Defensive Backs.

        # Just the ppl closest to the Rusher

        subset = row.sort_values('dist_to_rusher').S.iloc[:9]

        return [subset.min(), subset.mean(), subset.max(), subset.std()]

    defensive_S = grp[['dist_to_rusher','S']].apply(defense_closest9_S).reset_index()

    defensive_S.rename(columns={0:'temp'}, inplace=True)

    defensive_S['min_def9_S'] = defensive_S.temp.apply(lambda x: x[0])

    defensive_S['avg_def9_S'] = defensive_S.temp.apply(lambda x: x[1])

    defensive_S['max_def9_S'] = defensive_S.temp.apply(lambda x: x[2])

    defensive_S['std_def9_S'] = defensive_S.temp.apply(lambda x: x[3])

    del defensive_S['temp']

    rushers = rushers.merge(defensive_S, on='PlayId', how='inner')

    del defensive_S

    if isTrain: print(time()-t, 'Done D'); t=time()

    

    # Log transforms

    for col in 'dist_next_point_time,rusher_dist_from_center_Y,min_off5_dist_to_rusher,max_def9_dist_to_rusher,std_def9_dist_to_rusher,defense_centroid_to_rusher_dist,offense_centroid_to_rusher_dist,min_def9_S,avg_def9_S,closest_def_A,closest_def_S,closest_def_S,vArea,std_def9_S,ix_std_dist,RusherMaxObservedS'.split(','):

        if col in rushers.columns:

            rushers[col] = np.log1p(rushers[col])



            

    ###################

    def inrange(grp):

        return grp.Y.between(grp.rusher_Y-5, grp.rusher_Y+5).sum()

    

    temp = player_dist.groupby('PlayId').apply(inrange).reset_index()

    temp.columns = ['PlayId', 'Congestion']

    rushers = rushers.merge(temp, how='left', on='PlayId')

    

    

    # TODO: 

    # Log transforms on ix_mean_dist,ix_median_dist,ix_max_dist

    return rushers, means
# # IMPORTNAT EDA TO CORRECT A/S VALUES FOR 2017:

# col='A' # do for A,S, and do for IsRusher and !isRusher...



# rushers = train[train.NflId!=train.NflIdRusher].copy()

# std2017 = rushers.loc[rushers.Season==2017,col].std()

# std2018 = rushers.loc[rushers.Season==2018,col].std()

# mean2017 = rushers.loc[rushers.Season==2017,col].mean()

# mean2018 = rushers.loc[rushers.Season==2018,col].mean()

# median2017 = rushers.loc[rushers.Season==2017,col].median()

# median2018 = rushers.loc[rushers.Season==2018,col].median()



# std20189 = rushers.loc[rushers.Season==2018,col].append(test[test.NflId!=test.NflIdRusher][col]).std()

# median20189 = rushers.loc[rushers.Season==2018,col].append(test[test.NflId!=test.NflIdRusher][col]).median()

# mean20189 = rushers.loc[rushers.Season==2018,col].append(test[test.NflId!=test.NflIdRusher][col]).mean()



# rushers.loc[rushers.Season==2017,col] = (rushers.loc[rushers.Season==2017,col] - median2017) / std2017 * std20189 + median20189





# plt.figure(figsize=(8,5))

# plt.title(col)



# sns.distplot(rushers[(rushers.Season==2017)][col], label="2017")

# sns.distplot(rushers[(rushers.Season==2018)][col], label="2018")

# sns.distplot(test[test.NflId!=test.NflIdRusher][col], label="2019")

# plt.legend(prop={'size': 12})

# plt.show()



# print(f'{col}, {[median2017, std2017, std20189, median20189]}')
gc.collect()

what, means = fast_preprocess(train.copy(), isTrain=True)

what.columns
z,y = what.isna().sum(), what.isna().sum()

z[z>0], y[y>0]
reg_cols = [

    # Potential Overfitting:

    'Season',

    

    'rusher_A', 'rusher_SX', 'avg_def9_dist_to_rusher',

    'rusher_dist_scrimmage', 'rusher_S',

    'avg_off5_dist_to_rusher', 'min_off5_dist_to_rusher',

    'dist_next_point_time',

    'min_def9_dist_to_rusher', 'std_off5_dist_to_rusher',

    'defense_scrimmage_Y_std', 'std_def9_dist_to_rusher', 'Dis',

    'closest_def_S', 'closest_def_A',

    'defense_Y', 'InfluenceRusherX_flip',

    'vArea', 'avg_def9_A', 'avg_def9_S', 'std_def9_S', 'min_def9_S',

    # These are good, but we don't wanna insert 1000 of them:

    # 'def9_dist_to_rusher_0',

    # 'def9_dist_to_rusher_1', 'def9_dist_to_rusher_2',

    # 'def9_dist_to_rusher_3', 'def9_dist_to_rusher_4',

    # 'def9_dist_to_rusher_5', 'def9_dist_to_rusher_6',

    # 'def9_dist_to_rusher_7', 'def9_dist_to_rusher_8',

    

    'RusherMaxObservedS',

    'Mean_SX','Mean_AX','Mean_WX',

    'Congestion', 'def_triCon',

    'oline_length', 'oline_bline_area', 'oline_num_def_inbox', 'oline_avg_ol_dist',

    

    # New Experimental!!!:



]

len(reg_cols)
# # ALWAYS DOUBLE CHECK:

# if RUN_KAGGLE == False:

#     for col in reg_cols:

#         plt.title(col)

#         plt.hist(what[col], 100)

#         plt.show()
gc.collect()

X = what.copy()



# transformed for mse, we should also standardize....

ymu = np.log1p(X.Yards.values - MIN_CLIP) # we zero it for sparse xent



yent = np.zeros((X.Yards.shape[0], N_OUTPUTS))

for idx, target in enumerate(list(X.Yards.astype(int))):

    yent[idx, target-MIN_CLIP] = 1

    

yent_NOCLIP_CSUM = np.zeros((X.Orig_Yards_NOTRAIN.shape[0], 199))

for idx, target in enumerate(list(X.Orig_Yards_NOTRAIN.astype(int))):

    yent_NOCLIP_CSUM[idx, target+99] = 1

yent_NOCLIP_CSUM = np.clip(np.cumsum(yent_NOCLIP_CSUM, axis=1), 0, 1)

    

stdscale = StandardScaler()

stdscale.fit(X[reg_cols])
keys = X.Yards.unique()-MIN_CLIP

values = class_weight.compute_class_weight('balanced', keys, X.Yards.values-MIN_CLIP)

cweights = dict(zip(keys, values))

cweights

# cweights = None
# UNDERSTAND OUR TARGETS



idx = 1

orig_mu = X.Yards.values[idx] - MIN_CLIP

plt.title(f'mu:{ymu[idx]:.2f}, omu:{orig_mu}')

plt.plot(yent[idx])

plt.show()



plt.title('Yards')

plt.hist(X.Yards,100)

plt.show()



print(ymu.min(), ymu.max())

plt.title('ymu')

plt.hist(ymu,100)

plt.show()
from keras import regularizers



def BuildModel(divisor=1):

    global N_OUTPUTS

    inp = Input(shape=(len(reg_cols),))

    x = inp

    x = GaussianNoise(0.0025)(x)

    

    x = Dense(1024//divisor, activation='elu')(x) 

    x = Dropout(0.25)(x)

    x = Dense(256//divisor, activation='elu')(x)

    x = Dense(128//divisor, activation='elu')(x)

    x = Dense(64//divisor, activation='elu')(x)

    x = BatchNormalization()(x)

    

    # mu in raw space [0,N_OUTPUTS)

    # We don't have to add/sub Min_Clip because we're already in raw space starting from 0

    # So we just apply the log1p transform

    mu_raw = Dense(1, activation='relu')(x)

    mu = Lambda(lambda x: K.log(1 + K.reshape(x[:,0],(-1,1))), name='mu')(mu_raw)

    

    ent = Dense(N_OUTPUTS, name='ent', activation='sigmoid')(

        Concatenate()([

            x, mu_raw

        ])

    )

    reg_pass = Lambda(lambda x: x, name='reg_pass')(ent) # passthrough for different loss function...

    

    model = Model(inp, [ent,reg_pass,mu])

    model.compile(

        #optimizer=Nadam(learning_rate=0.0025, beta_1=0.9, beta_2=0.999),

        optimizer=RAdam(weight_decay=0.0003),

        loss={

            'ent': 'categorical_crossentropy', # binary_crossentropy

            'reg_pass': losses.mae,

            'mu': losses.mae,

        },

        loss_weights={

            'ent': 1,

            'reg_pass': 1,

            'mu': 1,

        }

    )

    

    return model
# Downconvert float64 to float32

for col, dt in zip(X.columns,X.dtypes):

    if col in reg_cols and dt=='int64':

        X[col] = X[col].astype(np.float32)

        print('Categorical:', col)

        

    if dt!='float64': continue

    X[col] = X[col].astype(np.float32)
t = time()

K.clear_session()



if RUN_KAGGLE:

    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

else:

    kf = GroupKFold(n_splits=n_splits)

    

oof = np.zeros((X.Yards.shape[0], N_OUTPUTS))

oof_mu = np.zeros((X.Yards.shape[0]))

lgb_oof = np.zeros((X.Yards.shape[0], N_OUTPUTS))

report = []

use_models = []

perms = []



for fold_, (tdx, vdx) in enumerate(kf.split(X, yent, X.GameId)):

    X_train, X_val = X.loc[tdx].copy(), X.loc[vdx].copy()

    X_train_bound, X_val_bound = X_train[bounding_cols].copy(), X_val[bounding_cols].copy()

    

    y_train_ent, y_train_mu = yent[tdx], ymu[tdx]

    y_val_ent, y_val_mu = yent[vdx], ymu[vdx]

    y_train_act = X.Yards[tdx] - MIN_CLIP

    y_val_act = X.Yards[vdx] - MIN_CLIP

    y_val_ent = yent[vdx]

    

    if DO_LGBM:

        # Add in a LGM model...

        trn_data = lgb.Dataset(X_train[lgb_cols], label=y_train_act, categorical_feature=cat_feats, free_raw_data=False)

        val_data = lgb.Dataset(X_val[lgb_cols], label=y_val_act, categorical_feature=cat_feats, free_raw_data=False)

        

        metric = Metric(None, [], []) # placeholder

        yval199  = metric.convert_to_y199_cumsum(y_val_ent)

        

        model_name = f'LGB_{fold_}.lgb'

            

        lgb_model = lgb.train(

            params,

            trn_data,

            num_rounds,

            valid_sets = [trn_data, val_data],

            verbose_eval=500,

            early_stopping_rounds = 50,

        )



        lgb_preds = lgb_model.predict(X_val[lgb_cols], num_iteration=lgb_model.best_iteration)

        lgb_model.save_model(model_name)



        preds = metric.bound_prediction(preds, X_val)

        lgb_oof[vdx] = preds.copy()

        preds = metric.convert_to_y199_cumsum(preds)

        lgb_score_ = crps(preds, yval199)

        

        print(f'(original) LGB RunScore: {lgb_score_}', end='\n\n')

        del trn_data, val_data, yval199; gc.collect()

        

        

    

    # Transform for NNet: (note, for lgb don't do this)

    X_train[reg_cols] = stdscale.transform(X_train[reg_cols])

    X_val[reg_cols]   = stdscale.transform(X_val[reg_cols])

    

    # NNet Setup

    y_train = {'ent':y_train_ent, 'reg_pass':y_train_ent, 'mu':y_train_mu}

    y_val   = {'ent':y_val_ent,   'reg_pass':y_val_ent,   'mu':y_val_mu  }

    y_val_act = X.Yards[vdx] - MIN_CLIP

    

    

    

    

    

    

    

    # For Drop Importance:

    y_val_crps = None

    y_val_crps_orig = yent_NOCLIP_CSUM[vdx]

    

    # I say blend the best 3 of 4 runs...

    oof_mu_runs = np.zeros((n_runs, X_val.shape[0]))

    oof_runs = np.zeros((n_runs, X_val.shape[0], N_OUTPUTS))

    oof_run_scores = []

    for run_ in range(n_runs):

        print(f'Fold: {fold_}, Run: {run_}')



        model = BuildModel(divisor=DIVISOR)

        if fold_==0 and run_==0: model.summary()

        es = EarlyStopping(

            monitor='val_CRPS',

            mode='min',

            restore_best_weights=True, 

            verbose=2, 

            patience=25

        )



        es.set_model(model)

        metric = Metric(model, [es], [(X_train[reg_cols],y_train), (X_val[reg_cols],y_val)])

        metric.set_original_data(X_train_bound, X_val_bound)

        hist = model.fit(X_train[reg_cols], y_train, class_weight=cweights, callbacks=[metric], epochs=400, batch_size=1024//2, verbose=False)

        model.save(f'fold_{fold_}_{run_}.h5')

        score_ = min(hist.history['val_CRPS'])

        oof_run_scores.append(score_)

        

        # Stash OOF:

        preds, _, preds_mu = model.predict(X_val[reg_cols], batch_size=1024)

        preds = metric.bound_prediction(preds, X_val_bound)

        oof_runs[run_] = preds.copy()

        oof_mu_runs[run_] = preds_mu.flatten().copy()

        preds = metric.convert_to_y199_cumsum(preds)

        

        

        if y_val_crps is None:

            y_val_crps = metric.convert_to_y199_cumsum(y_val_ent)

        

        orig_score_ = crps(preds, y_val_crps_orig)

        print(f'RunScore: {score_}, (Original): {orig_score_}')



        if True:#RUN_KAGGLE == False:

            plt.title(str(score_))

            plt.plot(hist.history['tr_CRPS'])

            plt.plot(hist.history['val_CRPS'])

            plt.show()

                

            # Then, we do perm-importance on all folds, not just the top-k

            perm = {}

            for idx, feature in enumerate(tqdm(reg_cols)):

                backup = X_val[feature].values.copy()

                X_val[feature] = 0



                _y_pred_ent = model.predict(X_val[reg_cols], batch_size=1024)[0]

                _y_pred_ent = metric.bound_prediction(_y_pred_ent, X_val_bound)

                _y_pred_ent = metric.convert_to_y199_cumsum(_y_pred_ent)

                perm[feature] = crps(_y_pred_ent, y_val_crps) - score_

                

                X_val[feature] = backup

            perms.append(perm)



    # Top-k, by minimizing score

    worst_run = np.argmax(oof_run_scores)

    bad_model = f'fold_{fold_}_{worst_run}.h5'

    use_models += [f'fold_{fold_}_{run_}.h5' for run_ in range(n_runs) if run_ != worst_run]

    

    # Average the good runs

    oof[vdx] = np.delete(oof_runs, worst_run, axis=0).mean(axis=0)

    oof_mu[vdx] = np.delete(oof_mu_runs, worst_run, axis=0).mean(axis=0)

    

    # Transform to competition space, only for evaluation

    preds = metric.convert_to_y199_cumsum(oof[vdx])

    score_ = crps(preds, y_val_crps)

    orig_score_ = crps(preds, y_val_crps_orig)

    print(oof_run_scores, np.array(oof_run_scores).std(), 'STD')

    report.append(f'Fold: {fold_}, TopK Mean Score: {score_}, (Original): {orig_score_}')

    print(report[-1], end='\n\n')

    print('—'*100)

    

oof = metric.convert_to_y199_cumsum(oof)

oof_yent = metric.convert_to_y199_cumsum(yent)

score_ = crps(oof, oof_yent)

orig_score_ = crps(oof, yent_NOCLIP_CSUM)

report.append(f'Final Model Score: {score_}, (Original): {orig_score_}')



if DO_LGBM:

    lgb_oof = metric.convert_to_y199_cumsum(lgb_oof)

    total_val = metric.convert_to_y199_cumsum(yent)

    lgb_score_ = crps(lgb_oof, total_val)

    print('Final Model Score LGB', lgb_score_)



print(report[-1])

print(time() - t)
# Looking at influence in the direction the user is moving rather than +x

# Also bounding to +15 lookahead instead of |MIN_CLIP|

report
report
dfplt = pd.DataFrame({'y_true':X.Yards.values, 'y_pred':np.argmax(np.diff(oof), axis=1)+MIN_CLIP})

sns.jointplot(x=dfplt.y_true, y=dfplt.y_pred, kind='kde')
# pidf = build_pidf(perms)

# # pidf.to_csv('pidf50__{}.csv'.format(score_), index=False)



# plt.title(f'Drop Importance, OOF CV {np.round(score_,8)}')

# plt.plot(pidf.means.values)

# plt.plot(pidf.medians.values)

# plt.scatter(np.arange(pidf.shape[0]), pidf.means.values, s=10)



# plt.show()



# pidf
# plot_feature_importance(perms)#, cutoff=5)
# for season in [2017,2018]:

#     look = train[(train.NflId==train.NflIdRusher) & (train.Season==season)].reset_index()

#     look.sort_values('PlayId', ascending=False, inplace=True)

#     look = look[look.index<look.shape[0]//3]

#     print(look.shape)

#     look.sort_values('PlayId', ascending=False, inplace=True)

#     diffs = look.groupby('GameId').YardLine.diff().reset_index()

#     diffs = diffs[~diffs.YardLine.isna() & diffs.YardLine.between(MIN_CLIP, MAX_CLIP)]



#     plt.title(f'{season} Mean: {diffs.YardLine.mean()}, STD: {diffs.YardLine.std()}')

#     plt.hist(diffs.YardLine, 100)

#     plt.xlim(0,50)

#     plt.show()

    

# look = test[test.NflId==test.NflIdRusher].copy()

# print(look.shape)

# look.sort_values('PlayId', ascending=False, inplace=True)

# diffs = look.groupby('GameId').YardLine.diff().reset_index()

# diffs = diffs[~diffs.YardLine.isna() & diffs.YardLine.between(MIN_CLIP, MAX_CLIP)]



# plt.title(f'2019 Mean: {diffs.YardLine.mean()}, STD: {diffs.YardLine.std()}')

# plt.hist(diffs.YardLine, 100)

# plt.xlim(0,50)

# plt.show()
models = []

for model_path in use_models:

    models.append(load_model(model_path))
from kaggle.competitions import nflrush

env = nflrush.make_env()

iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    basetable, _ = fast_preprocess(test_df, isTrain=False, means=means.copy())

    basetable_bound = basetable[bounding_cols].copy()



    y_pred = [

        metric.bound_prediction(

            lgb.Booster(model_file=f'LGB_{fold_}.lgb').predict(basetable[lgb_cols]),

            basetable_bound

        )

        for fold_ in range(n_splits)

    ]

    

    basetable[reg_cols] = stdscale.transform(basetable[reg_cols])

    for model in models:

        y_pred.append(

            metric.bound_prediction(

                model.predict(basetable[reg_cols], batch_size=1024)[0],

                basetable_bound

            )

        )

        

    y_pred = np.mean(y_pred, axis=0)

    y_pred = metric.convert_to_y199_cumsum(y_pred)

    preds_df = pd.DataFrame(data=[y_pred.flatten()], columns=sample_prediction_df.columns)

    env.predict(preds_df)

    

env.write_submission_file()