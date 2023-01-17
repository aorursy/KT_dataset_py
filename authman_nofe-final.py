RUN_KAGGLE = True



''' PRODUCTION SETTINGS

10SPLITS-KFOLD



'''



n_runs = 4 # n-1 will be kept.

n_splits = 10 # Use 10 for final sub!!!!!!!



# Cols we use to bound predictions:

bounding_cols = ['rusherX','rusherSX','YardLine']



MIN_CLIP = -9

MAX_CLIP = 70

NUM_PLAYERS_TRIM = 2 # how many distant players to trim off the field



DO_LGBM = False



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
@numba.jit(numba.f8[:](numba.f8[:]))

def norm(x):

    return x / x.sum(axis=1).reshape(-1,1)

    

@numba.jit(numba.f8[:](numba.f8[:]))

def cumsum(x):

    return np.clip(np.cumsum(x, axis=1), 0, 1)



@numba.jit(numba.f8[:](numba.f8[:], numba.f8[:]))

def crps(y_pred, y_true):

    return ((y_true - y_pred) ** 2).sum(axis=1).sum(axis=0) / (y_true.shape[1] * y_true.shape[0]) 



@numba.jit

def euclidean_distance(x1,y1,x2,y2):

    x_diff = x1-x2

    y_diff = y1-y2

    return math.sqrt(x_diff*x_diff + y_diff*y_diff)

    

@numba.jit

def euclidean_flat(p1,p2):

    x_diff = p2[:,0]-p1[:,0]

    y_diff = p2[:,1]-p1[:,1]

    return np.sqrt(x_diff*x_diff + y_diff*y_diff)
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



def NearestEnemy(subset):

    mycoords = subset.XY

    defp_coords = subset.defp_coords



    mycoords = np.repeat(mycoords.reshape(1,-1), defp_coords.shape[0], axis=0)

    return euclidean_flat(defp_coords, mycoords).min()

    
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

        rusherX = non_scaled_data.rusherX.astype(int).values

        YardLine = non_scaled_data.YardLine.values

        Slippage = (non_scaled_data.rusherSX.values > 0).astype(int)



        # We can lose up to our rusher position yards and 3 extra

        # However if rusher is currently moving in the +x direction

        # Then that 3 extra turns into 2

        # We also bound to the stadium (e.g. 1 yard line)

        MaxLossYardLine = np.maximum(0, rusherX - 3 + Slippage)



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

        

        return cumsum(output)



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
def fast_preprocess(df, isTrain=False):

    global MIN_CLIP,MAX_CLIP

    seconds_in_year = 60*60*24*365.25



    t = time()

    

    #####################################################################################

    # Clean Data:

    

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



    # TODO: Either do this or keep season as a feature:

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

    

    df['dist_next_point_time'] = np.square(df.S) + 2 * df.A * df.Dis

    df.dist_next_point_time = np.log(10+df.dist_next_point_time)

    

    

    #####################################################################################

    # Rusher Features:

    

    features = [

        'GameId','PlayId','X','Y','Dir',

        'YardLine','Season','S','A','Dis',

        'dist_next_point_time',

        

    ]

    if isTrain: features += ['Yards']

    rushers = df[df.IsRusher][features].copy()

    

    # Per dset, these should be stable. I'm not worried:

    rushers.Dir.replace({np.nan: rushers.Dir.mean()}, inplace=True)

    rushers.A.replace({np.nan: rushers.A.mean()}, inplace=True)

    rushers.S.replace({np.nan: rushers.S.mean()}, inplace=True)

    rushers.Dis.replace({np.nan: rushers.Dis.mean()}, inplace=True)

    

    rushers.rename(columns={

        'X':'rusherX',

        'Y':'rusherY',

    }, inplace=True)

    

    rushers['rusherSX'] = rushers.S * np.cos(rushers.Dir)

    rushers['rusherSY'] = rushers.S * np.sin(rushers.Dir)

    rushers['dist_yardline'] = rushers.YardLine - rushers.rusherX

    rushers.A = np.log1p(rushers.A)

    rushers.Dis = np.log1p(rushers.Dis) 



    if isTrain: 

        rushers['Orig_Yards_NOTRAIN'] = rushers.Yards.copy()

        rushers.Yards = rushers.Yards.clip(MIN_CLIP,MAX_CLIP)

    

    

    # Verify rusher feats...

    rushercols = [

        # Very Important

        'A', 'S', 'rusherSX', 'dist_yardline', 'Season',



        # Testing

        # ?

        

        # Hurts Me

        #'rusherX','Dis','Dir',

        



        # Removing seems to hurt marginally

        'dist_next_point_time', 'YardLine', 'rusherY',

    ]

    

    #####################################################################################

    # Player features

    

    

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

    

    del ball_coords, InfluenceRusherX; gc.collect()

    rushercols.append('InfluenceRusherX_flip')

    

    

    ### TODO:

    rushers = rushers.merge(

        defp_coords[['PlayId','defp_coords']],

        how='left', on='PlayId'

    )

    rushers['XY'] = rushers[['rusherX','rusherY']].apply(lambda x: x.tolist(), axis=1)

    rushers.XY = rushers.XY.apply(np.array)

    rushers['distEnemyNearest'] = rushers[['XY','defp_coords']].apply(NearestEnemy, axis=1)

    rushercols.append('distEnemyNearest')

    

    del offp_coords, defp_coords; gc.collect()

    

    ##########################################################################################

    

    # Split into offensive / defensive

    player_dist = df[~df.IsRusher][[

        'PlayId','NflId',

        'X','Y','S','A',

        'Dir','Dis','PlayerWeight',

        'IsOnOffense','Position',

    ]]

    player_dist.A = np.log1p(player_dist.A)

    player_dist.Dis = np.log1p(player_dist.Dis)

    player_dist['SX'] = player_dist.S * np.cos(player_dist.Dir)

    player_dist['AX'] = player_dist.A * np.cos(player_dist.Dir)

    player_dist['WX'] = player_dist.PlayerWeight * np.cos(player_dist.Dir)

    

    player_dist = player_dist.merge(rushers[['PlayId','rusherX','rusherY']], on='PlayId', how='inner')

    player_dist['rusherDist'] = player_dist[['X','Y','rusherX','rusherY']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    



    # Sorting by distance to rusher

    player_dist.sort_values('rusherDist', inplace=True)

    defensivePlayers = player_dist[player_dist.IsOnOffense==False].copy()

    offensivePlayers = player_dist[player_dist.IsOnOffense==True].copy()



    ##########################################################

    # Additional rusher columns



    # Count how many defenders are on the side of the field the rusher is moving towards

    # relative to the rusher:

    defense = defensivePlayers.groupby('PlayId').Y.agg(list).reset_index()

    defense.columns = ['PlayId', 'defenseY']



    rushers = rushers.merge(defense, on='PlayId', how='left')

    rushers.defenseY = rushers[['defenseY','rusherY','rusherSY']].apply(lambda row: sum([1 for dy in row.defenseY if np.sign(row.rusherSY)==np.sign(dy - row.rusherY)]), axis=1)

    rushercols.append('defenseY')

    

    

    ##########################################################################################

    # Voroni features of Rusher VS Offense

    vf = vfeats(df[df.IsRusher | ~df.IsOnOffense][['PlayId','IsRusher','X','Y']].copy())

    vf.columns = ['PlayId','vArea']

    rushers = rushers.merge(vf, how='left',on='PlayId')

    rushercols.append('vArea')

    ##########################################################################################

    

    

    

    # TODO: Any feature that isn't being used, remove to speed up compute

    # Doesnt help: Dir, Y, X, Dis, PlayerWeight

    # Helps: rusherDist

    # On the fence: 'A'

    basecols = ['S','rusherDist']

    

    # Per dset, these should be stable. I'm not worried:

    defensivePlayers.sort_values('rusherDist', inplace=True)

    if 'Dir' in defensivePlayers: defensivePlayers.Dir.replace({np.nan: defensivePlayers.Dir.mean()}, inplace=True)

    if 'Dis' in defensivePlayers: defensivePlayers.Dis.replace({np.nan: defensivePlayers.Dis.mean()}, inplace=True)

    defensivePlayers.A.replace({np.nan: defensivePlayers.A.mean()}, inplace=True)

    defensivePlayers.S.replace({np.nan: defensivePlayers.S.mean()}, inplace=True)

    defensivePlayers.Position = defensivePlayers.Position.isin('OLB,MLB,LB,DE,DT,DL,DB'.split(',')).astype(np.float32)

    defensivePlayers['feats'] = defensivePlayers[basecols].apply(lambda x: x.tolist(), axis=1)

    

    offensivePlayers.sort_values('rusherDist', inplace=True)

    if 'Dir' in offensivePlayers: offensivePlayers.Dir.replace({np.nan: offensivePlayers.Dir.mean()}, inplace=True)

    if 'Dis' in offensivePlayers: offensivePlayers.Dis.replace({np.nan: offensivePlayers.Dis.mean()}, inplace=True)

    offensivePlayers.A.replace({np.nan: offensivePlayers.A.mean()}, inplace=True)

    offensivePlayers.S.replace({np.nan: offensivePlayers.S.mean()}, inplace=True)

    offensivePlayers.Position = offensivePlayers.Position.isin('TE,T,G,C,OT,OG'.split(',')).astype(np.float32)

    offensivePlayers['feats'] = offensivePlayers[basecols].apply(lambda x: x.tolist(), axis=1)

    

    

    # Different dir handling for offensive vs defensive players—once we decide too include it

    #if 'Dir' in offensivePlayers: offensivePlayers.Dir = np.mod(offensivePlayers.Dir+np.pi, np.pi*2)

    

    # Flat Lists:

    off_pfeats = offensivePlayers[['PlayId','feats']].groupby('PlayId').agg(lambda x: sum(x, [])[:-len(basecols)*NUM_PLAYERS_TRIM] ).reset_index() # 10

    def_pfeats = defensivePlayers[['PlayId','feats']].groupby('PlayId').agg(lambda x: sum(x, [])[:-len(basecols)*NUM_PLAYERS_TRIM] ).reset_index() # 11



    off_cols = [f'off{player_}_{col_}' for player_ in range(10-NUM_PLAYERS_TRIM) for col_ in basecols]

    def_cols = [f'def{player_}_{col_}' for player_ in range(11-NUM_PLAYERS_TRIM) for col_ in basecols]

    off_pfeats[off_cols] = pd.DataFrame(off_pfeats.feats.tolist(), columns=off_cols)

    def_pfeats[def_cols] = pd.DataFrame(def_pfeats.feats.tolist(), columns=def_cols)

    del off_pfeats['feats'], def_pfeats['feats']



    

    

    

    

    ##########################################################################################

    # Finally (we do these afterward just out of fear of re-ordering; plus we need to alter offPlayers:

    # For the rest of the analysis, we dont count the QB or any other offensive player behind the RB

    offensivePlayers = offensivePlayers[offensivePlayers.X>offensivePlayers.rusherX]

    

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

    rushercols += ['Mean_SX', 'Mean_AX', 'Mean_WX']

    

    ##########################################################################################

    

    

    rushers.sort_values('PlayId', inplace=True)

    off_pfeats.sort_values('PlayId', inplace=True)

    def_pfeats.sort_values('PlayId', inplace=True)

    rushers.reset_index(drop=True, inplace=True)

    off_pfeats.reset_index(drop=True, inplace=True)

    def_pfeats.reset_index(drop=True, inplace=True)

    pfeats = pd.concat([

        rushers,

        off_pfeats[[c for c in off_pfeats.columns if c != 'PlayId']],

        def_pfeats[[c for c in def_pfeats.columns if c != 'PlayId']],

    ], axis=1)

    del off_pfeats, def_pfeats, rushers; gc.collect()

    

    

    # Downcast everything to float32

    for col, dt in zip(pfeats.columns, pfeats.dtypes):

        if col not in ['GameId', 'PlayId', 'Yards', 'Orig_Yards_NOTRAIN'] and dt=='int64':

            pfeats[col] = pfeats[col].astype(np.float32)



        if dt!='float64': continue

        pfeats[col] = pfeats[col].astype(np.float32)

    

    print(time() - t)

    return pfeats, basecols, rushercols
def prep_data(data, basecols, rushercols):

    # Currently sorted by dist to rusher...

    filtered_cols = []

    train_cols = []

    out = {}

    

    for player_ in range(10-NUM_PLAYERS_TRIM):

        cols = [f'off{player_}_{col_}' for col_ in basecols]

        out[f'off{player_}'] = data[cols]

        filtered_cols = sum([filtered_cols, cols], [])



    for player_ in range(11-NUM_PLAYERS_TRIM):

        cols = [f'def{player_}_{col_}' for col_ in basecols]

        out[f'def{player_}'] = data[cols]

        filtered_cols = sum([filtered_cols, cols], [])

        

    # Add rusher specific data

    out['rusher'] = data[rushercols]

    filtered_cols = sum([filtered_cols, rushercols], [])

    

    train_cols = list(out.keys())

    

    # Add non-train columns here

    for col in data.columns:

        if col in filtered_cols: continue

        out[col] = data[col]



    

    return out, train_cols
gc.collect()

what, basecols, rushercols = fast_preprocess(train.copy(), isTrain=False)

what.columns
z = what.isna().sum()

z[z>0]
# # Investigate:

# for col in what.columns:

#     #if 'dist_next_point_time' not in col: continue

#     plt.title(col)

#     plt.hist(what[col], 100)

#     plt.show()
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
keys = X.Yards.unique()-MIN_CLIP

values = class_weight.compute_class_weight('balanced', keys, X.Yards.values-MIN_CLIP)

cweights = dict(zip(keys, values))

cweights

# cweights = None
# Original model, 16,16,32 -> 1024 -> 64

def BuildModel(basecols, rushercols):

    global N_OUTPUTS

    

    n_basecols = len(basecols)

    n_rusher_cols = len(rushercols)

    

    # Inputs

    inputs = [ Input(shape=(n_rusher_cols,), name='rusher') ]

    for i in range(10-NUM_PLAYERS_TRIM): inputs.append(Input(shape=(n_basecols,), name=f'off{i}'))

    for i in range(11-NUM_PLAYERS_TRIM): inputs.append(Input(shape=(n_basecols,), name=f'def{i}'))

    

    # Reused Layers

    off_dense = Dense( 4, activation='elu', name='off_dense') # 8

    def_dense = Dense(16, activation='elu', name='def_dense') # 16?

    run_dense = Dense(128, activation='elu', name='run_dense')

    

    

    # Flow

    x = [GaussianNoise(0.0025)(inp) for inp in inputs]

    x[0] = run_dense(x[0])

    for i in range(1,11-NUM_PLAYERS_TRIM):      x[i] = off_dense(x[i])

    for i in range(11-NUM_PLAYERS_TRIM,len(x)): x[i] = def_dense(x[i])



    x = Concatenate(name='flow')(x)

    x = Dropout(0.25)(x)

    x = Dense(256, activation='elu')(x)

    x = Dense(64, activation='elu')(x)

    x = BatchNormalization()(x)

    

    # mu in raw space [0,N_OUTPUTS)

    # We don't have to add/sub Min_Clip because we're already in raw space starting from 0

    # So we just apply the log1p transform

    mu_raw = Dense(1, activation='relu', name='mu_raw')(x)

    mu = Lambda(lambda x: K.log(1 + K.reshape(x[:,0],(-1,1))), name='mu')(mu_raw)

    

    ent = Dense(N_OUTPUTS, name='ent', activation='sigmoid')(

        Concatenate()([

            x, mu_raw

        ])

    )

    reg_pass = Lambda(lambda x: x, name='reg_pass')(ent) # passthrough for different loss function...

    

    model = Model(

        inputs,

        [ent,reg_pass,mu]

    )

    

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
kX, train_cols = prep_data(X, basecols, rushercols)



global_scalars = {

    tcolgrp: StandardScaler().fit(kX[tcolgrp])

    for tcolgrp in train_cols

}



run_scalars = StandardScaler().fit(kX['rusher'])



def_scalars = StandardScaler().fit(

    pd.concat([

        kX[colgrp].rename(columns={oc:basecols[i] for i,oc in enumerate(kX[colgrp].columns)})

        for colgrp in train_cols

        if 'def' in colgrp

    ], axis=0)

)



off_scalars = StandardScaler().fit(

    pd.concat([

        kX[colgrp].rename(columns={oc:basecols[i] for i,oc in enumerate(kX[colgrp].columns)})

        for colgrp in train_cols

        if 'off' in colgrp

    ], axis=0)

)



del kX; gc.collect()
t = time()

K.clear_session()



if RUN_KAGGLE:

    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)

else:

    kf = GroupKFold(n_splits=n_splits)

    

oof = np.zeros((X.Yards.shape[0], N_OUTPUTS))

oof_mu = np.zeros((X.Yards.shape[0]))

report = []

use_models = []

perms = []



for fold_, (tdx, vdx) in enumerate(kf.split(X, yent, X.GameId)):

    X_train, X_val = X.loc[tdx].copy(), X.loc[vdx].copy()

    X_train_bound, X_val_bound = X_train[bounding_cols].copy(), X_val[bounding_cols].copy()

    y_train_ent, y_train_mu = yent[tdx], ymu[tdx]

    y_val_ent, y_val_mu = yent[vdx], ymu[vdx]

    

    kX_train, train_cols = prep_data(X_train, basecols, rushercols)

    kX_val, _ = prep_data(X_val, basecols, rushercols)

    

    # Global Scaling

    for tcolgrp in train_cols:

        kX_train[tcolgrp] = global_scalars[tcolgrp].transform(kX_train[tcolgrp])

        kX_val[tcolgrp] = global_scalars[tcolgrp].transform(kX_val[tcolgrp])

    

    y_train = {'ent':y_train_ent, 'reg_pass':y_train_ent, 'mu':y_train_mu}

    y_val   = {'ent':y_val_ent,   'reg_pass':y_train_ent, 'mu':y_val_mu  }



    # For Drop Importance:

    y_val_crps = None

    y_val_crps_orig = yent_NOCLIP_CSUM[vdx]

    

    # I say blend the best 3 of 4 runs...

    oof_mu_runs = np.zeros((n_runs, X_val.shape[0]))

    oof_runs = np.zeros((n_runs, X_val.shape[0], N_OUTPUTS))

    oof_run_scores = []

    for run_ in range(n_runs):

        print(f'Fold: {fold_}, Run: {run_}')

        model = BuildModel(basecols, rushercols)

        

        if fold_==0 and run_==0: model.summary()

        es = EarlyStopping(

            monitor='val_CRPS',

            mode='min',

            restore_best_weights=True, 

            verbose=2, 

            patience=20 if RUN_KAGGLE == False else 40

        )



        es.set_model(model)

        metric = Metric(model, [es], [(kX_train,y_train), (kX_val,y_val)])

        metric.set_original_data(X_train_bound, X_val_bound)

        hist = model.fit(kX_train, y_train, class_weight=cweights, callbacks=[metric], epochs=400, batch_size=1024//2, verbose=False)

        model.save(f'fold_{fold_}_{run_}.h5')

        score_ = min(hist.history['val_CRPS'])

        oof_run_scores.append(score_)

        

        # Stash OOF:

        preds, _, preds_mu = model.predict(kX_val, batch_size=1024)

        preds = metric.bound_prediction(preds, X_val_bound)

        oof_runs[run_] = preds.copy()

        oof_mu_runs[run_] = preds_mu.flatten().copy()

        preds = metric.convert_to_y199_cumsum(preds)

        

        

        if y_val_crps is None:

            y_val_crps = metric.convert_to_y199_cumsum(y_val_ent)

        

        orig_score_ = crps(preds, y_val_crps_orig)

        print(f'RunScore: {score_}, (Original): {orig_score_}')





    # Top-k, by minimizing score

    worst_run = np.argmax(oof_run_scores)

    bad_model = f'fold_{fold_}_{worst_run}.h5'

    use_models += [f'fold_{fold_}_{run_}.h5' for run_ in range(n_runs) if run_ != worst_run]

    

    if DO_LGBM:

        # Add in a LGM model...

        pass

    

    # Average the good runs

    oof[vdx] = np.delete(oof_runs, worst_run, axis=0).mean(axis=0)

    oof_mu[vdx] = np.delete(oof_mu_runs, worst_run, axis=0).mean(axis=0)

    

    # Transform to competition space, only for evaluation

    preds = metric.convert_to_y199_cumsum(oof[vdx])

    score_ = crps(preds, y_val_crps)

    orig_score_ = crps(preds, y_val_crps_orig)

    print(oof_run_scores, np.array(oof_run_scores).std(), 'STD') # todo: - make these orig score

    report.append(f'Fold: {fold_}, TopK Mean Score: {score_}, (Original): {orig_score_}')

    print(report[-1], end='\n\n')

    print('—'*100)

    

oof = metric.convert_to_y199_cumsum(oof)

oof_yent = metric.convert_to_y199_cumsum(yent)

score_ = crps(oof, oof_yent)

orig_score_ = crps(oof, yent_NOCLIP_CSUM)

report.append(f'Final Model Score: {score_}, (Original): {orig_score_}')

report.append(str(int(time() - t)) + ' Seconds to Train')

report
from kaggle.competitions import nflrush

env = nflrush.make_env()

iter_test = env.iter_test()
models = []

for model_path in use_models:

    models.append(load_model(model_path))
for (test_df, sample_prediction_df) in iter_test:

    test_df1, _, _ = fast_preprocess(test_df.copy(), isTrain=False)

    kX_sub_bound = test_df1[bounding_cols].copy()

    kX_sub, _ = prep_data(test_df1, basecols, rushercols)



    # Global Scaling

    for tcolgrp in train_cols:

        kX_sub[tcolgrp] = global_scalars[tcolgrp].transform(kX_sub[tcolgrp])



    y_pred = np.mean(

        [

            metric.bound_prediction(

                model.predict(kX_sub, batch_size=1024)[0],

                kX_sub_bound

            )

            for model in models    

        ],

        axis=0

    )

    y_pred = metric.convert_to_y199_cumsum(y_pred)

    preds_df = pd.DataFrame(data=[y_pred.flatten()], columns=sample_prediction_df.columns)

    env.predict(preds_df)

    

env.write_submission_file()