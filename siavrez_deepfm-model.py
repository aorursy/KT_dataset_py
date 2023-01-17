!pip install --no-warn-conflicts -q deepctr
from sklearn.metrics import log_loss, roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names

from tensorflow.keras.models import Model, load_model

from deepctr.models import DeepFM

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.optimizers import Adam,RMSprop

from tensorflow.keras.layers import Activation

from tensorflow.keras import backend as K

from tensorflow.keras import callbacks

from tensorflow.keras import utils

import tensorflow.keras as keras

import tensorflow as tf

import pandas as pd

import numpy as np

import warnings

warnings.simplefilter('ignore')
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')
test['target'] = -1
data = pd.concat([train, test]).reset_index(drop=True)
data['null'] = data.isna().sum(axis=1)
sparse_features = [feat for feat in train.columns if feat not in ['id','target']]



data[sparse_features] = data[sparse_features].fillna('-1', )
for feat in sparse_features:

    lbe = LabelEncoder()

    data[feat] = lbe.fit_transform(data[feat].fillna('-1').astype(str).values)
train = data[data.target != -1].reset_index(drop=True)

test  = data[data.target == -1].reset_index(drop=True)
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique()) for feat in sparse_features]



dnn_feature_columns = fixlen_feature_columns

linear_feature_columns = fixlen_feature_columns



feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
def focal_loss(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon()+pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed



get_custom_objects().update({'focal_loss_fn': focal_loss()})
def custom_gelu(x):

    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))



get_custom_objects().update({'custom_gelu': Activation(custom_gelu)})
class Mish(Activation):

    '''

    Mish Activation Function.

    .. math::

        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`

        (tuple of integers, does not include the samples axis)

        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Examples:

        >>> X = Activation('Mish', name="conv1_act")(X_input)

    '''



    def __init__(self, activation, **kwargs):

        super(Mish, self).__init__(activation, **kwargs)

        self.__name__ = 'Mish'





def mish(inputs):

    return inputs * tf.math.tanh(tf.math.softplus(inputs))





get_custom_objects().update({'Mish': Mish(mish)})
class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):

    """Warmup learning rate scheduler

    """



    def __init__(self, warmup_batches, init_lr, verbose=0):

        """Constructor for warmup learning rate scheduler



        Arguments:

            warmup_batches {int} -- Number of batch for warmup.

            init_lr {float} -- Learning rate after warmup.



        Keyword Arguments:

            verbose {int} -- 0: quiet, 1: update messages. (default: {0})

        """



        super(WarmUpLearningRateScheduler, self).__init__()

        self.warmup_batches = warmup_batches

        self.init_lr = init_lr

        self.verbose = verbose

        self.batch_count = 0

        self.learning_rates = []



    def on_batch_end(self, batch, logs=None):

        self.batch_count = self.batch_count + 1

        lr = K.get_value(self.model.optimizer.lr)

        self.learning_rates.append(lr)



    def on_batch_begin(self, batch, logs=None):

        if self.batch_count <= self.warmup_batches:

            lr = self.batch_count*self.init_lr/self.warmup_batches

            K.set_value(self.model.optimizer.lr, lr)

            if self.verbose > 0:

                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '

                      'rate to %s.' % (self.batch_count + 1, lr))
class CyclicLR(keras.callbacks.Callback):



    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',

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

                self.scale_fn = lambda x: 1 / (2. ** (x - 1))

                self.scale_mode = 'cycle'

            elif self.mode == 'exp_range':

                self.scale_fn = lambda x: gamma ** (x)

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

        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))

        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)

        if self.scale_mode == 'cycle':

            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)

        else:

            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(

                self.clr_iterations)



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



        K.set_value(self.model.optimizer.lr, self.clr())

target = ['target']

N_Splits = 50

Verbose = 0

Epochs = 10

SEED = 2020

Batch_S_T = 128

Batch_S_P = 512
oof_pred_deepfm = np.zeros((len(train), ))

y_pred_deepfm = np.zeros((len(test), ))





skf = StratifiedKFold(n_splits=N_Splits, shuffle=True, random_state=SEED)

for fold, (tr_ind, val_ind) in enumerate(skf.split(train, train[target])):

    X_train, X_val = train[sparse_features].iloc[tr_ind], train[sparse_features].iloc[val_ind]

    y_train, y_val = train[target].iloc[tr_ind], train[target].iloc[val_ind]

    train_model_input = {name:X_train[name] for name in feature_names}

    val_model_input = {name:X_val[name] for name in feature_names}

    test_model_input = {name:test[name] for name in feature_names}

    model = DeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256), dnn_dropout=0.0, dnn_activation='Mish', dnn_use_bn=False, task='binary')

    model.compile('adam', loss = 'binary_crossentropy', metrics=[auc], )

    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=3, verbose=Verbose, mode='max', baseline=None, restore_best_weights=True)

    sb = callbacks.ModelCheckpoint('./nn_model.w8', save_weights_only=True, save_best_only=True, verbose=Verbose)

    clr = CyclicLR(base_lr=1e-7, max_lr = 1e-4, step_size= int(1.0*(test.shape[0])/(Batch_S_T*4)) , mode='exp_range', gamma=1.0, scale_fn=None, scale_mode='cycle')

    history = model.fit(train_model_input, y_train,

                        validation_data=(val_model_input, y_val),

                        batch_size=Batch_S_T, epochs=Epochs, verbose=Verbose,

                        callbacks=[es, sb, clr],)

    model.load_weights('./nn_model.w8')

    val_pred = model.predict(val_model_input, batch_size=Batch_S_P)

    print(f'validation AUC fold {fold+1} : {round(roc_auc_score(y_val, val_pred), 5)}')

    oof_pred_deepfm[val_ind] = val_pred.ravel()

    y_pred_deepfm += model.predict(test_model_input, batch_size=Batch_S_P).ravel() / (N_Splits)

    K.clear_session()

print(f'OOF AUC : {round(roc_auc_score(train.target.values, oof_pred_deepfm), 5)}')
test_idx = test.id.values

submission = pd.DataFrame.from_dict({

    'id': test_idx,

    'target': y_pred_deepfm

})

submission.to_csv('submission.csv', index=False)

print('Submission file saved!')
np.save('oof_pred_deepfm.npy',oof_pred_deepfm)

np.save('y_pred_deepfm.npy',    y_pred_deepfm)