import tensorflow as tf
import os



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from PIL import Image

np.random.seed(42)

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score

import itertools





import keras

from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, Callback

from keras.wrappers.scikit_learn import KerasClassifier

from keras.applications.resnet50 import ResNet50

from keras import backend as K 





import csv

from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, balanced_accuracy_score
os.listdir('../input/data/data/NR-ER-train')
smiles_train = pd.read_csv('../input/data/data/NR-ER-train/names_labels.csv', names=["names", "label"])

smiles_test = pd.read_csv('../input/data/data/NR-ER-test/names_labels.csv', names=["names", "label"])
root = '/kaggle/input/data'

path_train_names = root + '/data/NR-ER-train/names_onehots.npy'

path_train_labels = root + '/data/NR-ER-train/names_labels.csv'

path_test_names = root + '/data/NR-ER-test/names_onehots.npy'

path_test_labels = root + '/data/NR-ER-test/names_labels.csv'



# Write Lables from csv to onehot list

def construct_labels(path_to_file):

        labels = []

        with open(path_to_file) as csv_file:

                csv_reader = csv.reader(csv_file, delimiter= ',')

                for row in csv_reader:

                        if int(row[1]) == 0:

                                labels.append([1,0])

                

                        elif int(row[1]) == 1:

                                labels.append([0,1])



        return np.asarray(labels)



# Write OneHots to list

def construct_names (path_to_file):

        names = []

        df = np.load(path_to_file, allow_pickle=True).tolist()

        names = df.get('onehots')

        return np.asarray(names)
y_train = construct_labels(path_train_labels)

y_test = construct_labels(path_test_labels)



X_train = construct_names(path_train_names)

X_test = construct_names(path_test_names)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=11)
X_train = np.argmax(X_train, axis=1)

X_test = np.argmax(X_test, axis=1)

X_val = np.argmax(X_val, axis=1)



print(X_train.shape)

print(X_val.shape)

print(X_test.shape)

print(y_train.shape)

print(y_val.shape)

print(y_test.shape)
train_df = pd.DataFrame(np.argmax(y_train, axis=-1))

train_df.hist()

train_df[0].value_counts()
val_df = pd.DataFrame(np.argmax(y_val, axis=-1))

val_df.hist()

val_df[0].value_counts()
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

from tensorflow.python import ops, math_ops, state_ops, control_flow_ops

from tensorflow.python.keras import backend as K



__all__ = ['RAdam']





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
from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train, axis=-1)), np.argmax(y_train, axis=-1))
network_input = tf.keras.layers.Input(shape=(X_train[0].shape), dtype='float32')



embedding_layer = tf.keras.layers.Embedding(1000, 32, input_length= X_train[0].shape[0])

embedding = embedding_layer(network_input)



# input_shape=(None, max_len, 16), output_shape=(None, max_len, 64)

lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True))

lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))

layer_norm1 = tf.keras.layers.LayerNormalization()

layer_norm2 = tf.keras.layers.LayerNormalization()

network = lstm2(layer_norm2(lstm1(layer_norm1(embedding))))

output = tf.keras.layers.Dense(2, activation="softmax")(network)





model = tf.keras.Model([network_input], [output])

model.compile(loss="categorical_crossentropy", optimizer=RAdam(learning_rate=1e-3, 

                                                          min_lr=1e-7,

                                                          warmup_proportion=0.15), metrics=['accuracy'])

print(model.summary())



# for this to succeed run `brew install graphviz && pip install pydot_ng`

tf.keras.utils.plot_model(

    model,

    to_file='model.png',

    show_shapes=False,

    show_layer_names=True,

    rankdir='TB',

)
epochs = 30

batch_size = 128
class Balanced_Accuracy(tf.keras.callbacks.Callback):

    def __init__(self, val_data, batch_size = 128):

        super().__init__()

        self.validation_data = val_data

        self.batch_size = batch_size

        

    def on_train_begin(self, logs={}):

        self._data = [] 



    def on_epoch_end(self, epoch, logs={}):

        batches = len(self.validation_data)

        total = batches * self.batch_size



        xVal, yVal = self.validation_data

        val_pred = np.argmax((self.model.predict(xVal, verbose= 0)), axis= 1)

        val_true = np.argmax(yVal, axis= 1)

            

        val_pred = np.squeeze(val_pred)

        _val_ba = balanced_accuracy_score(val_true, val_pred)

        

        print('val balanced accuracy: ', _val_ba)

        self._data.append({'val_balanced_accuracy': _val_ba})

        return



balanced_accuracy = Balanced_Accuracy((X_val, y_val), batch_size = batch_size)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), class_weight= class_weight,

                    epochs= epochs, batch_size= batch_size, verbose=1, 

                    callbacks=[balanced_accuracy]

                   )

 

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.metrics import recall_score

pred_val = model.predict(X_val)

print(balanced_accuracy_score(np.argmax(y_val,  axis= -1), np.argmax(pred_val,  axis= -1)))

print(recall_score(np.argmax(y_val,  axis= -1), np.argmax(pred_val,  axis= -1)))

print(recall_score(np.argmax(y_val,  axis= -1), np.argmax(pred_val,  axis= -1), pos_label=0))
y_pred = model.predict(X_test)
print(confusion_matrix(np.argmax(y_test,  axis= -1), np.argmax(y_pred,  axis= -1)))

print(balanced_accuracy_score(np.argmax(y_test,  axis= -1), np.argmax(y_pred,  axis= -1)))