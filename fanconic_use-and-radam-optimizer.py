import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate, Bidirectional, LayerNormalization, LSTM, Reshape, Embedding

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint
SEED = 11



np.random.seed(SEED)

tf.random.set_seed(SEED)
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_data = train.text.values

train_labels = train.target.values

test_data = test.text.values
%%time

module_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/4'

embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
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
def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def build_model(embed):

    model = Sequential([

        Input(shape=[], dtype=tf.string),

        embed,

        Reshape((1, 512)),

        Bidirectional(LSTM(128,return_sequences=True)),

        LayerNormalization(),

        Bidirectional(LSTM(64)),

        LayerNormalization(),

        Dense(64, activation='relu'),

        Dropout(0.5),

        Dense(1, activation='sigmoid')

    ])

    model.compile(RAdam(learning_rate=1e-4, min_lr=1e-7, warmup_proportion=0.125), loss='binary_crossentropy', metrics=['accuracy', f1])

    

    return model
model = build_model(embed)

model.summary()



tf.keras.utils.plot_model(

    model,

    to_file='model.png',

    show_shapes=False,

    show_layer_names=True,

    rankdir='TB',

)
checkpoint = ModelCheckpoint('model.h5', 

                             monitor='val_f1', 

                             save_best_only=True,

                             mode='max',

                             verbose=1)



learn_control = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1', 

                                  mode='max',  

                                  patience=4,

                                  verbose=1,

                                  factor=0.2, 

                                  min_lr=1e-7)



history = model.fit(train_data, train_labels,

                          validation_split=0.2,

                          epochs=20,

                          callbacks=[checkpoint, learn_control],

                          batch_size=32

                         )
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()
history_df = pd.DataFrame(history.history)

history_df[['f1', 'val_f1']].plot()
model.load_weights('model.h5')

test_pred = model.predict(test_data)
submission['target'] = test_pred.round().astype(int)

submission.to_csv('submission.csv', index=False)