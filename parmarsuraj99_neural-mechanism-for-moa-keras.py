import numpy as np

import pandas as pd



from sklearn.metrics import log_loss

from sklearn.preprocessing import MinMaxScaler



import tensorflow as tf

from tensorflow.keras import layers as L

from tensorflow.keras.callbacks import *

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential



train_df = pd.read_csv('../input/lish-moa/train_features.csv')

test_df = pd.read_csv('../input/lish-moa/test_features.csv')

train_target_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')



target_cols = train_target_df.columns[1:]

N_TARGETS = len(target_cols)
print(train_df["cp_dose"].unique())

print(train_df["cp_type"].unique())
def preprocess_df(df, target=False):

    

    

    scaler = MinMaxScaler()

    df["cp_time"]=scaler.fit_transform(df["cp_time"].values.reshape(-1, 1))

    

    df["cp_dose"]=(df["cp_dose"]=="D1").astype(int)

    df["cp_type"]=(df["cp_type"]=="trt_cp").astype(int)

    

    return df
train_df
train_target_df
test_df
x_train = preprocess_df(train_df.drop(["sig_id"], axis=1))

y_train = train_target_df.drop(["sig_id"], axis=1)



x_test = preprocess_df(test_df.drop(["sig_id"], axis=1))
sample_sub
x_train.shape, y_train.shape, x_test.shape
def get_keras_model(input_dim=875, output_dim=206):

    

    model = Sequential()

    model.add(L.Dense(512, input_dim=875, activation="elu"))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.5))

    model.add(L.Dense(256, activation="elu"))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.3))

    model.add(L.Dense(128, activation="elu"))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.2))

    model.add(L.Dense(206, activation="sigmoid"))

    

    return model

    

    
model = get_keras_model()

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),

             loss="binary_crossentropy",

             metrics=["accuracy", "AUC"])
def multi_log_loss(y_true, y_pred):

    losses = []

    for col in y_true.columns:

        losses.append(log_loss(y_true.loc[:, col], y_pred.loc[:, col]))

    return np.mean(losses)
#https://github.com/bckenstler/CLR



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
clr = CyclicLR(base_lr=0.003, max_lr=0.004,

                    step_size=745, mode='exp_range',

                    gamma=0.99994)
hist = model.fit(x_train, y_train, epochs=10, callbacks=[clr])
#TODO: Cross validation
y_train.values.shape
ps = model.predict(x_train); ps.shape
ps_df = y_train.copy()

ps_df.iloc[:, : ] = ps



tr_score = multi_log_loss(y_train, ps_df)



print(f"Train score: {tr_score}")
test_df
test_preds = sample_sub.copy()

test_preds[target_cols] = 0



test_preds.loc[:,target_cols] = model.predict(x_test)



test_preds.loc[x_test['cp_type'] == 0, target_cols] = 0

test_preds.to_csv('submission.csv', index=False)
test_preds