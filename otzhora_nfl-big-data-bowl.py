from kaggle.competitions import nflrush

import pandas as pd



# You can only call make_env() once, so don't lose it!

env = nflrush.make_env()
import matplotlib

%matplotlib inline 

import seaborn as sns

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm

import numpy as np



import sklearn

from sklearn.preprocessing import StandardScaler



import keras

from keras.layers import Dense, Input

from keras.models import Sequential

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

import keras.backend as K
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df.describe()
train_df.count()
train_df.columns[(train_df.count() - 509762).to_numpy().nonzero()]
-train_df.count() + 509762
def change_play_direction(x):

    x["X"] = 120 - x["X"]

    x["Orientation"] = 360 - x["Orientation"]

    x["Dir"] = 360 - x["Dir"]

    return x
train_df[train_df["PlayDirection"] == "left"] = train_df[train_df["PlayDirection"] == "left"].apply(change_play_direction, axis=1)
train_df["Home"] = train_df["Team"] == "home"
map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

for abb in train_df['PossessionTeam'].unique():

    map_abbr[abb] = abb
train_df["PossessionTeam"] = train_df["PossessionTeam"].map(map_abbr)

train_df["HomeTeamAbbr"] = train_df["HomeTeamAbbr"].map(map_abbr)

train_df["VisitorTeamAbbr"] = train_df["VisitorTeamAbbr"].map(map_abbr)
train_df['HomePossession'] = train_df['PossessionTeam'] == train_df['HomeTeamAbbr']
train_df["Possession"] = train_df["HomePossession"] == train_df["Home"]
train_df.head(44)
train_df['IsRusher'] = train_df['NflId'] == train_df['NflIdRusher']
train_df = train_df.sort_values(by=['PlayId', "Possession", 'IsRusher', "Position"]).reset_index()
np.all(train_df.iloc[21::22]["IsRusher"].tolist())
train_df.iloc[110:132]["Position"]
plays = train_df["PlayId"].unique()
def merge_df(train_df, player_features_list, game_features_list):

    columns_list = []

    for i in range(22):

        for feature in player_features_list:

            columns_list.append(feature+"_"+str(i))

            

    merged_df = pd.DataFrame(index=range(len(plays)))

    

    # game features 

    for game_feature in game_features_list:

        merged_df = merged_df.assign(**{game_feature: train_df[game_feature][::22].tolist()})

        

    j = 0

    for i in range(22):

        for feature in player_features_list:

            merged_df = merged_df.assign(**{columns_list[j]: train_df[feature][i::22].tolist()})

            j += 1

    return merged_df
merged_df = merge_df(train_df, ["X", "Y", "A", "S", "Orientation", "Dir"], ["Yards"])
merged_df.head()
X_train = merged_df.drop(columns=["Yards"])

y_train = merged_df["Yards"]
X_train = X_train.to_numpy()
y_train = np.zeros(shape=(X_train.shape[0], 199))

for i,yard in enumerate(merged_df['Yards']):

    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
def crps(y_true, y_pred):

    return K.mean(K.square(y_true - K.cumsum(y_pred, axis=1)), axis=1)
class RAdam(keras.optimizers.Optimizer):

    """RAdam optimizer.

    # Arguments

        learning_rate: float >= 0. Learning rate.

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



    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., weight_decay=0., amsgrad=False,

                 total_steps=0, warmup_proportion=0.1, min_lr=0., **kwargs):

        learning_rate = kwargs.pop('lr', learning_rate)

        super(RAdam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.learning_rate = K.variable(learning_rate, name='learning_rate')

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

            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)

            decay_rate = (self.min_lr - lr) / decay_steps

            lr = K.switch(

                t <= warmup_steps,

                lr * (t / warmup_steps),

                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),

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

                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))

                self.updates.append(K.update(vhat, vhat_t))

            else:

                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))



            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) *

                         (sma_t - 2.0) / (sma_inf - 2.0) *

                         sma_inf / sma_t)



            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)



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



    @property

    def lr(self):

        return self.learning_rate



    @lr.setter

    def lr(self, learning_rate):

        self.learning_rate = learning_rate



    def get_config(self):

        config = {

            'learning_rate': float(K.get_value(self.learning_rate)),

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
def get_model():

    x = keras.layers.Input(shape=[X_train.shape[1]])

    fc1 = keras.layers.Dense(units=250, input_shape=[X_train.shape[1]])(x)

    act1 = keras.layers.PReLU()(fc1)

    bn1 = keras.layers.BatchNormalization()(act1)

    dp1 = keras.layers.Dropout(0.55)(bn1)

    gn1 = keras.layers.GaussianNoise(0.15)(dp1)

    concat1 = keras.layers.Concatenate()([x, gn1])

    fc2 = keras.layers.Dense(units=300)(concat1)

    act2 = keras.layers.PReLU()(fc2)

    bn2 = keras.layers.BatchNormalization()(act2)

    dp2 = keras.layers.Dropout(0.55)(bn2)

    gn2 = keras.layers.GaussianNoise(0.15)(dp2)

    concat2 = keras.layers.Concatenate()([concat1, gn2])

    fc3 = keras.layers.Dense(units=300)(concat2)

    act3 = keras.layers.PReLU()(fc3)

    bn3 = keras.layers.BatchNormalization()(act3)

    dp3 = keras.layers.Dropout(0.55)(bn3)

    gn3 = keras.layers.GaussianNoise(0.15)(dp3)

    concat3 = keras.layers.Concatenate([concat2, gn3])

    output = keras.layers.Dense(units=199, activation='softmax')(concat2)

    model = keras.models.Model(inputs=[x], outputs=[output])

    return model



def train_model(X_train, y_train, X_val, y_val):

    model = get_model()

    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-7), loss=crps)

    er = EarlyStopping(patience=20, min_delta=1e-4, restore_best_weights=True, monitor='val_loss')

    model.fit(X_train, y_train, epochs=200, validation_data=[X_val, y_val], batch_size=128)

    return model
from sklearn.model_selection import RepeatedKFold



rkf = RepeatedKFold(n_splits=5, n_repeats=5)



models = []



for tr_idx, vl_idx in rkf.split(X_train, y_train):

   

    x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

    x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

    #model = train_model(x_tr, y_tr, x_vl, y_vl)33

    #models.append(model)
# You can only iterate through a result from `env.iter_test()` once

# so be careful not to lose it once you start iterating.

iter_test = env.iter_test()
(test_df, sample_prediction_df) = next(iter_test)
for (test_df, sample_prediction_df) in iter_test:

    

    env.predict(sample_prediction_df)

    
env.write_submission_file()