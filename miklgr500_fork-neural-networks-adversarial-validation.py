import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import datetime

from kaggle.competitions import nflrush

import tqdm

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

import keras



sns.set_style('darkgrid')

mpl.rcParams['figure.figsize'] = [15,10]
env = nflrush.make_env()
train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})
train.drop('Yards', axis=1, inplace=True) 
train['target'] = 1

merge = [train] 



for test, sample in tqdm.tqdm(env.iter_test()):

    test['target'] = 0

    merge.append(test)

    env.predict(pd.DataFrame(data=np.zeros((len(test.PlayId.unique()), 199)) ,columns=sample.columns))



train = pd.concat(merge) 
train.head()
train['PlayId'].value_counts()
cat_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append((col, len(train[col].unique())))
off_form = train['OffenseFormation'].unique()

train['OffenseFormation'].value_counts()
train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)

dummy_col = train.columns
train['GameClock'].value_counts()
def strtoseconds(txt):

    txt = txt.split(':')

    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60

    return ans
train['GameClock'] = train['GameClock'].apply(strtoseconds)
sns.distplot(train['GameClock'])
train['PlayerHeight']
train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
train['TimeHandoff']
train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))

train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
seconds_in_year = 60*60*24*365.25

train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)
train['WindSpeed'].value_counts()
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
train['WindSpeed'].value_counts()
#let's replace the ones that has x-y by (x+y)/2

# and also the ones with x gusts up to y

train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)

train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
def str_to_float(txt):

    try:

        return float(txt)

    except:

        return -1
train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)
train['WindDirection'].value_counts()
train.drop('WindDirection', axis=1, inplace=True)
train['PlayDirection'].value_counts()
train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')
train['GameWeather'].unique()
train['GameWeather'] = train['GameWeather'].str.lower()

indoor = "indoor"

train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)

train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)

train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)

train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
train['GameWeather'].unique()
from collections import Counter

weather_count = Counter()

for weather in train['GameWeather']:

    if pd.isna(weather):

        continue

    for word in weather.split():

        weather_count[word]+=1

        

weather_count.most_common()[:15]
def map_weather(txt):

    ans = 1

    if pd.isna(txt):

        return 0

    if 'partly' in txt:

        ans*=0.5

    if 'climate controlled' in txt or 'indoor' in txt:

        return ans*3

    if 'sunny' in txt or 'sun' in txt:

        return ans*2

    if 'clear' in txt:

        return ans

    if 'cloudy' in txt:

        return -ans

    if 'rain' in txt or 'rainy' in txt:

        return -2*ans

    if 'snow' in txt:

        return -3*ans

    return 0
train['GameWeather'] = train['GameWeather'].apply(map_weather)
train['IsRusher'] = train['NflId'] == train['NflIdRusher']
train.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
train.drop(['GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1, inplace=True)
cat_features = []

for col in train.columns:

    if train[col].dtype =='object':

        cat_features.append(col)

        

train = train.drop(cat_features, axis=1)
train.fillna(-999, inplace=True)
players_col = []

for col in train.columns:

    if train[col][:22].std()!=0:

        players_col.append(col)
X_train = np.array(train[players_col]).reshape(-1, 11*22)
play_col = train.drop(players_col+['target'], axis=1).columns

X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))

for i, col in enumerate(play_col):

    X_play_col[:, i] = train[col][::22]
X_train = np.concatenate([X_train, X_play_col], axis=1)

y_train = train['target'][::22].values
from keras.callbacks import EarlyStopping
from keras import backend as K





__all__ = ['RAdam']





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
def plot_with_dots(ax, np_array):

    ax.scatter(list(range(1, len(np_array) + 1)), np_array, s=50)

    ax.plot(list(range(1, len(np_array) + 1)), np_array)
def train_model(x_tr, y_tr, x_vl, y_vl):

    model = keras.models.Sequential([

        keras.layers.Dense(units=512, input_shape=[X_train.shape[1]]),

        keras.layers.BatchNormalization(),

        keras.layers.LeakyReLU(0.3),

        keras.layers.Dropout(0.25),

        

        keras.layers.Dense(units=512),

        keras.layers.BatchNormalization(),

        keras.layers.LeakyReLU(0.3),

    

        keras.layers.Dropout(0.25),

        

        keras.layers.Dense(units=1, activation='sigmoid')

    ])

    

    er = EarlyStopping(patience=3, min_delta=1e-4, restore_best_weights=True, monitor='val_accuracy')

    

    model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5), 

                  loss='binary_crossentropy',

                  metrics=['accuracy'] 

                 )

    h=model.fit(x_tr, y_tr, epochs=10, callbacks=[er], validation_data=[x_vl, y_vl], verbose=0)

    return h.history['val_accuracy'] 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 

X_train = scaler.fit_transform(X_train) 
from sklearn.model_selection import StratifiedKFold 



rkf = StratifiedKFold(n_splits=5)
from keras import backend as K
list_acc = []

for tr_idx, vl_idx in tqdm.tqdm_notebook(rkf.split(X_train, y_train)):

    

    x_tr, y_tr = X_train[tr_idx], y_train[tr_idx]

    x_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

    

    acc = train_model(x_tr, y_tr, x_vl, y_vl)

    list_acc.append(acc)
plt.figure(figsize=(10, 7))

for acc in list_acc:

    plot_with_dots(plt, [0] + list(acc))
env.write_submission_file()