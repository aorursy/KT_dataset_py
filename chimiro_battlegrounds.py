# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Standard libraries

import os

import numpy as np

import random

import pandas as pd

import time

import gc # memory



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

from pdpbox import pdp

from plotnine import *



# Pre-processing

from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Correlation

from scipy.cluster import hierarchy as hc # dendrogram



# Model

from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf

from keras import backend as K

from keras.losses import mse, binary_crossentropy

from keras import optimizers, regularizers

from keras.layers import Input, Dense, Lambda

from keras.models import Sequential, Model, load_model 

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# Evaluate

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn import metrics

from sklearn.tree import export_graphviz



# For notebook plotting

%matplotlib inline
pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 300)

# pd.reset_option('display.float_format')
# Detect hardware, return appropriate distribution strategy

import tensorflow as tf

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):

    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose)
def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
def reload():

    gc.collect()

    df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values

    df = df[-df['matchId'].isin(invalid_match_ids)]

    return df
train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

train = reduce_mem_usage(train)
train.info()
# Summary Statistics of the training data.

round(train.describe(include=np.number).drop('count').T,2)
for c in ['Id','groupId','matchId']:

    print(f'unique [{c}] count:', train[c].nunique())
plt.figure(figsize=(7,7))

sns.barplot(x=train['matchType'].unique(),

            y=train['matchType'].value_counts())

train['matchType'].value_counts().plot.bar()
# Continuous & Discrete variable 



# group id: team id // solo의 경우 각자 group id 할당

# matchid: 각 게임의 고유값

display(list(train.columns[train.dtypes != 'object']))

display(list(train.columns[train.dtypes == 'object']))
train = train.dropna(axis='rows')

display(train.isna().sum())
# display(train[train.duplicated()].count())
def delete_cheaters(df):

    ### Anomalies in roadKills ### 

    # Drop roadKill 'cheaters'

    df.drop(df[df['roadKills'] >= 10].index, inplace=True)



    ### Anomalies in aim 1 (More than 50 kills) ### 

    df.drop(df[df['kills'] >= 50].index, inplace=True)



    ### Anomalies in aim 2 (100% headshot rate) ### 

    #df['headshot_rate'] = df['headshotKills'] / df['kills']

    #df['headshot_rate'] = df['headshot_rate'].fillna(0)   



    ### Anomalies in aim 3 (Longest kill) ### 

    df.drop(df[df['longestKill'] >= 1000].index, inplace=True)



    ### Anomalies in movement ### 

    # walkDistance anomalies 

    df.drop(df[df['walkDistance'] >= 13000].index, inplace=True)



    # rideDistance anomalies 

    df.drop(df[df['rideDistance'] >= 25000].index, inplace=True)



    # swimDistance

    df.drop(df[df['swimDistance'] >= 1500].index, inplace=True)



    ### Anomalies in supplies 1 (weaponsAcquired) ### 

    df.drop(df[df['weaponsAcquired'] >= 50].index, inplace=True)



    ### Anomalies in supplies 2 (heals) ###

    # Remove outliers

    df.drop(df[df['heals'] >= 40].index, inplace=True)

    

    ## ETC ## 

    # drop savage killer (kill streak > 10)

    df = df.drop(df[df['killStreaks'] >= 10].index)

    

    df.drop(df[(df['walkDistance']<=10.0) & (df['damageDealt'] >= 1000)].index, inplace=True)

    df.drop(df[(df['walkDistance']<=10.0) & (df['kills'] >= 5)].index, inplace=True)

    df.drop(df[(df['walkDistance']<=10.0) & (df['heals'] >= 5)].index, inplace=True)

    df.drop(df[(df['walkDistance']<=10.0) & (df['headshotKills'] >= 5)].index, inplace=True)

    df.drop(df[(df['walkDistance']<=10.0) & (df['headshotKills'] >= 5)].index, inplace=True)

    

    return df
def top_k_corr(data, k):

    #f, ax = plt.subplot(figsize=(7,7))

    plt.figure(figsize=(7,7))

    cols = data.corr().nlargest(5, 'winPlacePerc').index

    sns.heatmap(data[cols].corr(), annot=True, fmt='.2f',cbar=True,

                yticklabels=cols.values, xticklabels=cols.values,

                linecolor='white', linewidths=0.1)


plt.figure(figsize=(15,15))

col = train.columns[train.dtypes != 'object']

df_corr = train[col].corr()

sns.heatmap(df_corr, annot=True, fmt='.2f', cbar=True, 

           xticklabels=train[col].columns.values, yticklabels=train[col].columns.values,

           linecolor='white', linewidths=0.1)

top_k_corr(train, 5)           
def feature_engineering(df):

    # New feature

    df['heals_and_boosts'] = df['heals']+df['boosts']

    df['total_distance'] = df['walkDistance']+df['rideDistance']+df['swimDistance']

    df['kills_over_walkDistance'] = df['kills'] / df['walkDistance']

    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']

    df['players_joined'] = df.groupby('matchId')['matchId'].transform('count')

    df['players_in_team'] = df.groupby('groupId')['matchId'].transform('count')

    

    # Drop feature (low correlation with winPlacePerc or similar feature)

    df.drop(['boosts','heals'], axis=1, inplace=True)

    df.drop(['rideDistance','swimDistance','matchDuration'], axis=1, inplace=True)

    df.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)

    df.drop(['headshotKills','roadKills','vehicleDestroys','teamKills'], axis=1, inplace=True)

    

    # Rank as percent

    match = df.groupby('matchId')

    df['killsPerc'] = match['kills'].rank(pct=True).values

    df['killPlacePerc'] = match['killPlace'].rank(pct=True).values

    df['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

    df['walkPerc_killsPerc'] = df['walkDistancePerc'] / df['killsPerc']



    check_cols = ['kills_over_walkDistance', 'killPlace_over_maxPlace', 'walkPerc_killsPerc']

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN # - Inf

    for c in check_cols: df[c].fillna(0, inplace=True)

    

    numcols = df.select_dtypes(include='number').columns

    use_cols = numcols[numcols != 'winPlacePerc']

    

    rank_cols = use_cols.drop(['numGroups', 'maxPlace', 'players_joined', 'players_in_team'])

    

    stat_cols = use_cols.drop(['numGroups', 'maxPlace', 'players_joined', 'players_in_team',

                               'assists', 'longestKill', 'weaponsAcquired','heals_and_boosts', 

                               'total_distance', 'kills_over_walkDistance'])

    

    group = df.groupby(['matchId', 'groupId'])

    

    # Median_by_team

    df = df.merge(group[stat_cols].median(), suffixes=['', '_team_median'], how='left', 

                  on=['matchId', 'groupId'])

    df = reduce_mem_usage(df)

    

    # Max_by_team

    df = df.merge(group[stat_cols].max(), suffixes=['', '_team_max'], how='left', 

                  on=['matchId', 'groupId'])

    df = reduce_mem_usage(df)

    

    # Min_by_team 

    df = df.merge(group[stat_cols].min(), suffixes=['', '_team_min'], how='left', 

                  on=['matchId', 'groupId'])

    df = reduce_mem_usage(df)

    

    # Mean_by_team

    df = df.merge(group[stat_cols].mean(), suffixes=['', '_team_mean'], how='left', 

                  on=['matchId', 'groupId'])

    df = reduce_mem_usage(df)

    

    # Sum_by_team

    df = df.merge(group[stat_cols].sum(), suffixes=['', '_team_sum'], how='left', 

                  on=['matchId', 'groupId'])

    df = reduce_mem_usage(df)

    

    # Rank_by_team 

    df = df.merge(group[rank_cols].mean().groupby('matchId')[rank_cols].rank(pct=True), 

                  suffixes=['', '_team_mean_rank'], how='left', on=['matchId', 'groupId'])

    df = reduce_mem_usage(df)

    

    # One hot encode matchType

    mapper = lambda x: 'solo' if 'solo' in x else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

    df['matchType'] = df['matchType'].apply(mapper)

    df = pd.get_dummies(df, columns=['matchType'])

    

    df.drop(['Id', 'matchId','groupId'], axis=1, inplace=True)

    

    # drop constant column

    constant_column = [col for col in df.columns if df[col].nunique() == 1]

    print('drop constant columns:', constant_column)

    df.drop(constant_column, axis=1, inplace=True)

    

    assert df.isna().sum().sum() == 0





    return df
from sklearn.ensemble import IsolationForest

def isolation_forest(df, contamination):

    cols = list(df.columns[df.dtypes != 'object'])

    train_df = df[cols]   

    

    # max_samples: 각 estimators를 학습시키는데 사용하는 샘플 수 

    # contamination: 데이터셋의 오염 정도 

    # max_features: outlier 측정에 사용할 변수의 수 

    outlier_detect = IsolationForest(n_estimators=500, 

                                     contamination=contamination, 

                                     max_features=train_df.shape[1])

    outlier_detect.fit(train_df) 

    outliers_predicted = outlier_detect.predict(train_df)

    

    df['outlier'] = outliers_predicted

    df = df[df['outlier'] == 1]

    

    return df 
def delete_outlier(y_pred, y_true, remain=0.99):

    

    mse_array = np.square(np.subtract(y_pred, y_true)).mean(axis=1)

    mse_series = pd.Series(mse_array)

    

    check_value = mse_series.quantile(remain)

    check_outlier = np.where(mse_array <= check_value, 1, -1)

    

    return check_outlier, mse_series



def sampling(args):

    """

    Reparameterization trick by sampling from an isotropic unit Gaussian.



    # Arguments

        args (tensor): mean and log of variance of Q(z|X)



    # Returns

        z (tensor): sampled latent vector

    """



    z_mean, z_log_var = args

    batch = tf.keras.backend.shape(z_mean)[0]

    dim = tf.keras.backend.int_shape(z_mean)[1]

    # by default, random_normal has mean = 0 and std = 1.0

    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon



def VAE_model(inputs, origin_dim):

    # network parameters

    latent_dim = 2

    origin_half_dim = origin_dim//2



    # VAE model = encoder + decoder

    # build encoder model

 

    x = tf.keras.layers.Dense(origin_half_dim, activation='relu')(inputs)

    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)

    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)



    # use reparameterization trick to push the sampling out as input

    # note that "output_shape" isn't necessary with the TensorFlow backend

    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])



    # instantiate encoder model

    encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')



    # build decoder model

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')

    x = tf.keras.layers.Dense(origin_half_dim, activation='relu')(latent_inputs)

    outputs = tf.keras.layers.Dense(origin_dim, activation='sigmoid')(x)



    # instantiate decoder model

    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')



    # instantiate VAE model

    outputs = decoder(encoder(inputs)[2])

    vae = tf.keras.Model(inputs, outputs, name='vae_mlp')



    # VAE loss = mse_loss + kl_loss

    reconstruction_loss = tf.keras.losses.MSE(inputs, outputs)



    reconstruction_loss *= origin_dim

    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)

    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)

    kl_loss *= -0.5

    vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)

    

    encoder.summary()

    decoder.summary()



    return vae
def VAE(df, remain_ratio):

    np.random.seed(0)

    tf.random.set_seed(0)

    

    lr_sched = step_decay_schedule(initial_lr=0.001, decay_factor=0.97, step_size=1, verbose=0)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    

    cols = list(df.columns[df.dtypes != 'object'])

    train_df = df[cols]   

    input_shape = (train_df.shape[1], ) 

    origin_dim = train_df.shape[1]

    inputs = tf.keras.Input(shape = input_shape)

    

    # training parameters

    epochs = 300

    batch_size = 20480  # 8  * strategy.num_replicas_in_sync * 16 * 20

    validation_split = 0.2

    steps = len(train_df) * (1-validation_split) // batch_size

    optimizer = tf.keras.optimizers.Adam()

    callbacks_list = [lr_sched, early_stopping]



    # Scaling

    scaler = StandardScaler()                                  

    x_train = scaler.fit_transform(train_df.astype(float))



    # VAE

    with strategy.scope():

        model = VAE_model(inputs, origin_dim)

        model.compile(optimizer=optimizer)



        model.fit( 

                x_train.astype(np.float32),

                epochs=epochs,

                batch_size=batch_size,

                validation_split=validation_split,

                steps_per_epoch=steps,

                callbacks=callbacks_list,

                verbose=0)

        

        print('\nVAE: Predict cheaters')

        y_pred = model.predict(x_train.astype(np.float32), verbose=1, batch_size=batch_size, 

                      workers=strategy.num_replicas_in_sync, use_multiprocessing=True)

        

        print('VAE: Check cheaters')

        # outlier: 1 (normal), -1 (cheater)

        df['outlier'], mse_serise = delete_outlier(y_pred, y_true=x_train, remain=remain_ratio) 

        df = df[df['outlier'] == 1]

        df.drop(columns='outlier', inplace=True)

        

        return df
def std_n_sigma(df, n, filter_):

    cols = list(df.columns[df.dtypes != 'object'])

    filter_ = list(df[filter_]) 

    df = df[cols]   



    scaler = StandardScaler()

    std_array = scaler.fit_transform(df.astype(float))

    std_df = pd.DataFrame(std_array, columns=df.columns, index=filter_)



    # delete outlier  

    remove_idx = set([])

    for col in std_df.columns:

        remove_idx.update(std_df[(std_df[col] < -n) | (n < std_df[col])].index)

                        

    print('PUBG cheaters: {}%'.format(round(len(remove_idx) / len(df) * 100,2)))

    return remove_idx  
def data_preparation(df, train_flag=True):

    origin_size = df.shape[0]

    

    if train_flag:

        ## Delete cheaters ## 

        print("Delete PUBG Cheaters \n")

        

        # Experience-based

        # df = delete_cheaters(df)



        # Standadization

        # remove_id = std_n_sigma(df, 3.5, 'Id')

        # df = df[-df.Id.isin(remove_id)]



        # Isolation forest

        #contamination = 0.001

        #df = isolation_forest(df, contamination)

        

        # VAE

        df = VAE(df, remain_ratio=0.998)

        

        print('PUBG cheaters: {0}, {1}% \n'.format(origin_size -len(df), 

                                                round(100 - 100 * len(df) / origin_size,4))) 

    else:

        pass

    

    print("Feature engineering")

    df = feature_engineering(df)

    

    return df
train_ = data_preparation(train)

train_.head()

gc.collect()
## Debug mode ## 

# sample = 1000000

# df_sample = train.sample(sample, random_state=0)

# df_sample = reduce_mem_usage(df_sample)



train_data = train_.drop(columns = ['winPlacePerc'])

train_labels = train_['winPlacePerc']



train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.1, random_state=0)

print(train_x.shape, train_y.shape, val_y.shape, val_x.shape)



del train_, train_data, train_labels

gc.collect()
from lightgbm import LGBMRegressor

params = {

    'n_estimators': 300,

    'learning_rate': 0.3, 

    'num_leaves': 20,

    'objective': 'regression_l2', 

    'metric': 'mae',

    'verbose': -1,

}



model = LGBMRegressor(**params)

model.fit(

    train_x, train_y,

    eval_set=[(val_x, val_y)],

    eval_metric='mae',

    verbose=20,

)



feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, train_x.columns)), columns=['Value','Feature'])



plt.figure(figsize=(10, 6))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False)[:20])

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()

gc.collect()
import scipy



# Keep only significant features

to_keep = feature_importance.sort_values(by='Value', ascending=False)[:50].Feature



## Create a Dendrogram to view highly correlated features

corr = np.round(scipy.stats.spearmanr(train_x[to_keep]).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(14,20))

dendrogram = hc.dendrogram(z, labels=train_x[to_keep].columns, orientation='left', leaf_font_size=16)

plt.plot()

use_features = 80

im_features = feature_importance.sort_values(by='Value', ascending=False)[:use_features].Feature 



## Scaling ##

scaler = StandardScaler()

X_train = scaler.fit_transform(train_x[im_features].astype(np.float32))

Y_train = train_y.values

X_val = scaler.fit_transform(val_x[im_features].astype(np.float32))

Y_val = val_y.values



gc.collect()
from keras import optimizers, regularizers

from keras.models import Sequential, Model, load_model 

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.layers import Input, Dense, Lambda

from keras.losses import mse, binary_crossentropy

from keras.models import Model

from keras import backend as K

from keras.losses import mse, binary_crossentropy

from sklearn import preprocessing
def model():

    hidden_layer = tf.keras.layers.Dense(2048, kernel_initializer='he_normal', activation='relu')(inputs)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    hidden_layer = tf.keras.layers.Dropout(rate=0.1, seed=1234)(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(1024, kernel_initializer='he_normal', activation='relu')(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    hidden_layer = tf.keras.layers.Dropout(rate=0.1, seed=1234)(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(512, kernel_initializer='he_normal', activation='relu')(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(256, kernel_initializer='he_normal', activation='relu')(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(128, kernel_initializer='he_normal', activation='relu')(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    

    hidden_layer = tf.keras.layers.Dense(64, kernel_initializer='he_normal', activation='relu')(hidden_layer)

    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)

    

    prediction = tf.keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid')(hidden_layer) 

    

    return prediction
np.random.seed(0)

tf.random.set_seed(0)



# Train parameters

epochs = 500

batch_size = 20480 # 8  * strategy.num_replicas_in_sync * 16 * 20

steps = len(X_train) // batch_size

optimizer = tf.keras.optimizers.Adam()

lr_sched = step_decay_schedule(initial_lr=0.001, decay_factor=0.97, step_size=1, verbose=0)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', mode='min', patience=10, verbose=1)



callbacks_list = [lr_sched, early_stopping]



# Deep learning model

with strategy.scope():

    inputs = tf.keras.Input(shape=(X_train.shape[1],))

    predictions = model()

    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae']) # loss='mae' or 'mse'



    history = model.fit(

            X_train.astype(np.float32), Y_train.astype(np.float32),

            shuffle=True,

            epochs=epochs,

            batch_size=batch_size,

            validation_data = (X_val.astype(np.float32), Y_val.astype(np.float32)),

            steps_per_epoch=steps,

            callbacks=callbacks_list,

            verbose=0)
# Plot training & validation loss values

plt.figure(figsize=(7,5))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation mae values

plt.figure(figsize=(7,5))

plt.plot(history.history['mae'])

plt.plot(history.history['val_mae'])

plt.title('Mean Abosulte Error')

plt.ylabel('Mean absolute error')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model.summary()
del X_train, Y_train, X_val, Y_val

gc.collect()    
## Scaling ##

scaler = StandardScaler()

X_train = scaler.fit_transform(train_x[im_features].astype(np.float32))

Y_train = train_y.values

X_val = scaler.fit_transform(val_x[im_features].astype(np.float32))

Y_val = val_y.values



test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

test = reduce_mem_usage(test)



pre_test = data_preparation(test, train_flag=False)

X_test = scaler.fit_transform(pre_test[im_features].astype(np.float32))



# Predict using DNN

pred = model.predict(X_test.astype(np.float32)) # .ravel()

print('Prediction values range: {0} ~ {1}'.format(pred.min(), pred.max()))



test['winPlacePerc'] = pred

submission = test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)



# Last check of submission

print('Head of submission: ')

display(submission.head())