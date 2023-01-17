#Import packages

from __future__ import print_function

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



import seaborn as sns

import gc



import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

from sklearn.metrics import mean_squared_error, roc_auc_score

from scipy.stats import norm, rankdata



import keras

from keras import regularizers

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,PReLU, LSTM

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import SGD

from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import tensorflow as tf
# reduce memory

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
#read in data

train = reduce_mem_usage(pd.read_csv('../input/train.csv'))

test = reduce_mem_usage(pd.read_csv('../input/test.csv'))



#Select features

features = [f for f in train if f not in ['ID_code','target']]



#Join for easier feature engineering

df_original = pd.concat([train, test],axis=0,sort=False)

df = df_original[features]

target = df_original['target'].values
#for feature in features:

#    df['mean_'+feature] = (train[feature].mean()-train[feature])

#    df['z_'+feature] = (train[feature] - train[feature].mean())/train[feature].std(ddof=0)

#    df['sq_'+feature] = (train[feature])**2

#    df['sqrt_'+feature] = np.abs(train[feature])**(1/2)

#    df['cp_'+feature] = pd.DataFrame(rankdata(train[feature]))

#    df['cnp_'+feature] = pd.DataFrame((norm.cdf(train[feature])))
#Guass transformation

from scipy.special import erfinv

trafo_columns = [c for c in df.columns if len(df[c].unique()) != 2]

for col in trafo_columns:

    values = sorted(set(df[col]))

    # Because erfinv(1) is inf, we shrink the range into (-0.9, 0.9)

    f = pd.Series(np.linspace(-0.9, 0.9, len(values)), index=values)

    f = np.sqrt(2) * erfinv(f)

    f -= f.mean()

    df[col] = df[col].map(f)
#from sklearn.decomposition import PCA

#pca = PCA(n_components=200)

#pca.fit(df[trafo_columns])

#df = pca.transform(df[trafo_columns])

#df = pd.DataFrame(df)
df['target'] = df_original.target.values

df = reduce_mem_usage(df)

train = df[df['target'].notnull()]

target = train['target']

test = df[df['target'].isnull()]

train.shape



trafo_columns = [c for c in train.columns if c not in ['target']]
from keras.activations import elu

from keras.layers import Input, Dense, Lambda

from keras.models import Model

from keras.objectives import binary_crossentropy

from keras.callbacks import LearningRateScheduler

from keras import backend as K

from imblearn.keras import balanced_batch_generator

from imblearn.under_sampling import NearMiss, RandomUnderSampler, CondensedNearestNeighbour, AllKNN, InstanceHardnessThreshold

from sklearn.model_selection import KFold

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE





nb_folds = 3

nb_epoch = 3

batch_size = 128

encoding_dim =1000

hidden_dim = int(encoding_dim * 2) #i.e. 7

sgd = SGD(lr=0.001, momentum=0.030, decay=0.76)

folds = StratifiedKFold(n_splits=nb_folds, shuffle=True, random_state=420)

#folds = KFold(n_splits = nb_folds, random_state = 338, shuffle = True)

train_auto = np.zeros(train[trafo_columns].shape)

test_auto = np.zeros(test[trafo_columns].shape)

predictions = np.zeros(len(train))

label_cols = ["target"]

y_split = train[label_cols].values



cp = ModelCheckpoint(filepath="autoencoder_0.h5",

                               save_best_only=True,

                               verbose=0)



tb = TensorBoard(log_dir='./logs',

                histogram_freq=0,

                write_graph=True,

                write_images=True)



es= EarlyStopping(monitor='val_acc',

                  min_delta=0,

                  patience=20,

                  verbose=1, mode='min')







for fold_, (trn_idx, val_idx) in enumerate(folds.split(y_split[:,0], y_split[:,0])):

    print("fold {}".format(fold_))

#for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):

#    print("fold {}".format(fold_))



    trn_data, trn_y = train[trafo_columns].iloc[trn_idx], train['target'].iloc[trn_idx]

    val_data, val_y = train[trafo_columns].iloc[val_idx], train['target'].iloc[val_idx]



#    classes=[]

#    for i in np.unique(trn_y):

#        classes.append(i)

#        print("Before OverSampling, counts of label " + str(i) + ": {}".format(sum(trn_y==i)))



 #   sm=SMOTE(random_state=2)

 #   trn_data, trn_y = sm.fit_sample(trn_data, trn_y.ravel())



#    print('After OverSampling, the shape of train_X: {}'.format(trn_data.shape))

#    print('After OverSampling, the shape of train_y: {} \n'.format(train_y.shape))



#    for eachClass in classes:

#        print("After OverSampling, counts of label " + str(eachClass) + ": {}".format(sum(train_y==eachClass)))



    input_dim = trn_data.shape[1] #num of columns, 30

    input_layer = Input(shape=(input_dim, ))

    

    # Q(z|X) -- encoder

    h_q = Dense(encoding_dim, activation='relu')(input_layer)

    mu = Dense(hidden_dim, activation='linear')(h_q)

    log_sigma = Dense(hidden_dim, activation='linear')(h_q)

    

    def sample_z(args):

        mu, log_sigma = args

        batch = K.shape(mu)[0]

        dim = K.int_shape(mu)[1]

        eps = K.random_normal(shape=(batch, dim))

        return mu + K.exp(0.5 * log_sigma) * eps



    # Sample z ~ Q(z|X)

    z = Lambda(sample_z)([mu, log_sigma])

    

    # P(X|z) -- decoder

    decoder_hidden = Dense(hidden_dim, activation='relu')

    decoder_out = Dense(input_dim, activation='softmax')

    h_p = decoder_hidden(z)

    outputs = decoder_out(h_p)

    

    # Overall VAE model, for reconstruction and training

    vae = Model(input_layer, outputs)

    

    # Encoder model, to encode input into latent variable

    # We use the mean as the output as it is the center point, the representative of the gaussian

    encoder = Model(input_layer, mu)



    # Generator model, generate new data given latent variable z

    d_in = Input(shape=(hidden_dim,))

    d_h = decoder_hidden(d_in)

    d_out = decoder_out(d_h)

    decoder = Model(d_in, d_out)

    

    def vae_loss(y_true, y_pred):

        """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

        # E[log P(X|z)]

        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)

        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian

        kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)

        return recon + kl

    

    vae.compile(optimizer='sgd', loss=vae_loss, metrics=['acc'])

    

    def add_noise(series, noise_level):

        return series * (1 + noise_level * np.random.randn(series.shape[1]))

    

    noisy_trn_data = add_noise(trn_data, 0.07)

#    noisy_val_data = add_noise(val_data, 0.07)

    



#    training_generator, steps_per_epoch = balanced_batch_generator(trn_data, train_y, sampler=RandomUnderSampler(),

#                                                batch_size=batch_size, random_state=42)



#    callback_history = vae.fit(training_generator,epochs=nb_epoch,

#                       validation_data=[val_data, val_data], 

#                       steps_per_epoch=steps_per_epoch, verbose=1,

#                       callbacks=[cp, tb, es])





    vae.fit(noisy_trn_data, trn_data, batch_size=batch_size, epochs=nb_epoch, validation_data=[val_data, val_data], 

                               callbacks=[cp, tb, es]).history

    train_auto[val_idx] += vae.predict(train.iloc[val_idx][trafo_columns], verbose=1)

    test_auto += vae.predict(test[trafo_columns], verbose=1)



    mse = vae.predict(train[trafo_columns] / folds.n_splits, verbose=1)

    predictions += np.mean(np.power(train[trafo_columns] - mse, 2), axis=1)



train_auto = pd.DataFrame(train_auto / folds.n_splits)

test_auto = pd.DataFrame(test_auto / folds.n_splits)

error_df = pd.DataFrame({'Reconstruction_error': predictions,

                        'True_class': train['target']})

error_df.describe()
####Edit thiS!!!!!####

from sklearn.metrics import confusion_matrix



LABELS = ["Normal","Outlier"]

threshold_fixed = 2.056936



pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(train['target'], pred_y)



plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()