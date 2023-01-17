from __future__ import print_function

#from hdfs import InsecureClient

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split

from sklearn.metrics import mean_squared_error

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

#import horovod.keras as hvd
# Horovod: initialize Horovod.

#hvd.init()



# Horovod: pin GPU to be used to process local rank (one GPU per process)

#config = tf.ConfigProto()

#config.gpu_options.allow_growth = True

#config.gpu_options.visible_device_list = str(hvd.local_rank())

#K.set_session(tf.Session(config=config))
#datasets with Reconstruction Accuracy

#new_merchant_saleslag3_pivot - 10%

#new_merchant_subsector_pivot_time - 40%

#new_merchant_state_pivot_time - 0%

#new_merchant_monthlag_pivot - 0%

#new_merchant_id_pivot_time - 72%

#new_card_id_state_pivot - 2%

#new_card_id_monthlag_pivot - 13%

#new_card_id_installment_pivot - 4%

#new_card_id_feat3_pivot - 55%

#new_card_id_feat2_pivot - 55%

#new_card_id_feat1_pivot - 84%

#new_card_id_cat3_pivot - 43%

#new_card_id_cat2_pivot - 59%

#new_card_id_cat1_pivot - 99%





#hist_merchant_subsector_pivot_time - 65%

#hist_merchant_state_pivot_time - 05%

#hist_merchant_saleslag3_pivot - 75%

#hist_merchant_monthlag_pivot_score - 95%

#hist_merchant_monthlag_pivot - 57%

#hist_merchant_id_pivot_time - 92%

#hist_merchant_state_pivot_time - 3%

#hist_merchant_saleslag3_pivot - 75%

#hist_merchant_monthlag_pivot_score - 94%

#hist_merchant_monthlag_pivot - 33%

#hist_month_lag_real_price - 11%
# load csv

df_train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv', index_col=['card_id'])

df_test = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv')
####find outliers

df_train['outliers'] = 0

df_train.loc[(df_train['target'] < -2) | (df_train['target'] > 2), 'outliers'] = 1

target = df_train['target']

df_train['outliers'].value_counts()
df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target']]

df_test_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]



target = df_train['target']



train_x, test_x = train_test_split(df_train[df_train_columns], test_size=0.2, random_state=420)

train_x = train_x[train_x.outliers == 0] #where normal transactions

train_x = train_x.drop(['outliers'], axis=1) #drop the class column



test_y = test_x['outliers'] #save the class column for the test set

test_x = test_x.drop(['outliers'], axis=1) #drop the class column



train_x = train_x.values #transform to ndarray

test_x = test_x.values
nb_epoch = 3

batch_size = 160

input_dim = train_x.shape[1] #num of columns, 30

encoding_dim = 32

hidden_dim = int(encoding_dim / 2) #i.e. 7

learning_rate = 0.001



input_layer = Input(shape=(input_dim, ))

encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)

encoder = Dense(hidden_dim, activation="relu")(encoder)

decoder = Dense(hidden_dim, activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.summary()





sgd = SGD(lr=0.001, momentum=0.02)

autoencoder.compile(metrics=['accuracy'],

                    loss='mean_squared_error',

                    optimizer='sgd')



cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",

                               save_best_only=True,

                               verbose=0)



tb = TensorBoard(log_dir='./logs',

                histogram_freq=0,

                write_graph=True,

                write_images=True)





# Horovod: adjust learning rate based on number of GPUs.

#opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0, nesterov=True)



# Horovod: add Horovod Distributed Optimizer.

#opt = hvd.DistributedOptimizer(opt)



history = autoencoder.fit(train_x, train_x,

                    epochs=nb_epoch,

                    batch_size=batch_size,

                    shuffle=True,

                    validation_data=(test_x, test_x),

                    verbose=1,

                    callbacks=[cp, tb]).history
test_x_predictions = autoencoder.predict(test_x)

mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,

                        'True_class': test_y})

error_df.describe()
#error_df.sort_values(by='Reconstruction_error', ascending=False)
from sklearn.metrics import confusion_matrix

LABELS = ["Normal","Unusual"]

threshold_fixed = 0.3524

pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, pred_y)



plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
mse=autoencoder.predict(df_train[df_test_columns])

df_train['anomoly'] = np.mean(np.power(df_train[df_test_columns] - mse, 2), axis=1)

df_train.anomoly.head(5)
mse=autoencoder.predict(df_test[df_test_columns])

df_test['anomoly'] = np.mean(np.power(df_test[df_test_columns] - mse, 2), axis=1)

df_test.anomoly.head(5)
ae_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]
param = {'boosting': 'goss',

         'objective': 'regression',

         'metric': 'rmse',

         'learning_rate': 0.01,

         'subsample': 0.9855232997390695,

         'max_depth': 7,

         'top_rate': 0.9064148448434349,

         'num_leaves': 63,

         'min_child_weight': 41.9612869171337,

         'other_rate': 0.0721768246018207,

         'reg_alpha': 9.677537745007898,

         'colsample_bytree': 0.5665320670155495,

         'min_split_gain': 9.820197773625843,

         'reg_lambda': 8.2532317400459,

         'min_data_in_leaf': 21,

         "verbosity": -1,

#         "device" : "gpu",

         "n_jobs" : -1}



folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=4590)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()

label_cols = ["outliers"]

y_split = df_train[label_cols].values





for fold_, (trn_idx, val_idx) in enumerate(folds.split(y_split[:,0], y_split[:,0])):

    print("fold {}".format(fold_))

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][ae_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][ae_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 5000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 50)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][ae_columns], num_iteration=clf.best_iteration)



    

    train_prediction = clf.predict(df_train[ae_columns] / folds.n_splits)

    predictions += clf.predict(df_test[ae_columns] / folds.n_splits)



np.sqrt(mean_squared_error(oof, target))
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission_new.csv", index=False)