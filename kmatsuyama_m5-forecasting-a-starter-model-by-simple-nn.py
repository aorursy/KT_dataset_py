# Libraries

import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')

%matplotlib inline

import copy

import datetime

import warnings

warnings.filterwarnings('ignore')



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder



import os

import re

import gc

import pickle  

import random

import keras



import numpy as np

import pandas as pd

import tensorflow as tf

#import tensorflow_hub as hub

import keras.backend as K



from keras.models import Model

from keras.layers import Dense, Input, Dropout, Lambda

from keras.optimizers import Adam

from keras.callbacks import Callback

from scipy.stats import spearmanr, rankdata

from os.path import join as path_join

from numpy.random import seed

from urllib.parse import urlparse

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import KFold

from sklearn.linear_model import MultiTaskElasticNet

from sklearn.linear_model import Ridge

import glob

from sklearn.model_selection import train_test_split



seed(42)

random.seed(42)
data_dict = {}

for i in glob.glob('../input/m5-forecasting-accuracy/*'):

    name = i.split('/')[-1].split('.')[0]

    if name != 'MTeamSpellings':

        data_dict[name] = pd.read_csv(i)

    else:

        data_dict[name] = pd.read_csv(i, encoding='cp1252')
data_dict.keys()
# check the subimission fotmat

data_dict['sample_submission']
# name the data respectively 



cal = data_dict['calendar']

stv = data_dict['sales_train_validation']

ss = data_dict['sample_submission']

sellp = data_dict['sell_prices']
stv_df = stv.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1).set_index('id').T

stv_df['d'] = stv_df.index
df = pd.merge(cal, stv_df, left_on='d', right_on='d', how='left')
def event_detector(x):

    if x == None:

        return 0

    else:

        return 1
#drop the date we won't use

drp = ['wm_yr_wk', 'weekday', 'year', 'd', 'event_type_1', 'event_type_2']



#we will use these columns

cols_x = ['wday', 'month', 'event_name_1', 'event_name_2','snap_CA', 'snap_TX', 'snap_WI']





# process events to binary

df = df.drop(drp, axis=1)

df['event_name_1'] = df['event_name_1'].apply(lambda x: event_detector(x))

df['event_name_2'] = df['event_name_2'].apply(lambda x: event_detector(x))
# separate validation data and evaluation data

ddf = df[(pd.to_datetime(df['date']) < '2016-04-25')&(pd.to_datetime(df['date']) >= '2015-06-19')].drop('date', axis=1)

valid_df = df[(pd.to_datetime(df['date']) >= '2016-04-25')&(pd.to_datetime(df['date']) < '2016-05-23')].drop('date', axis=1)

eval_df = df[pd.to_datetime(df['date']) >= '2016-05-23'].drop('date', axis=1)
X_ddf = ddf[cols_x]

y_ddf = ddf.drop(cols_x, axis=1)



X_train, X_test, y_train, y_test = train_test_split(

         X_ddf, y_ddf, test_size=0.33, random_state=42)
from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.callbacks import TensorBoard

import keras.backend as K

EarlyStopping = tf.keras.callbacks.EarlyStopping()





def rmse(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred -y_true)))



epochs=250

batch_size = 96

verbose = 1

validation_split = 0.2

input_dim = X_train.shape[1]

n_out = y_train.shape[1]



model = Sequential([

                Dense(512, input_shape=(input_dim,)),

                Activation('relu'),

                Dropout(0.2),

                Dense(512),

                Activation('relu'),

                Dropout(0.2),

                Dense(n_out),

                Activation('relu'),

                    ])



model.compile(loss='mse',

                 optimizer='adam',

                 metrics=['mse', rmse])

hist = model.fit(X_train, y_train,

                         batch_size = batch_size, epochs = epochs,

                         callbacks = [EarlyStopping],

                         verbose=verbose, validation_split=validation_split)



score = model.evaluate(X_test, y_test, verbose=verbose)

print("\nTest score:", score[0])
plt.clf()

plt.figsize=(15, 10)

loss = hist.history['loss']

val_loss = hist.history['val_loss']

epochs = range(1, len(loss) +1)



plt.plot(epochs, loss, 'bo', label = 'Training loss')

plt.plot(epochs, val_loss, 'b', label = 'Validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
import shap

df_train_normed = X_train

df_train_normed_summary = shap.kmeans(df_train_normed.values, 25)

# Instantiate an explainer with the model predictions and training data summary

explainer = shap.KernelExplainer(model.predict, df_train_normed_summary)

# Extract Shapley values from the explainer

shap_values = explainer.shap_values(df_train_normed.values)
X_valid = valid_df[cols_x]

X_eval = eval_df[cols_x]



# predict validation and evaluation respectively

pred_valid = pd.DataFrame(model.predict(X_valid), columns = ss[0:int(len(ss)/2)].set_index('id').T.columns)

pred_eval = pd.DataFrame(model.predict(X_eval), columns = ss[int(len(ss)/2):].set_index('id').T.columns)

ss_valid =  pred_valid.T

ss_eval = pred_eval.T



# concatenate val and eval

submission_df = pd.concat([ss_valid, ss_eval]).reset_index()

submission_df.columns = ss.columns



submission_df.head()
#check some of prediction

d_cols = [c for c in submission_df.columns if 'F' in c]

pred_example = submission_df.sample(10, random_state=2020).set_index('id')[d_cols].T



fig, axs = plt.subplots(5, 2, figsize=(15, 10))

axs = axs.flatten()

ax_idx = 0

for item in pred_example.columns:

    pred_example[item].plot(title=item,

                              ax=axs[ax_idx])

    ax_idx += 1

plt.tight_layout()

plt.show()
submission_df.to_csv('submission.csv', index=False)