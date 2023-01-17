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
import os

def log(*args):

    os.system(f'echo \"{args}\"')

    print(*args)
! pip install --upgrade pip

! pip install tensorflow_datasets
import os

import pickle

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from joblib import dump, load

import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator,RegressorMixin

from sklearn.preprocessing import PolynomialFeatures

import multiprocessing as mp

import gc

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, f_regression

import tensorflow as tf

import lightgbm as lgb

import tensorflow_datasets.public_api as tfds
np.random.seed(12345)

tf.random.set_seed(12345)

encoders = pickle.load(open('/kaggle/input/data-preparation/encoders.pkl','rb'))

columns = pickle.load(open('/kaggle/input/data-preparation/columns.pkl','rb'))

max_day = 1941
# We will build one model for each store

print(columns[2]) # The store id is the third columns

group_index = 2   # We will group our data by this column

n_groups = len(encoders['store_id'].classes_)

keys = list(range(n_groups))

encoders['store_id'].classes_
log(columns)

log(columns[3]) # We retrieve our target column

target_col = 3
# This function is just a wrapper to a function that calls add_time_steps

def preprocess(lookback, delay, target_col=target_col ,lookback_step=1,test=False,val_days=0,return_key=False):

    # It takes as input a single time series data and applies some transformations to add time step

    def fn(inputs):

        # Remember _generate_examples of MyFullDataset we yielded a dictionary containing key and input

        values = inputs['input'] 

        key = inputs['key'] 

        return add_time_steps((key,values),

                              lookback,delay,

                              lookback_step=lookback_step,

                              target_col=target_col,

                              val_days=val_days,

                              test=test,

                              return_key=return_key)

    return fn

   

# This function takes an item with all it 1941 days salles, prices, and calendar data adds lookback 

# and generate the inputs and targets. 

def add_time_steps(inputs, lookback, delay,target_col=target_col,test=False,lookback_step=1,val_days=0,return_key=False):

    key,values = inputs

    max_index = values.shape[0] - 1

    min_index = 0

    y=None

    idx = tf.signal.frame(tf.range(min_index,max_index),lookback,lookback_step)

    if not test:

        idx = idx[tf.reduce_all(idx+delay <= max_index,axis=-1)]

        if val_days:

            val_idx = idx[-val_days:]

            val_y_idx = val_idx[...,-1]+delay

            y_val = tf.gather(values, val_y_idx)[...,target_col]

            X_val = tf.gather(values, val_idx)

            # remove val_days from training

            idx = idx[:-val_days]

        y_idx = idx[...,-1]+delay

        y = tf.gather(values, y_idx)[...,target_col]

    else:

        idx = idx[-delay:]

    X = tf.gather(values, idx)

    if not test and return_key:

        return (key,(X,y),(X_val,y_val))

    return ((X,y),(X_val,y_val)) if not test else (key,X)
# We need again this class

class MyFullDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('0.1.0')

    

    def _split_generators(self, dl_manager):

        return [

            tfds.core.SplitGenerator(

                    name=f'train',

                    gen_kwargs={

                    },

            )

        ]

    

    def _info(self):

        shape = (max_day,len(columns))

        return tfds.core.DatasetInfo(

            builder=self,

            description=(""),

            features=tfds.features.FeaturesDict({

                "input": tfds.features.Tensor(shape=shape,dtype=tf.float32),

                "key": tfds.features.Tensor(shape=(),dtype=tf.int32),

            }),

        )

    

   

    def _generate_examples(self,**args):

        # We no longer need this function because we already build our dataset

        pass
log("Download dataset")

ds_path = '/kaggle/input/data-preparation/'

builder = MyFullDataset(data_dir=ds_path)

dataset_ = builder.as_dataset()[f'train'].repeat()

builder.download_and_prepare()

log(builder.info)
total_num_examples = 30490

val_days = 28*2

lookback = 1

lookback_step = 1

delay = 28

gpu = False
def getData(key):

    preprocessor = preprocess(lookback, delay,lookback_step=lookback_step,val_days=val_days)

    fn_key_filter = lambda item: item['input'][0][group_index]==key

    dataset = dataset_.take(total_num_examples).filter(fn_key_filter)

    dataset = dataset.map(preprocessor,num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(3049)

    for (X_train, y_train),(X_val, y_val) in dataset:

        log("X_train : ",X_train.shape,'; y_train : ',y_train.shape)

        log("X_val : ",X_val.shape,'; y_train : ',y_val.shape)

        X_train, y_train = X_train.numpy(), y_train.numpy()

        X_val, y_val = X_val.numpy(), y_val.numpy()

    del dataset

    tf.keras.backend.clear_session()

    gc.collect()

    return key,(X_train, y_train),(X_val, y_val)

    

def train(key,nthread=1,save=True,verbose_eval=True,gpu=False):

    train_(getData(key),nthread=nthread,save=save,verbose_eval=verbose_eval,gpu=gpu)

    

def train_(input_,nthread=1,save=True,verbose_eval=False,gpu=False):

    key,(X_train, y_train),(X_val, y_val) = input_

    params = {

        'boosting_type': 'gbdt',

        'objective': 'poisson',# regression

        'metric':{'rmse',},

        'tree': 'feature_parallel',

        'num_leaves':1000,

        'learning_rate': 0.02,

        'feature_fraction': 0.5,

        'bagging_fraction': 0.8,

        "min_data_in_leaf":1000,

        'bagging_freq': 9,

        "bagging_seed" : 1234,

        'seed':1234,

        'verbosity': 1,

    }

    if gpu:

        params['device']='gpu'

        params['gpu_platform_id']=0

        params['gpu_device_id']=0

    else:

        params['nthread'] = nthread

        

    path = f'regressor-{key}.model'

    pipe = Pipeline([

        ('kbest',SelectKBest(f_regression, k=15)),

        #('poly',PolynomialFeatures(degree=2,include_bias=True,interaction_only=True,)),

        ('scaller',StandardScaler()),

    ])

     

    # Let correct the shape

    X_train = X_train.reshape(-1,X_train.shape[-1]*lookback)

    y_train = y_train.reshape((-1,))

    idx = np.arange(X_train.shape[0])

    # We shuffle

    np.random.shuffle(idx)

    X_train,y_train = X_train[idx],y_train[idx]

    log("X_train : ",X_train.shape)

    X_train = pipe.fit_transform(X_train,y_train)

    log("X_train : ",X_train.shape)

    #log("Scores : ",pipe.named_steps['kbest'].scores_)

    

    # No need to shuffle validation data

    X_val = X_val.reshape(-1,X_val.shape[-1]*lookback)

    y_val = y_val.reshape((-1,))

    X_val = pipe.transform(X_val)

    log("X_val : ",X_val.shape)

    

    # Training

    train_data = lgb.Dataset(X_train, label=y_train)

    test_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, 

                      train_data,

                      valid_sets=[test_data],

                      num_boost_round=3000,

                      early_stopping_rounds=100,

                      verbose_eval=verbose_eval)

    

    mse = mean_squared_error(y_train,model.predict(X_train,num_iteration_predict=model.best_iteration))

    val_mse = mean_squared_error(y_val,model.predict(X_val,num_iteration_predict=model.best_iteration))

    log(f'Key {key};  MSE: {mse}, VAL_MSE: {val_mse}')

    if not save:

        return (key,model)

    else:

        with open(path,'wb') as f:

            dump({

                'model':model,

                'num_iteration_predict':model.best_iteration,

                'pipe':pipe

            },f,compress=9)

            

        del model

    gc.collect()
for key in keys:

    train(key,nthread=-1,verbose_eval=True,gpu=gpu)

    tf.keras.backend.clear_session()

    gc.collect()
gc.collect()
log('Loading models')

models = {}

for key in keys:

    path = f'regressor-{key}.model'

    models[key] = load(path)
# Evaluate our validation data with the leaderboard

val_days = 28

df_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv',index_col=0)

df_val.iloc[total_num_examples:] = 0 # set evaluation to 0

    

preprocessor = preprocess(lookback, delay,lookback_step=lookback_step,val_days=val_days,return_key=True)

dataset = dataset_.take(total_num_examples)

dataset = dataset.map(preprocessor,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(1000)

c = 1  

for idx,_,(X_val, y_val) in dataset:

    idx = idx.numpy()

    X_val, y_val = X_val.numpy(), y_val.numpy()

    X_val = X_val.reshape(-1,X_val.shape[-1]*lookback)

    

    group_key = X_val[0][group_index]

    pipe = models[group_key]['pipe']

    model = models[group_key]['model']

    num_iteration_predict = models[group_key]['num_iteration_predict']

    X_val = pipe.transform(X_val)

    df_val.iloc[idx] = model.predict(X_val,num_iteration_predict=num_iteration_predict).flatten()

    print(f'{c}\r',end='')

    c += 1

df_val.to_csv('validation_submission.csv')

del dataset

gc.collect()
def generate_submission(models,group_index, lookback, delay, lookback_step,xgb=False,out_path='submission.csv'):

    df_sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv',index_col=0)

    df_sub.iloc[:total_num_examples,:] = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv',index_col=0).iloc[:,-28:].values

    df_sub.iloc[total_num_examples:] = 0

    

    preprocessor = preprocess(lookback, delay,lookback_step=lookback_step,val_days=val_days,test=True)

    dataset = dataset_.take(total_num_examples)

    dataset = dataset.map(preprocessor,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(1000)

    

    c = 1

    for idx,X in dataset:

        idx = idx.numpy()

        X = X.numpy()

        X = X.reshape(-1,X.shape[-1]*lookback)

        group_key = X[0][group_index]

        pipe = models[group_key]['pipe']

        model = models[group_key]['model']

        num_iteration_predict = models[group_key]['num_iteration_predict']

        X = pipe.transform(X)

        df_sub.iloc[total_num_examples+idx] = model.predict(X,num_iteration_predict=num_iteration_predict).flatten()

        print(f'{c}\r',end='')

        c += 1

    df_sub.to_csv(out_path)
log('Generate submission')

generate_submission(models,group_index, lookback, delay,lookback_step)