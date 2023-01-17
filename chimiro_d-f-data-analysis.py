import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import requests

from urllib import parse



# Standard libraries

import os

import numpy as np

import random

import pandas as pd

import time

import re

import gc 

import os

import glob

from tqdm import tqdm 

import math



# Crawling

import threading

from multiprocessing import Process



# Visualization

import statsmodels as sm

import matplotlib.pyplot as plt

import seaborn as sns

import graphviz

import eli5

from eli5.sklearn import PermutationImportance



# import matplotlib.font_manager as fm

#plt.rc('font', family='Malgun Gothic')

#plt.rcParams['axes.unicode_minus'] = False



# Pre-processing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Correlation

from scipy.cluster import hierarchy as hc # dendrogram

from sklearn.tree import export_graphviz

import statsmodels as sm

from statsmodels.tsa.seasonal import STL



# Dataset

from sklearn.model_selection import TimeSeriesSplit



# Model

import tensorflow as tf

from keras import backend as K

from keras.losses import mse, binary_crossentropy

from keras import optimizers, regularizers

from keras.layers import Input, Dense, Lambda

from keras.models import Sequential, Model, load_model 

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



from tensorflow.keras.layers import Bidirectional, LSTM, Dense

from tensorflow.keras.activations import elu, relu



# Evaluate

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn import metrics               

from sklearn.metrics import roc_auc_score



# For notebook plotting

%matplotlib inline



import warnings                             

warnings.filterwarnings("ignore") 





pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option('display.max_columns', 3000)

pd.set_option('display.max_rows', 3000)

pd.set_option('display.max_colwidth', 3000)

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
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10, verbose=0):

    ''' Wrapper function to create a LearningRateScheduler with step decay schedule. '''

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose)
def inverse_transform(predictResult):

    return predictResult * train_std['meanPricePerHour'] + train_mean['meanPricePerHour']
'''

requests.adapters.DEFAULT_RETRIES = 5



def crawler(itemNm, apiKey):

    url = 'https://api.neople.co.kr/df/auction?itemName=' + itemNm + '&auctionNo:desc&limit=400&apikey=' + apiKey

    cnt = 9000

    while(True):

        try:

            resp = requests.get(url)

            contents = resp.json()

            df = pd.DataFrame(contents['rows'])

            df.to_pickle(os.path.join('./', itemNm, itemNm+'_'+str(cnt)+'.pkl'))

            #print('save', itemNm+'_'+str(cnt)+'.pkl')

            cnt += 1

            time.sleep(30)

        except:

            print("Connection refused by the server..")

            time.sleep(5)

            continue





if __name__ == '__main__':

    apiKey= "비공개"

    itemNms = ['시간의 결정', '[7월]신비한 시간의 인도석 상자', '고대 지혜의 잔해', '사도의 강림 풀세트 상자', 

               '사도의 강림 칭호 상자', '사도의 강림 오라 상자', '사도의 강림 스페셜 보물 상자', '사도 강림']

    procs = []



    for itemNm in itemNms:

        url = 'https://api.neople.co.kr/df/auction?itemName=' + itemNm + '&limit=400&apikey=' + apiKey

        proc = Process(target=crawler, args=(itemNm, apiKey))

        procs.append(proc)

        proc.start()

'''
#files = glob.glob('/kaggle/input/dataset-df/*')

#df = pd.concat((pd.read_pickle(file) for file in files), ignore_index=True)
dataset = pd.read_pickle('/kaggle/input/dfdataset/time_stone.pkl')

dataset = reduce_mem_usage(dataset)

dataset = dataset.reset_index()
dataset.head()
dataset.info()
dataset.describe()
fit_cols = ['auctionNo', 'regDate', 'expireDate', 'count', 'currentPrice', 'unitPrice', 'averagePrice']

dataset = dataset[fit_cols]

dataset.head(1)
dataset.isna().sum()
print('중복된 autionNo 수: ', dataset.duplicated(['auctionNo']).sum())



dataset.drop_duplicates(subset=['auctionNo'], keep='first', inplace=True)



print('중복된 autionNo 수: ', dataset.duplicated(['auctionNo']).sum())
dataset['regDate'] = pd.to_datetime(dataset['regDate'], errors='raise')

dataset.sort_values(by='regDate', inplace=True)

dataset = dataset[dataset['regDate'] >= '2020-08-04']



start = dataset['regDate'].to_list()[0]

startDate = str(start)[:11] + str(start.hour).zfill(2)

startDate = pd.to_datetime(startDate, errors='raise')



end = dataset['regDate'].to_list()[-1]

endDate = str(end)[:11] + str(end.hour+1).zfill(2)

endDate = pd.to_datetime(endDate, errors='raise')



display('경매장 아이템 정보 수집 기간: {0} ~ {1}'.format(startDate,endDate))



date_range = pd.date_range(start=startDate, end='2020-08-20 17', freq='H')

hourCut = pd.cut(dataset['regDate'], date_range, labels=date_range[:-1], right=False)

dataset['hourCut'] = hourCut



date_range = pd.date_range(start=startDate, end='2020-08-21', freq='D')

dateCut = pd.cut(dataset['regDate'], date_range, labels=date_range[:-1], right=False)

dataset['dayCut'] = dateCut





display(dataset.head(1))

display(dataset.tail(1))
dataset['meanPricePerHour'] = dataset.groupby('hourCut')['unitPrice'].transform(np.mean)

dataset['upperLimitPricePerHour'] = dataset.groupby('hourCut')['meanPricePerHour'].transform(lambda x: x*1.5)



dataset['meanPricePerDay'] = dataset.groupby('dayCut')['unitPrice'].transform(np.mean)

dataset['upperLimitPricePerDay'] = dataset.groupby('dayCut')['meanPricePerDay'].transform(lambda x: x*1.5)



display(dataset.head(1))

display(dataset.tail(1))
dataset['maxPricePerHour'] = dataset.groupby('hourCut')['unitPrice'].transform(np.max)

dataset['minPricePerHour'] = dataset.groupby('hourCut')['unitPrice'].transform(np.min)

dataset['maxPricePerDay'] = dataset.groupby('dayCut')['unitPrice'].transform(np.max)

dataset['minPricePerDay'] = dataset.groupby('dayCut')['unitPrice'].transform(np.min)



display(dataset.head(1))

display(dataset.tail(1))
dataset = dataset.set_index('regDate')



plt.figure(figsize=(10,7))

plt.xticks(rotation = 80)

snsPlot = sns.boxplot(x='dayCut', y='unitPrice', data=dataset)

snsPlot = sns.boxplot(x='dayCut', y='unitPrice', data=dataset)

snsPlot.set_xlabel("Date",fontsize=15)

snsPlot.set_ylabel("Price",fontsize=15)

sns.set(font_scale=1)

plt.grid()
plt.figure(figsize=(8,4))

plt.xticks(rotation = 80)

snsPlot = sns.countplot('dayCut', data=dataset)

snsPlot.set_xlabel("Date",fontsize=15)

snsPlot.set_ylabel("Count",fontsize=15)

plt.grid()
ax = dataset[['averagePrice', 'upperLimitPricePerDay', 'maxPricePerDay', 'minPricePerDay', 'meanPricePerDay']].plot(figsize=(15,7), kind='line')

ax.set(xlabel='Date', ylabel='Price')
ax = dataset[['averagePrice', 'upperLimitPricePerHour', 'maxPricePerHour', 'minPricePerHour', 'meanPricePerHour']].plot(figsize=(20,7), kind='line')

ax.set(xlabel='Date', ylabel='Price')
changed_averagePrice = dataset.groupby('averagePrice').head(1) 



plt.figure(figsize=(15,8))

plt.xticks(rotation = 45)

ax = sns.pointplot(x=changed_averagePrice.index, y='averagePrice', data=changed_averagePrice)

ax.set(xlabel='Date', ylabel='Price')

plt.grid()
dataset['countSumPerHour'] = dataset.groupby('hourCut')['count'].transform(np.sum)

dataset['countSumPerDay'] = dataset.groupby('dayCut')['count'].transform(np.sum)
ax = dataset[['countSumPerHour']].plot(figsize=(15,7), kind='line')

ax.set(xlabel='Date', ylabel='Count')
ax = dataset[['countSumPerDay']].plot(figsize=(15,7), kind='line')

ax.set(xlabel='Date', ylabel='Count')
plt.rc("figure", figsize=(16,8))

plt.rc("font", size=14)



data = dataset['count'].resample('H').sum().ffill()

res = STL(data).fit()

res.plot()

plt.show()
plt.rc("figure", figsize=(16,8))

plt.rc("font", size=14)



data = dataset['unitPrice'].resample('H').mean().ffill()

res = STL(data).fit()

res.plot()

plt.show()
plt.figure(figsize=(15,15))

col = dataset.columns[(dataset.dtypes != 'object') & (dataset.dtypes != 'category')]

col = col.drop(['auctionNo'])

df_corr = dataset[col].corr()

sns.heatmap(df_corr, annot=True, fmt='.2f', cbar=True, 

           xticklabels=dataset[col].columns.values, yticklabels=dataset[col].columns.values,

           linecolor='white', linewidths=0.1) 
split_date = '2020-08-17'

trainset = dataset.loc[dataset.index <= split_date].copy()

testset = dataset.loc[dataset.index > split_date].copy()



print('trainset: ', trainset.index[0], '~', trainset.index[-1])

print('testset: ', testset.index[0], '~', testset.index[-1])
target_cols = ['meanPricePerHour']

window = 3



# Exponential Moving Verage (EMA) 특징 추가

trainset = trainset.resample('H').mean().ffill()

trainset_ewm = trainset.ewm(window).mean()

trainset = pd.merge(trainset, trainset_ewm, suffixes=["","_ewm"], how='left', on='regDate')



testset = testset.resample('H').mean().ffill()

testset_ewm = testset.ewm(window).mean()

testset = pd.merge(testset, testset_ewm, suffixes=["","_ewm"], how='left', on='regDate')



# Nomalization or Standardization

train_mean = trainset.mean()

train_std = trainset.std()



trainset_ = (trainset - train_mean) / train_std

testset_ = (testset - train_mean) / train_std



# Split dataset

train_cols = trainset_.columns[trainset.columns.str.contains('PerHour')]

x_trainset = trainset_[train_cols]

y_trainset = trainset_[target_cols]



x_testset = testset_[train_cols]

y_testset = testset_[target_cols]



display(x_trainset.head(3))

display(y_trainset.head(3))
target_test = testset['meanPricePerHour'].resample('H').mean().ffill()

target_train = trainset['meanPricePerHour'].resample('H').mean().ffill()



plt.figure(figsize=(15,5))

target_train.plot()

target_test.plot()



#del target_train, target_test



gc.collect()
X_train = x_trainset.rolling(window).mean()[window-1:-1].values

Y_train = y_trainset[window:].values[:,0]

print('X_train shape', X_train.shape)

print('Y_train shape', Y_train.shape)



X_test = x_testset.rolling(window).mean()[window-1:-1].values

Y_test = y_testset[window:].values[:,0]

print('X_test shape', X_test.shape)

print('Y_test shape', Y_test.shape)
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from scipy.stats import uniform, randint

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split



import xgboost as xgb

from xgboost import plot_importance, plot_tree
def report_best_scores(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
xgb_model = xgb.XGBRegressor()



params = {

    "colsample_bytree": uniform(0.7, 0.3),

    "gamma": uniform(0, 0.5),

    "learning_rate": uniform(0.01, 1), # default 0.1 

    "max_depth": randint(2, 6), # default 3

    "n_estimators": randint(100, 500), # default 100

    "subsample": uniform(0.6, 0.4)}



search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=500, cv=3, verbose=1, n_jobs=6, return_train_score=True)



search.fit(X_train, Y_train)



report_best_scores(search.cv_results_, 1)
param_dist = {'eval_metric':["rmse"],

              'colsample_bytree': 0.768, 'gamma': 0.17, 'learning_rate': 0.39, 'max_depth': 3, 'n_estimators': 348, 'subsample': 0.821}
xgb_model = xgb.XGBRegressor(**param_dist)



scaler = StandardScaler()

tscv = TimeSeriesSplit(n_splits=5)

score = []



for tr_index, val_index in tscv.split(X_train):

    #X_tr, X_val = scaler.fit_transform(X_train[tr_index].astype(np.float32)), scaler.fit_transform(X_train[val_index].astype(np.float32))

    X_tr, X_val = X_train[tr_index], X_train[val_index]

    y_tr, y_val = Y_train[tr_index], Y_train[val_index]

    

    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

    y_pred = xgb_model.predict(X_val)

    score.append(mean_squared_error(y_val, y_pred, squared=False))
# make predictions

trainPredict = xgb_model.predict(X_train)

testPredict = xgb_model.predict(X_test)

# invert predictions

trainPredict = inverse_transform(trainPredict)

trainY = inverse_transform(Y_train)

testPredict = inverse_transform(testPredict)

testY = inverse_transform(Y_test)

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY, testPredict))

print('Test Score: %.2f RMSE' % (testScore))
predictXGB = pd.DataFrame(testPredict, columns=['PredictXGB_meanPricePerHour'])

predictXGB.index = testset['meanPricePerHour'].index[window:]



plt.figure(figsize=(20,10))

trainset['meanPricePerHour'].plot()

testset['meanPricePerHour'].plot()

predictXGB['PredictXGB_meanPricePerHour'].plot()



gc.collect()
feature_importance = pd.DataFrame(sorted(zip(xgb_model.feature_importances_, x_trainset.columns)), columns=['Value','Feature'])



plt.figure(figsize=(15, 5))

sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))

plt.title('XGBRegressor Features')

plt.show()
import scipy

# Dendrogram 



# Keep only significant features

to_keep = feature_importance.sort_values(by='Value', ascending=False).Feature



# Create a Dendrogram to view highly correlated features

corr = np.round(scipy.stats.spearmanr(x_trainset[to_keep]).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(14,5))

dendrogram = hc.dendrogram(z, labels=x_trainset[to_keep].columns, orientation='left', leaf_font_size=16)
perm = PermutationImportance(xgb_model, random_state=42).fit(X_test, Y_test)

eli5.show_weights(perm, feature_names=list(x_testset.columns))
def resample_dataset(dataset_x, dataset_y, window=2):

    result_x, result_y= [], []

    

    for i in range(len(dataset_x.index)):

        if len(dataset_x.index) <= i+window:

            break

            

        result_x.append(dataset_x.iloc[i:i+window].values)

        result_y.append(dataset_y.iloc[i+window].values[0])



    return np.asarray(result_x), np.asarray(result_y)
window = 3

X_train, Y_train = resample_dataset(x_trainset, y_trainset, window=window)

X_test, Y_test = resample_dataset(x_testset, y_testset, window=window)
X_train.shape
Y_train.shape
def get_model():

    lstm_dropout = 0.4

    

    hidden_layer = Bidirectional(LSTM(32, dropout=lstm_dropout, recurrent_dropout=lstm_dropout))(inputs)

    hidden_layer = Dense(32, activation='relu')(hidden_layer)

    hidden_layer = Dense(16, activation='relu')(hidden_layer)

    

    outputs = tf.keras.layers.Dense(1)(hidden_layer) 



    return outputs
np.random.seed(0)

tf.random.set_seed(0)



epochs = 1000

lr = 0.0001

batch_size = strategy.num_replicas_in_sync * 3

steps = len(X_train) // batch_size



optimizer = tf.keras.optimizers.Adam()

lr_sched = step_decay_schedule(initial_lr=lr, decay_factor=0.97, step_size=5, verbose=0)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)

callbacks_list = [lr_sched, early_stopping]



with strategy.scope():

    inputs = tf.keras.Input(shape=X_train[0].shape)

    

    outputs = get_model()

    lstm_model = tf.keras.Model(inputs=inputs, outputs=outputs)



    lstm_model.compile(optimizer=optimizer, loss='mae', metrics=[tf.keras.metrics.MeanAbsoluteError()]) # MeanSquaredError, MeanAbsoluteError



    history = lstm_model.fit(

            X_train, Y_train,

            shuffle=True,

            epochs=epochs,

            batch_size=batch_size,

            validation_data = (X_test, Y_test),

            steps_per_epoch=steps,

            callbacks=callbacks_list,

            verbose=2)
his = list(history.history)



plt.figure(figsize=(7,5))

plt.plot(history.history[his[1]])

plt.plot(history.history[his[3]])

plt.title('LSTM model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# make predictions

trainPredict = lstm_model.predict(X_train)[3:]

testPredict = lstm_model.predict(X_test)[3:]

# invert predictions

trainPredict = inverse_transform(trainPredict[:,0])

trainY = inverse_transform(Y_train)

testPredict = inverse_transform(testPredict[:,0])

testY = inverse_transform(Y_test)

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))

print('Train Score (LSTM): %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY, testPredict))

print('Test Score (LSTM): %.2f RMSE' % (testScore))
predictLSTM = pd.DataFrame(testPredict, columns=['PredictLSTM_meanPricePerHour'])

predictLSTM.index = testset['meanPricePerHour'].index[window:]



plt.figure(figsize=(20,10))

trainset['meanPricePerHour'].plot()

testset['meanPricePerHour'].plot()

predictXGB['PredictXGB_meanPricePerHour'].plot()

predictLSTM['PredictLSTM_meanPricePerHour'].plot()