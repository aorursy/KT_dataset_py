# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd 



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



from os import listdir



from sklearn.model_selection import TimeSeriesSplit



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
train = pd.read_csv("../input/train.csv", nrows=10000000,

                    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
pd.options.display.precision = 15

train.head(10)
train.rename({"acoustic_data": "signal", "time_to_failure": "earthquake_time"}, axis="columns", inplace=True)
train.head(10)
for n in range(10):

    print(train.earthquake_time.values[n])
fig, ax = plt.subplots(2,1, figsize=(20,12))

ax[0].plot(train.index.values, train.earthquake_time.values, c="darkred")

ax[0].set_title("Earthquake remaining time of 10 Mio rows")

ax[0].set_xlabel("Index")

ax[0].set_ylabel("EarthQuake remaining time in ms");

ax[1].plot(train.index.values, train.signal.values, c="mediumseagreen")

ax[1].set_title("Signal of 10 Mio rows")

ax[1].set_xlabel("Index")

ax[1].set_ylabel("Acoustic Signal");
fig, ax = plt.subplots(3,1,figsize=(20,18))

ax[0].plot(train.index.values[0:50000], train.earthquake_time.values[0:50000], c="Blue")

ax[0].set_xlabel("Index")

ax[0].set_ylabel("Time to Earthquake")

ax[0].set_title("How does the second Earthquaketime pattern look like?")

ax[1].plot(train.index.values[0:49999], np.diff(train.earthquake_time.values[0:50000]))

ax[1].set_xlabel("Index")

ax[1].set_ylabel("Difference between Earthquaketimes")

ax[1].set_title("Are the jumps always the same?")

ax[2].plot(train.index.values[0:4000], train.earthquake_time.values[0:4000])

ax[2].set_xlabel("Index from 0 to 4000")

ax[2].set_ylabel("Earthquake Remaining time")

ax[2].set_title("How does the Earthquaketime changes within the first block?");
from os import listdir
test_path = "../input/test/"
test_files = listdir("../input/test/")

print(test_files[0:10])
len(test_files)
pd.set_option("display.precision", 4)

test1 = pd.read_csv('../input/test/seg_37669c.csv', dtype='int16')

print(test1.describe())

plt.figure(figsize=(10,5))

plt.title("Acoustic data distribution")

ax = sns.distplot(test1.acoustic_data, label='seg_37669c', kde=False)
import os

from random import shuffle
test_folder_files = os.listdir("../input/test/")
fig, axis = plt.subplots(5, 2, figsize=(12,14))

shuffle(test_folder_files)

xrow = xcol = 0

for f in test_folder_files[:10]:

    tmp = pd.read_csv('../input/test/{}'.format(f), dtype='int16')

    ax = sns.distplot(tmp.acoustic_data, label=f.replace('.csv',''), ax=axis[xrow][xcol], kde=False)

    if xcol == 0:

        xcol += 1

    else:

        xcol = 0

        xrow += 1
fig, axis = plt.subplots(5, 2, figsize=(12,14))

xrow = xcol = 0

for f in test_folder_files[:10]:

    tmp = pd.read_csv('../input/test/{}'.format(f), dtype='int16')

    ax = sns.lineplot(data=tmp.acoustic_data.values,

                      label=f.replace('.csv',''),

                      ax=axis[xrow][xcol],

                      color='orange')

    if xcol == 0:

        xcol += 1

    else:

        xcol = 0

        xrow += 1
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head(10)
len(sample_submission)
fig, ax = plt.subplots(5,1, figsize=(20,25))



for n in range(5):

    seg = pd.read_csv(test_path  + test_files[n])

    ax[n].plot(seg.acoustic_data.values, c="mediumseagreen")

    ax[n].set_xlabel("Index")

    ax[n].set_ylabel("Signal")

    ax[n].set_ylim([-300, 300])

    ax[n].set_title("Test {}".format(test_files[n]));
train.describe()
fig, ax = plt.subplots(1,2, figsize=(20,5))

sns.distplot(train.signal.values, ax=ax[0], color="Green", bins=100, kde=False)

ax[0].set_xlabel("Signal")

ax[0].set_ylabel("Density")

ax[0].set_title("Signal distribution")



low = train.signal.mean() - 3 * train.signal.std()

high = train.signal.mean() + 3 * train.signal.std() 

sns.distplot(train.loc[(train.signal >= low) & (train.signal <= high), "signal"].values,

             ax=ax[1],

             color="Green",

             bins=150, kde=False)

ax[1].set_xlabel("Signal")

ax[1].set_ylabel("Density")

ax[1].set_title("Signal distribution without peaks");
stepsize = np.diff(train.earthquake_time)

train = train.drop(train.index[len(train)-1])

train["stepsize"] = stepsize

train.head(10)
train.stepsize = train.stepsize.apply(lambda l: np.round(l, 10))
stepsize_counts = train.stepsize.value_counts()

stepsize_counts
from sklearn.model_selection import TimeSeriesSplit



cv = TimeSeriesSplit(n_splits=5)
window_sizes = [10, 50, 100, 500, 1000, 2000]

for window in window_sizes:

    train["rolling_mean_" + str(window)] = train.signal.rolling(window=window).mean()

    train["rolling_std_" + str(window)] = train.signal.rolling(window=window).std()
fig, ax = plt.subplots(len(window_sizes),1,figsize=(20,6*len(window_sizes)))



n = 0

for col in train.columns.values:

    if "rolling_" in col:

        if "mean" in col:

            mean_df = train.iloc[4435000:4445000][col]

            ax[n].plot(mean_df, label=col, color="green")

        if "std" in col:

            std = train.iloc[4435000:4445000][col].values

            ax[n].fill_between(mean_df.index.values,

                               mean_df.values-std, mean_df.values+std,

                               facecolor='lightblue',

                               alpha = 0.5, label=col)

            ax[n].legend()

            n+=1
train["rolling_q25"] = train.signal.rolling(window=50).quantile(0.25)

train["rolling_q75"] = train.signal.rolling(window=50).quantile(0.75)

train["rolling_q50"] = train.signal.rolling(window=50).quantile(0.5)

train["rolling_iqr"] = train.rolling_q75 - train.rolling_q25

train["rolling_min"] = train.signal.rolling(window=50).min()

train["rolling_max"] = train.signal.rolling(window=50).max()

train["rolling_skew"] = train.signal.rolling(window=50).skew()

train["rolling_kurt"] = train.signal.rolling(window=50).kurt()
peaks = train[train.signal.abs() > 500]

peaks.earthquake_time.describe()
plt.figure(figsize=(10,5))

plt.title("Cumulative distribution - time to failure with high signal")

ax = sns.distplot(peaks.earthquake_time, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
rolling_mean = []

rolling_std = []

last_time = []
rolling_mean = []

rolling_std = []

last_time = []

init_idx = 0

for _ in range(4194):  # 629M / 150k = 4194

    x = train.iloc[init_idx:init_idx + 150000]

   # last_time.append(x.earthquake_time.values[0])

    rolling_mean.append(x.signal.abs().mean())

    rolling_std.append(x.signal.abs().std())

    init_idx += 150000

    

rolling_mean = np.array(rolling_mean)

last_time = np.array(last_time)



# plot rolling mean

fig, ax1 = plt.subplots(figsize=(10, 5))

fig.suptitle('Mean for chunks with 150k samples of training data', fontsize=14)



ax2 = ax1.twinx()

ax1.set_xlabel('index')

ax1.set_ylabel('Acoustic data')

ax2.set_ylabel('Time to failure')

p1 = sns.lineplot(data=rolling_mean, ax=ax1, color='orange')

p2 = sns.lineplot(data=last_time, ax=ax2, color='gray')
from tqdm import tqdm
from numpy.random import seed

seed(639)

from tensorflow import set_random_seed

set_random_seed(5944)



float_data = pd.read_csv("../input/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}).values
def extract_features(z):

     return np.c_[z.mean(axis=1), 

                  np.transpose(np.percentile(np.abs(z), q=[0, 50, 75, 100], axis=1)),

                  z.std(axis=1)]
def create_X(x, last_index=None, n_steps=150, step_length=1000):

    if last_index == None:

        last_index=len(x)

       

    assert last_index - n_steps * step_length >= 0



    # Reshaping and approximate standardization with mean 5 and std 3.

    #temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3

    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1).astype(np.float32) - 5 ) / 3

    

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 

    # of the last 10 observations. 

    return np.c_[extract_features(temp),

                 extract_features(temp[:, -step_length // 10:]),

                 extract_features(temp[:, -step_length // 100:])]
# Query "create_X" to figure out the number of features

n_features = create_X(float_data[0:150000]).shape[1]

print("RNN is based on %i features"% n_features)

    

# The generator endlessly selects "batch_size" ending positions of sub-time series. For each ending position,

# the "time_to_failure" serves as target, while the features are created by the function "create_X".

def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):

    if max_index is None:

        max_index = len(data) - 1

     

    while True:

        # Pick indices of ending positions

        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

         

        # Initialize feature matrices and targets

        samples = np.zeros((batch_size, n_steps, n_features))

        targets = np.zeros(batch_size, )

        

        for j, row in enumerate(rows):

            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)

            targets[j] = data[row - 1, 1]

        yield samples, targets

        

batch_size = 32



# Position of second (of 16) earthquake. Used to have a clean split

# between train and validation

second_earthquake = 50085877

float_data[second_earthquake, 1]



# Initialize generators

# train_gen = generator(float_data, batch_size=batch_size) # Use this for better score

train_gen = generator(float_data, batch_size=batch_size, min_index=second_earthquake + 1)

valid_gen = generator(float_data, batch_size=batch_size, max_index=second_earthquake)
from tensorflow.contrib.rnn import *

from keras.models import Sequential

from keras.layers import Dense, CuDNNLSTM, LSTM 

from keras.optimizers import adam

from keras.callbacks import ModelCheckpoint

import tensorflow as tf
cb = [ModelCheckpoint("model.hdf5", save_best_only=True, period=3)]



model = Sequential()

model.add(LSTM(48, input_shape=(None, n_features)))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))



model.summary()
import keras

import keras.utils

from keras import utils as np_utils

from keras import optimizers

from keras.optimizers import sgd
model.compile(optimizer=sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss="hinge")



history = model.fit_generator(train_gen,

                              steps_per_epoch=150,

                              epochs=20,

                              verbose=2,

                              callbacks=cb,

                              validation_data=valid_gen,

                              validation_steps=300)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})



# Load each test data, create the feature matrix, get numeric prediction

for i, seg_id in enumerate(tqdm(submission.index)):

  #  print(i)

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    x = seg['acoustic_data'].values

    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))



submission.head()



# Save

submission.to_csv('submission_rnn1.csv')
from sklearn.preprocessing import StandardScaler

import gc
rows = 150000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['mean','std','99quat','50quat','25quat','1quat'])

y_train = pd.DataFrame(index = range(segments),dtype = np.float32,columns = ['time_to_failure'])
for segment in tqdm(range(segments)):

    x = train.iloc[segment*rows:segment*rows+rows]

    y = x['earthquake_time'].values[-1]

    x = x['signal'].values

    X_train.loc[segment,'mean'] = np.mean(x)

    X_train.loc[segment,'std']  = np.std(x)

    X_train.loc[segment,'99quat'] = np.quantile(x,0.99)

    X_train.loc[segment,'50quat'] = np.quantile(x,0.5)

    X_train.loc[segment,'25quat'] = np.quantile(x,0.25)

    X_train.loc[segment,'1quat'] =  np.quantile(x,0.01)

    y_train.loc[segment,'time_to_failure'] = y
scaler = StandardScaler()

X_scaler = scaler.fit_transform(X_train)
gc.collect()
model = Sequential()

model.add(Dense(32,input_shape = (6,),activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dense(1))

model.compile(loss = 'hinge',optimizer = 'adadelta')
model.fit(X_scaler,y_train.values.flatten(), epochs = 100)
sub_data = pd.read_csv('../input/sample_submission.csv',index_col = 'seg_id')
X_test = pd.DataFrame(columns = X_train.columns,dtype = np.float32,index = sub_data.index)
for seq in tqdm(X_test.index):

    test_data = pd.read_csv('../input/test/'+seq+'.csv')

    x = test_data['acoustic_data'].values

    X_test.loc[seq,'mean'] = np.mean(x)

    X_test.loc[seq,'std']  = np.std(x)

    X_test.loc[seq,'99quat'] = np.quantile(x,0.99)

    X_test.loc[seq,'50quat'] = np.quantile(x,0.5)

    X_test.loc[seq,'25quat'] = np.quantile(x,0.25)

    X_test.loc[seq,'1quat'] =  np.quantile(x,0.01)
X_test_scaler = scaler.transform(X_test)
pred = model.predict(X_test_scaler)
sub_data.head()
sub_data['time_to_failure'] = pred

sub_data['seg_id'] = sub_data.index
sub_data.head()
sub_data.to_csv('sub_earthquake_nn1.csv',index = False)
from multiprocessing import Pool

from catboost import CatBoostRegressor
X=X_train.copy()

y=y_train.copy()
#train_pool = Pool(X,y)

cat_model = CatBoostRegressor(

                               iterations=10000, 

                               learning_rate=0.03,

                               eval_metric='MAE',

                               verbose=1

                              )

cat_model.fit(X,y,silent=False)

y_pred_cat = cat_model.predict(X_test)
submission['time_to_failure'] = y_pred_cat

submission.to_csv('submission_cat.csv')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test.head()
X=X_train.copy()

y=y_train.copy()
from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())

y_pred_svm = svm.predict(X_train_scaled)
score = mean_absolute_error(y_train.values.flatten(), y_pred_svm)

print(f'Score: {score:0.3f}')
from sklearn.model_selection import KFold

import time

import lightgbm as lgb

from sklearn import svm, datasets

from sklearn.model_selection import GridSearchCV

#set_params(**params)



from sklearn.model_selection import ParameterGrid

from sklearn.svm import SVC

param_grid = ParameterGrid({'C': [.1, 1, 10], 'gamma':["auto", 0.01]})



for params in param_grid:

    svc_clf = SVC(**params)

    print (svc_clf)
folds = KFold(n_splits=20, shuffle=True, random_state=42)
%%time

y_pred_lgb = np.zeros(len(X_test_scaled))

for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(X))):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

    model = lgb.LGBMRegressor(**params, n_estimators = 22000, n_jobs = -1)

    model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

                    verbose=1000, early_stopping_rounds=200)

            

    y_pred_valid = model.predict(X_valid)

    y_pred_lgb += model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits