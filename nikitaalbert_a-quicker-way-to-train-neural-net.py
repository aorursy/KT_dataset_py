# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

test = pd.read_csv('../input/liverpool-ion-switching/test.csv')
train.head()
test.head()
import matplotlib.pyplot as plt



plt.figure(figsize=(15,5))

plt.title('Train Set: Signal over Time')

plt.plot(train.time,train.signal)

plt.figure(figsize=(15,5))

plt.title('Train Set: Open Channels over Time')

plt.plot(train['time'],train['open_channels'])

plt.figure(figsize=(15,5))

plt.title('Test Set: Signal over Time')

plt.plot(test['time'],test['signal'])

plt.show()
def add_batch(data, batch_size):

    c = 'batch_' + str(batch_size)

    data[c] = 0

    ci = data.columns.get_loc(c)

    n = int(data.shape[0] / batch_size)

    print('Batch size:', batch_size, 'Column name:', c, 'Number of batches:', n)

    for i in range(0, n):

        data.iloc[i * batch_size: batch_size * (i + 1), ci] = i

        

for batch_size in [500000, 50000]:

    add_batch(train, batch_size)

    add_batch(test, batch_size)
batch = train[train['batch_50000']==10]

plt.figure(figsize=(15,5))

plt.plot(batch.time,batch.signal)

plt.figure(figsize=(15,5))

plt.plot(batch.time,batch.open_channels)

plt.show()
from scipy import signal



train_detrend = train.copy()

test_detrend = test.copy()
train_batch1_coords = [[50.000,60.0000]]       # start and stop times of trend in this format [[start, stop], [start stop]]

train_batch1 = train[train['batch_500000']==1] # isolating overall batch of where linear trend was found

baseline = abs(min(train_batch1['signal']))    # establishing baseline value, by taking absolute value of overall batch's minimal value.

#print(baseline)



for i in range(0,len(train_batch1_coords)):    # iterate through start/stop coordinates

    s_index = train.index.get_loc(train.index[train['time'] == train_batch1_coords[i][0]][0]) # get index of where start time is in training set

    #print(s_index)

    f_index = train.index.get_loc(train.index[train['time'] == train_batch1_coords[i][1]][0]) # get index of where stop time is in training set

    #print(f_index)

    lin_batch = train[s_index:f_index]                                                        # get slice of df from start to stop index

    detrend_lin = signal.detrend(lin_batch['signal'])                                         # detrend that slice's signal and store result

    offset = abs(min(detrend_lin)) - baseline                                                 # calculate the offset

    train_detrend.loc[train.index[s_index:f_index], 'signal'] = detrend_lin + (offset)        # replace signal in new df in the respective indices.

    
plt.figure(figsize=(15,5))

plt.plot(train_detrend.time,train_detrend.signal)

plt.show()
test_batch0 = test[test['batch_500000']==0]

test_batch0_coords = [[500.0001,510.0000],[510.0001,520.0000],[540.0000,550.0000]]



sb_index = test_batch0.index.get_loc(test_batch0.index[test_batch0['time'] == 530.0001][0]) # get start index of the "baseline" slice we identify

fb_index = test_batch0.index.get_loc(test_batch0.index[test_batch0['time'] == 539.9999][0]) # get stop index of the "baseline" slice we identify

baseline_batch = test_batch0[sb_index:fb_index]                                             

baseline = abs(min(baseline_batch['signal']))                                               # get minimal value of the "baselines" slice



for i in range(0,len(test_batch0_coords)):

    s_index = test.index.get_loc(test.index[test['time'] == test_batch0_coords[i][0]][0])

    #print(s_index)

    f_index = test.index.get_loc(test.index[test['time'] == test_batch0_coords[i][1]][0])

    #print(f_index)

    lin_batch = test[s_index:f_index]

    detrend_lin = signal.detrend(lin_batch['signal'])

    offset = abs(min(detrend_lin)) - baseline

    test_detrend.loc[test.index[s_index:f_index], 'signal'] = detrend_lin + (offset)

    
test_batch1 = test[test['batch_500000']==1]

test_batch1_coords = [[560.0000, 569.9999], [570.0000, 580.0000],[580.0001,590.0000]]



sb_index = test_batch1.index.get_loc(test_batch1.index[test_batch1['time'] == 590.0001][0])

fb_index = test_batch1.index.get_loc(test_batch1.index[test_batch1['time'] == 599.9999][0])

baseline_batch = test_batch1[sb_index:fb_index]

baseline = abs(min(baseline_batch['signal']))



for i in range(0,len(test_batch1_coords)):

    s_index = test.index.get_loc(test.index[test['time'] == test_batch1_coords[i][0]][0])

    #print(s_index)

    f_index = test.index.get_loc(test.index[test['time'] == test_batch1_coords[i][1]][0])

    #print(f_index)

    lin_batch = test[s_index:f_index]

    detrend_lin = signal.detrend(lin_batch['signal'])

    offset = abs(min(detrend_lin)) - baseline

    #print(offset)

    test_detrend.loc[test.index[s_index:f_index], 'signal'] = detrend_lin + offset
plt.figure(figsize=(15,5))

plt.plot(test_detrend.time,test_detrend.signal)

plt.show()
def remove_poly_trend(x, y):

    model = np.polyfit(x, y, 4)

    predicted = np.polyval(model, x)

    

    detrended = y - predicted 

    

    #print(detrended)

    return detrended
sb_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == 190.0000][0]) # start index of relatively flat slice of dataset

fb_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == 199.9999][0]) # stop index of relatively flat slice of dataset

baseline_batch = train_detrend[sb_index:fb_index]

baseline = abs(min(baseline_batch['signal']))                                                     #base line value



n_train = int(train_detrend.shape[0] / 500000)



for i in range(6, n_train):

    batch = train[train['batch_500000']==i]

    detrend_poly = remove_poly_trend(batch['time'], batch['signal'])

    offset = abs(min(detrend_poly)) - baseline

    min_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == min(batch['time'])][0])

    #print(min_index)

    max_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == max(batch['time'])][0])

    #print(max_index)

    train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'] = detrend_poly + offset

    #print(detrend_poly)
plt.figure(figsize=(15,5))

plt.plot(train_detrend.time,train_detrend.signal)

plt.show()
sb_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == 340.0000][0])

fb_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == 349.9999][0])

baseline_batch = train_detrend[sb_index:fb_index]

baseline = abs(min(baseline_batch['signal']))



sm_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == 350.0000][0])

fm_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == 359.9999][0])

misalign_batch = train_detrend[sm_index:fm_index]

misalign = abs(min(misalign_batch['signal']))



batch = train_detrend[train_detrend['batch_500000']==7]



offset = misalign - baseline

min_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == min(batch['time'])][0])

max_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == max(batch['time'])][0])

train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'] + offset

#print(detrend_poly)
plt.figure(figsize=(15,5))

plt.plot(train_detrend.time,train_detrend.signal)

plt.show()
sb_index = test_detrend.index.get_loc(test_detrend.index[test_detrend['time'] == 660.0000][0])

fb_index = test_detrend.index.get_loc(test_detrend.index[test_detrend['time'] == 664.9999][0])

baseline_batch = test_detrend[sb_index:fb_index]

baseline = abs(min(baseline_batch['signal']))



batch = test[test['batch_500000']==2]

detrend_poly = remove_poly_trend(batch['time'], batch['signal'])

offset = abs(min(detrend_poly)) - baseline

min_index = test_detrend.index.get_loc(test_detrend.index[test_detrend['time'] == min(batch['time'])][0])

#print(min_index)

max_index = test_detrend.index.get_loc(test_detrend.index[test_detrend['time'] == max(batch['time'])][0])

#print(max_index)

test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'] = detrend_poly + offset

#print(detrend_poly)

plt.figure(figsize=(15,5))

plt.plot(test_detrend.time,test_detrend.signal)

plt.show()
import gc



del train

del test



gc.collect()
window_sizes = [10, 50, 100]

window_sizes1 = [1000, 2500, 5000]
n_train = int(train_detrend.shape[0] / 500000)



for i in range(0, n_train):

    batch = train_detrend[train_detrend['batch_500000']==i]

    

    min_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == min(batch['time'])][0])

    max_index = train_detrend.index.get_loc(train_detrend.index[train_detrend['time'] == max(batch['time'])][0])



    for window in window_sizes:

        train_detrend.loc[train_detrend.index[min_index : max_index], "rolling_mean_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).mean()

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_std_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).std()

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_var_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).var()

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_min_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).min()

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_max_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).max()

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_kurtosis_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).kurt()

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_covariance_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).cov()

       

    for window in window_sizes1:

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_25_quartile_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.25)

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_50_quartile" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.5)

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_75_quartile_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.75)

        train_detrend.loc[train_detrend.index[min_index : max_index],"rolling_90_quartile_" + str(window)] = train_detrend.loc[train_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.9)

        

train_detrend.fillna(0, inplace=True)



train_detrend.head()
n_test = int(test_detrend.shape[0] / 500000)



for i in range(0, n_test):

    batch = test_detrend[test_detrend['batch_500000']==i]

    

    min_index = test_detrend.index.get_loc(test_detrend.index[test_detrend['time'] == min(batch['time'])][0])

    max_index = test_detrend.index.get_loc(test_detrend.index[test_detrend['time'] == max(batch['time'])][0])



    for window in window_sizes:

        test_detrend.loc[test_detrend.index[min_index : max_index], "rolling_mean_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).mean()

        test_detrend.loc[test_detrend.index[min_index : max_index],"rolling_std_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).std()

        test_detrend.loc[test_detrend.index[min_index : max_index],"rolling_var_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).var()

        test_detrend.loc[test_detrend.index[min_index : max_index],"rolling_min_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).min()

        test_detrend.loc[test_detrend.index[min_index : max_index],"rolling_max_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).max()

        test_detrend.loc[train_detrend.index[min_index : max_index],"rolling_kurtosis_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).kurt()

        test_detrend.loc[train_detrend.index[min_index : max_index],"rolling_covariance_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).cov()

        

        

    for window in window_sizes1:

        test_detrend.loc[train_detrend.index[min_index : max_index],"rolling_25_quartile_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.25)

        test_detrend.loc[train_detrend.index[min_index : max_index],"rolling_50_quartile_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.5)

        test_detrend.loc[train_detrend.index[min_index : max_index],"rolling_75_quartile_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.75)

        test_detrend.loc[train_detrend.index[min_index : max_index],"rolling_90_quartile_" + str(window)] = test_detrend.loc[test_detrend.index[min_index : max_index], 'signal'].rolling(window=window, min_periods=1).quantile(0.9)

        



test_detrend.fillna(0, inplace=True)



test_detrend.head()
train_detrend = train_detrend.drop(columns=['batch_50000', 'batch_500000'])

test_detrend = test_detrend.drop(columns=['batch_50000', 'batch_500000'])
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        if col != 'time':

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

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train_detrend=reduce_mem_usage(train_detrend)

test_detrend=reduce_mem_usage(test_detrend)
def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

            

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    # add shifts_1

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==24999].index:

        df['signal_shift_-1'][i] = np.nan

    

    df = df.drop(columns=['batch', 'batch_index', 'batch_slices', 'batch_slices2'])



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels'] and 'quartile' not in c1 and 'kurtosis' not in c1 and 'covariance' not in c1 ]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)

    gc.collect()

    return df



train_detrend = features(train_detrend)

test_detrend = features(test_detrend)
train_detrend=reduce_mem_usage(train_detrend)

test_detrend=reduce_mem_usage(test_detrend)
train_final = train_detrend.drop(columns=['time'])

test_final = test_detrend.drop(columns=['time'])



#train_final = train_detrend.copy()

#test_final = train_detrend.copy()
from sklearn.model_selection import train_test_split



seed_random = 316



y_train = train_final['open_channels'].copy()

x_train = train_final.drop(['open_channels'], axis=1)



x_test = test_final.copy()



x_t_train, x_t_val, y_t_train, y_t_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed_random)
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils



encoder = LabelEncoder()

encoder = encoder.fit(y_t_train)



y_train_econded = encoder.transform(y_t_train)

y_val_econded = encoder.transform(y_t_val)



y_train_dummy = np_utils.to_categorical(y_train_econded)

y_val_dummy = np_utils.to_categorical(y_val_econded)
from keras.models import Sequential

from keras.optimizers import Adam, Nadam

from keras.layers import Dense, Dropout



input_size = len(x_t_train.columns)



deep_model = Sequential()

deep_model.add(Dense(180, input_dim=input_size, kernel_initializer='glorot_uniform', activation='softplus'))

#deep_model.add(Dropout(0.2))

deep_model.add(Dense(80, kernel_initializer='glorot_uniform', activation='softplus'))

deep_model.add(Dense(30,kernel_initializer='glorot_uniform', activation='softplus'))

deep_model.add(Dense(24,kernel_initializer='glorot_uniform', activation='softplus'))

#deep_model.add(Dense(18,kernel_initializer='glorot_uniform', activation='softplus'))

deep_model.add(Dense(11, kernel_initializer='glorot_uniform', activation='softmax'))



deep_model.compile(loss='categorical_crossentropy', 

                   optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),

                   metrics=['accuracy'])
deep_model.fit(x_t_train, y_train_dummy, 

               epochs=50, 

               batch_size=2500,

               validation_data=(x_t_val, y_val_dummy))
deep_test_pred = deep_model.predict_classes(x_test)

deep_test_pred_decoded = encoder.inverse_transform(deep_test_pred)

deep_val_pred = deep_model.predict_classes(x_t_val)

deep_val_pred_decoded = encoder.inverse_transform(deep_val_pred)
import seaborn as sns

from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer 



# Showing Confusion Matrix

# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title):

    figsize=(14,14)

    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
plot_cm(y_t_val, deep_val_pred_decoded, 'Confusion matrix for the ANN predictions on validation set')

f1_score(y_t_val, deep_val_pred_decoded, average = 'macro')
submission = test_detrend.filter(['time'], axis=1)

submission.reset_index(drop=True, inplace=True)

submission['open_channels'] = deep_test_pred_decoded

submission.to_csv('submission.csv', index=False, float_format='%.4f')



submission.head()