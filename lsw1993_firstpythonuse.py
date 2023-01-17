# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# %reset -f

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import gc

import os



import matplotlib.pyplot as plt

from numba import jit



from sklearn import preprocessing

import pywt



import tsfresh

from tsfresh import extract_features

from tsfresh.utilities.dataframe_functions import impute

from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters

import pickle

from pathlib import Path



from scipy.stats import kurtosis

from scipy.stats import skew

from scipy import stats

import scipy.signal as sg

import multiprocessing as mp

from scipy.signal import hann

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



import lightgbm as lgb

from sklearn.model_selection import KFold



import xgboost as xgb

import lightgbm as lgb

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV





from tqdm import tqdm

import warnings



warnings.filterwarnings("ignore")



print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
# REDUCE MEMORY USAGE - COMMON CODE SHARED ON KAGGLE KERNELS



# FAST AUC CALCULATION

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





class CustomUnpickler(pickle.Unpickler):



    def find_class(self, module, name):

        if name == 'Manager':

            from settings import Manager

            return Manager

        return super().find_class(module, name)
inp = '../input/safeornot/safety/'

pathlist = Path(inp + 'features').glob('**/*.csv')

chunks=[]



for path in pathlist:

    path_in_str = str(path)

    chunks.append(pd.read_csv(path_in_str))

    print('Done: ', str(path))

    

data1 = pd.concat(chunks, axis=0, ignore_index=True,sort=False)

print('Done: FEATURES')



del chunks

gc.collect()



pathlist2 = Path(inp + 'labels').glob('**/*.csv')

chunks=[]

for path in pathlist2:

    path_in_str = str(path)

    chunks.append(pd.read_csv(path_in_str))

    print('Done: ', str(path))

labels = pd.concat(chunks, axis=0, ignore_index=True,sort=False)

print('Done: LABELS')



del chunks
inp = '../input/safeornot/safety_test/'

pathlist = Path(inp + 'features').glob('**/*.csv')

chunks=[]



for path in pathlist:

    path_in_str = str(path)

    chunks.append(pd.read_csv(path_in_str))

    print('Done: ', str(path))

    

data2 = pd.concat(chunks, axis=0, ignore_index=True,sort=False)

print('Done: FEATURES')



del chunks

gc.collect()



pathlist2 = Path(inp + 'labels').glob('**/*.csv')

chunks=[]

for path in pathlist2:

    path_in_str = str(path)

    chunks.append(pd.read_csv(path_in_str))

    print('Done: ', str(path))

test_labels = pd.concat(chunks, axis=0, ignore_index=True,sort=False)

print('Done: LABELS')



del chunks
# ensure no duplicates

labels = labels.drop_duplicates()

print('Features Data bookingID Row #     :' , len(data1))

print('Features Data Unique bookingID #  :' , len(data1['bookingID'].unique()))



print('Label Data bookingID     #        :' , len(labels))

print('Label Data bookingID Row #        :' , len(labels.drop_duplicates().bookingID.unique()))



print('Some duplicated labels')

print(labels[labels.duplicated(['bookingID'], keep=False)].sort_values(by=['bookingID']).head())



to_rm = labels[labels.duplicated(['bookingID'], keep=False)].sort_values(by=['bookingID']).drop('label',1)

# to_rm.head()
# data 



# sort the data by bookingID and second

data =  pd.concat([data1, data2], axis=0)



del data1

del data2

gc.collect()



data = data.sort_values(by=['bookingID','second']).reset_index(drop=True)



# create diff to match - to flag not continuous part

data = data.assign(diff =  data['second'] - data['second'].shift(1) ).fillna(0)

data['Accuracy']=data.Accuracy.astype('float')

data['Bearing']=data.Bearing.astype('float')

data['acceleration_x']=data.acceleration_x.astype('float')

data['acceleration_y']=data.acceleration_y.astype('float')

data['acceleration_z']=data.acceleration_z.astype('float')

data['gyro_x']=data.gyro_x.astype('float')

data['gyro_y']=data.gyro_y.astype('float')

data['gyro_z']=data.gyro_z.astype('float')

data['Speed']=data.Speed.astype('float')
# REMOVE BOOKINGID WITH DUPLICATED LABELS

print("ORI ROW #                            : ", len(data))

data = data[~data.bookingID.isin(to_rm.bookingID)]



print("AFTER REMOVING DUPLICATED LABEL ROW #: ", len(data))



data = data.loc[data['diff'] <= 86_400]  #don't make sense to have same bookingID more than 1 day diff from previous rows



print("AFTER EXTREME DIFF REMOVAL      ROW #: ", len(data))



print("UNIQUE COUNT BOOKING ID:", len(data.bookingID.unique()))
data['acc'] = np.sqrt(np.power(data['acceleration_x'], 2) + np.power(data['acceleration_y'], 2) + np.power(data['acceleration_z'], 2))

data['gyr'] = np.sqrt(np.power(data['gyro_x'], 2) + np.power(data['gyro_y'], 2) + np.power(data['gyro_z'], 2))



# recreate data-set with average for every 2 secs or try to make it continuous 

# a portion of data is using 

data = data.assign(skip = data['second']//2)

# data = reduce_mem_usage(data)



data_gp = data.groupby(["bookingID","skip"])



data_max_sec = data_gp['second'].max().reset_index()

# data_max_sec = reduce_mem_usage(data_max_sec)

print(data_max_sec.head())



data_mean = data_gp['acc','gyr','Speed','Accuracy','Bearing'].mean().sort_values(by=['bookingID','skip']).reset_index()

# data_mean = reduce_mem_usage(data_mean)

print(data_mean.head())



label_y = labels[labels.bookingID.isin(data_mean.bookingID)].sort_values(by=['bookingID']).reset_index()

print(label_y.head())



gc.collect()

p_bearing = {'absolute_sum_of_changes': None,

'fft_coefficient': [{'coeff': 38, 'attr': 'abs'},

{'coeff': 70, 'attr': 'abs'}]}

        

p_speed = {'count_below_mean': None,

'fft_aggregated': [{'aggtype': 'centroid'}],

'fft_coefficient': [{'coeff': 40, 'attr': 'abs'},

{'coeff': 73, 'attr': 'abs'}],

'longest_strike_below_mean': None,

'number_crossing_m': [{'m': 1}],

'range_count': [{'max': 1, 'min': -1}]}

 

p_acc = {'agg_linear_trend': [{'f_agg': 'mean',

'chunk_len': 5,

'attr': 'stderr'},

{'f_agg': 'min', 'chunk_len': 50, 'attr': 'rvalue'}],

'change_quantiles': [{'f_agg': 'mean', 'isabs': True, 'qh': 1.0, 'ql': 0.6}],

'count_below_mean': None,

'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 1}],

'fft_coefficient': [{'coeff': 16, 'attr': 'abs'},

{'coeff': 49, 'attr': 'real'}]}

 

p_accuracy = {'count_below_mean': None}

 

p_gyr =  {'fft_aggregated': [{'aggtype': 'variance'}, {'aggtype': 'centroid'}],

  'number_peaks': [{'n': 1}],

  'length': None,

  'count_above_mean': None,

  'count_below_mean': None,

  'range_count': [{'max': 1000000000000.0, 'min': 0}]}
# Helper Functions

def mean_change_of_abs_change(x):

    return np.mean(np.diff(np.abs(np.diff(x))))

    

def _kurtosis(x):

    return kurtosis(x)



def CPT5(x):

    den = len(x)*np.exp(np.std(x))

    return sum(np.exp(x))/den



def skewness(x):

    return skew(x)



def SSC(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    xn_i1 = x[0:len(x)-2]  # xn-1

    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)

    return sum(ans[1:]) 



def wave_length(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1 

    return sum(abs(xn_i2-xn))

    

def norm_entropy(x):

    tresh = 3

    return sum(np.power(abs(x),tresh))



def SRAV(x):    

    SRA = sum(np.sqrt(abs(x)))

    return np.power(SRA/len(x),2)



def mean_abs(x):

    return sum(abs(x))/len(x)



def zero_crossing(x):

    x = np.array(x)

    x = np.append(x[-1], x)

    x = np.append(x,x[1])

    xn = x[1:len(x)-1]

    xn_i2 = x[2:len(x)]    # xn+1

    return sum(np.heaviside(-xn*xn_i2,0))





def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



def classic_sta_lta(x, length_sta, length_lta):

    sta = np.cumsum(x ** 2)

    # Convert to float

    sta = np.require(sta, dtype=np.float)

    # Copy for LTA

    lta = sta.copy()

    # Compute the STA and the LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta

    # Pad zeros

    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny

    return sta / lta

# Extra features



def feat_eng_acc(data):

    

    df = pd.DataFrame()



    

    for col in data.columns:

        if col in ['bookingID','skip']:

            continue

            

        df[col + '_abs_min'] = data.groupby(['bookingID'])[col].apply(lambda x: np.min(np.abs(x)))

        df[col + '_q95'] = data.groupby(['bookingID'])[col].quantile(0.95)



    return df



def feat_eng_accuracy(data):

    

    df = pd.DataFrame()



    

    for col in data.columns:

        if col in ['bookingID','skip']:

            continue

            

        df[col + '_mad'] = data.groupby(['bookingID'])[col].mad()

        df[col + '_median'] = data.groupby(['bookingID'])[col].median()

        df[col + '_q75'] = data.groupby(['bookingID'])[col].quantile(0.75)

        df[col + '_q95'] = data.groupby(['bookingID'])[col].quantile(0.95)

        df[col + '_range'] = data.groupby(['bookingID'])[col].max() - data.groupby(['bookingID'])[col].min()

        df[col + '_SSC'] = data.groupby(['bookingID'])[col].apply(SSC) 



    return df



def feat_eng_bearing(data):

    

    df = pd.DataFrame()



    

    for col in data.columns:

        if col in ['bookingID','skip']:

            continue

            

        df[col + '_iqr'] = data.groupby(['bookingID'])[col].quantile(0.75) - data.groupby(['bookingID'])[col].quantile(0.25)

        df[col + '_mad'] = data.groupby(['bookingID'])[col].mad()

        df[col + '_q95'] = data.groupby(['bookingID'])[col].quantile(0.95)

        df[col + '_std'] = data.groupby(['bookingID'])[col].std()



    return df



def feat_eng_gyr(data):

    

    df = pd.DataFrame()



    

    for col in data.columns:

        if col in ['bookingID','skip']:

            continue

            

        df[col + '_max'] = data.groupby(['bookingID'])[col].max()

        df[col + '_min'] = data.groupby(['bookingID'])[col].min()

        df[col + '_q75'] = data.groupby(['bookingID'])[col].quantile(0.75)

        df[col + '_q95'] = data.groupby(['bookingID'])[col].quantile(0.95)

        df[col + '_wave_length'] = data.groupby(['bookingID'])[col].apply(wave_length)



    return df



def feat_eng_speed(data):

    

    df = pd.DataFrame()



    

    for col in data.columns:

        if col in ['bookingID','skip']:

            continue

            

        df[col + '_max'] = data.groupby(['bookingID'])[col].max()

        df[col + '_mean'] = data.groupby(['bookingID'])[col].mean()

        df[col + '_mean_abs_chg'] = data.groupby(['bookingID'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        df[col + '_mean_change_of_abs_change'] = data.groupby('bookingID')[col].apply(mean_change_of_abs_change)

        df[col + '_median'] = data.groupby(['bookingID'])[col].median()

        df[col + '_norm_entropy'] = data.groupby(['bookingID'])[col].apply(norm_entropy)

        df[col + '_q25'] = data.groupby(['bookingID'])[col].quantile(0.25)

        df[col + '_q75'] = data.groupby(['bookingID'])[col].quantile(0.75)

        df[col + '_q95'] = data.groupby(['bookingID'])[col].quantile(0.95)

        df[col + '_range'] = data.groupby(['bookingID'])[col].max() - data.groupby(['bookingID'])[col].min()

        df[col + '_skew'] = data.groupby(['bookingID'])[col].skew()

        df[col + '_sla_tla_10_40_mean'] = data.groupby(['bookingID'])[col].apply(lambda x: classic_sta_lta(x, 10, 40).mean()) 

        df[col + '_sla_tla_10_50_mean'] = data.groupby(['bookingID'])[col].apply(lambda x: classic_sta_lta(x, 10, 50).mean()) 

        df[col + '_trend'] = data.groupby(['bookingID'])[col].apply(lambda x: add_trend_feature(x)) 

        df[col + '_wave_length'] = data.groupby(['bookingID'])[col].apply(wave_length)



    return df

extract_from = data_mean[['bookingID','skip','Accuracy']]

gp_accuracy  = feat_eng_accuracy(extract_from)

gp_accuracy['bookingID'] = gp_accuracy.index





ext_features = tsfresh.extract_features(extract_from,

                                          column_id='bookingID',

                                          column_sort='skip',

                                          n_jobs = 5, impute_function= impute,

                                          default_fc_parameters = p_accuracy)

ext_features['bookingID'] = ext_features.index



accuracy_var = pd.merge(ext_features, gp_accuracy, on = 'bookingID', how = 'left')



del extract_from

del ext_features

gc.collect()
extract_from = data_mean[['bookingID','skip','acc']]

gp_acc  = feat_eng_acc(extract_from)

gp_acc['bookingID'] = gp_acc.index



ext_features_2 = tsfresh.extract_features(extract_from,

                                          column_id='bookingID',

                                          column_sort='skip',

                                          n_jobs = 5, impute_function= impute,

                                          default_fc_parameters = p_acc)

ext_features_2['bookingID'] = ext_features_2.index

acc_var = pd.merge(ext_features_2, gp_acc, on = 'bookingID', how = 'left')



del extract_from

del ext_features_2

gc.collect()
extract_from = data_mean[['bookingID','skip','Bearing']]

gp_bearing  = feat_eng_bearing(extract_from)

gp_bearing['bookingID'] = gp_bearing.index



ext_features_3 = tsfresh.extract_features(extract_from,

                                          column_id='bookingID',

                                          column_sort='skip',

                                          n_jobs = 5, impute_function= impute,

                                          default_fc_parameters = p_bearing)

ext_features_3['bookingID'] = ext_features_3.index

bearing_var = pd.merge(ext_features_3, gp_bearing, on = 'bookingID', how = 'left')



del extract_from

del ext_features_3

gc.collect()
extract_from = data_mean[['bookingID','skip','Speed']]

gp_speed  = feat_eng_speed(extract_from)

gp_speed['bookingID'] = gp_speed.index



ext_features_4 = tsfresh.extract_features(extract_from,

                                          column_id='bookingID',

                                          column_sort='skip',

                                          n_jobs = 5, impute_function= impute,

                                          default_fc_parameters = p_speed)

ext_features_4['bookingID'] = ext_features_4.index

speed_var = pd.merge(ext_features_4, gp_speed, on = 'bookingID', how = 'left')



del extract_from

del ext_features_4

gc.collect()
extract_from = data_mean[['bookingID','skip','gyr']]

gp_gyr  = feat_eng_gyr(extract_from)

gp_gyr['bookingID'] = gp_gyr.index



ext_features_5 = tsfresh.extract_features(extract_from,

                                          column_id='bookingID',

                                          column_sort='skip',

                                          n_jobs = 5, impute_function= impute,

                                          default_fc_parameters = p_gyr)

ext_features_5['bookingID'] = ext_features_5.index

gyr_var = pd.merge(ext_features_5, gp_gyr, on = 'bookingID', how = 'left')



del extract_from

del ext_features_5

gc.collect()
data_all = pd.merge(accuracy_var, acc_var, on = 'bookingID', how = 'left')

data_all = pd.merge(data_all, bearing_var, on = 'bookingID', how = 'left')

data_all = pd.merge(data_all, speed_var, on = 'bookingID', how = 'left')

data_all = pd.merge(data_all, gyr_var, on = 'bookingID', how = 'left')

data_all = pd.merge(data_all, labels, on = 'bookingID', how = 'left')
len(data_all.columns)
data_all.to_pickle('data_all.pkl')
data_all.head()
train = data_all[data_all.label.notnull()]

test  = data_all[data_all.label.isnull()]



test = test.drop(['label'],axis = 1)

test = pd.merge(test, test_labels, on = 'bookingID', how = 'left')



features = train.drop(['label','bookingID'],axis = 1).columns



# train

X = train[features]

y = train.label



# test

X2 = test[features]

y2 = test.label

X2_id = pd.DataFrame(test['bookingID'])
num_round = 10000

kfold = 10

folds = StratifiedKFold(n_splits=kfold, shuffle=False, random_state=42)

oof = np.zeros(len(X))

predictions = np.zeros(len(X2))

importanceDF = pd.DataFrame()



# Parameter chosen from Bayesian Optimisation 

params =  {

    'learning_rate': 0.035, 

    'boosting': 'gbdt', 

    'objective': 'binary', 

    'metric': 'auc',

    'is_training_metric': True, 

    'seed': 999,   

    'bagging_fraction': 0.6473097703475733,

    'feature_fraction': 0.8559684085310095,

    'lambda_l1': 0.716766437045232,

    'lambda_l2': 2.8340067511487517,

    'max_depth': 5,

    'min_child_weight': 21.903773119545132,

    'min_split_gain': 0.027191005598358072,

    'num_leaves': 35}





for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values,y.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500, early_stopping_rounds = 50)

    

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),X.columns)), columns=['Value','Feature'])

    k = sum(feature_imp.Value)

    feature_imp['Value'] = feature_imp['Value']*100/k

    feature_imp.sort_values(by="Value", ascending=False)

    

    importanceDF = pd.concat([importanceDF, feature_imp], axis=0)

    

    oof[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)

    predictions += clf.predict(X2, num_iteration=clf.best_iteration) / folds.n_splits
feature_imp = pd.DataFrame(importanceDF.groupby(['Feature'])['Value'].apply(lambda x: x.mean()) )

feature_imp  = feature_imp.sort_values(by=["Value"], ascending = False)

feature_imp =  feature_imp.reset_index(level=['Feature'])
import seaborn as sns

sns.barplot(y="Feature", x="Value",data = feature_imp.head(15))

plt.show()
type(predictions)
test_prediction = X2_id.copy()

test_prediction['prediction'] = predictions

test_prediction = pd.merge(test_prediction, test_labels, on = 'bookingID', how = 'left')
fast_auc(test_prediction.label, test_prediction.prediction)