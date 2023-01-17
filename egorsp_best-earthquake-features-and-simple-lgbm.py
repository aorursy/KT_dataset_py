# from tqdm import tqdm_notebook as tqdm

import numpy as np

import pandas as pd



import os

import shutil

import warnings

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

from tsfresh.feature_extraction import feature_calculators

import librosa

import pywt

import wave

import random



from joblib import Parallel, delayed

import scipy as sp

import itertools

import gc



warnings.filterwarnings("ignore", category=FutureWarning) 
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



SEED = 2205

seed_everything(SEED)
train_curated = pd.read_csv("../input/train_curated.csv")

train_noisy = pd.read_csv("../input/train_noisy.csv")

train = pd.concat([train_curated, train_noisy], sort=True, ignore_index=True)



test = pd.read_csv("../input/sample_submission.csv")



LABELS = test.columns[1:].tolist()

num_classes = len(LABELS)
class Config(object):

    def __init__(self,

                 sampling_rate=16000, audio_duration=2, 

                 n_classes=num_classes,

                 use_mfcc=False, n_folds=10, learning_rate=0.0001, 

                 max_epochs=50, n_mfcc=20):

        self.sampling_rate = sampling_rate

        self.audio_duration = audio_duration

        self.n_classes = n_classes

        self.use_mfcc = use_mfcc

        self.n_mfcc = n_mfcc

        self.n_folds = n_folds

        self.learning_rate = learning_rate

        self.max_epochs = max_epochs



        self.audio_length = self.sampling_rate * self.audio_duration

        self.noise = np.random.normal(0, 0.5, self.audio_length)

        if self.use_mfcc:

            self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)

        else:

            self.dim = (self.audio_length, 1)



train.set_index("fname", inplace=True)

# test.set_index("fname", inplace=True)

train_curated.index = train_curated.fname

train_noisy.index = train_noisy.fname  

test.index = test.fname 
config = Config(sampling_rate=44100, audio_duration=3, n_folds=7, 

                learning_rate=0.001, use_mfcc=False)

def denoise_signal_simple(x, wavelet='sym5', level=1):

    coeff = pywt.wavedec(x, wavelet, mode="per")

    #univeral threshold

    uthresh = 10

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients

    return pywt.waverec(coeff, wavelet, mode='per')

def feature_gen(z,noise):

    X = pd.DataFrame(index=[0], dtype=np.float64)

    

    z = z + noise

    z = z - np.median(z)



    den_sample_simple = denoise_signal_simple(z)

    mfcc = librosa.feature.mfcc(z)

    mfcc_mean = mfcc.mean(axis=1)

    percentile_roll77_std_30 = np.percentile(pd.Series(z).rolling(77).std().dropna().values, 30)

    

    X['var_num_peaks_3_denoise_simple'] = feature_calculators.number_peaks(den_sample_simple, 3)

    X['var_percentile_roll77_std_30'] = percentile_roll77_std_30

    X['var_mfcc_mean13'] = mfcc_mean[13]

    X['var_mfcc_mean7'] = mfcc_mean[7]

    

    return X
def parse_sample(sample, config, data_dir):

    input_length = config.audio_length

    file_path = data_dir + sample.fname    

    data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

    # Random offset / Padding

    if len(data) > input_length:

        max_offset = len(data) - input_length

        offset = np.random.randint(max_offset)

        data = data[offset:(input_length+offset)]

    else:

        if input_length > len(data):

            max_offset = input_length - len(data)

            offset = np.random.randint(max_offset)

        else:

            offset = 0

        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

               

    return feature_gen(data, config.noise)
def sample_gen(df, config, data_dir):

    X = pd.DataFrame()

    result = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(parse_sample)(df.iloc[i], config, data_dir) for i in range(len(df))) 

    data = [r.values for r in result]

    data = np.vstack(data)

    X = pd.DataFrame(data,columns=result[0].columns)

    return X

   
X_train_curated = sample_gen(train_curated, config,'../input/train_curated/')

gc.collect()
X_train_noisy = sample_gen(train_noisy, config,'../input/train_noisy/')

gc.collect()
X_train = pd.concat([X_train_curated, X_train_noisy], sort=True, ignore_index=True)

X_test = sample_gen(test, config,'../input/test/')

gc.collect()
y_train = np.zeros((len(train), num_classes)).astype(int)

for i, row in enumerate(train['labels'].str.split(',')):

    for label in row:

        idx = LABELS.index(label)

        y_train[i, idx] = 1



print('Y_train',y_train.shape)
features = ['var_num_peaks_3_denoise_simple','var_percentile_roll77_std_30','var_mfcc_mean7',  'var_mfcc_mean13']

train_X = X_train[features].values

test_X = X_test[features].values
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)

# kf = list(kf.split(np.arange(len(X_train))))



oof = np.zeros((len(train_X),num_classes))

prediction = np.zeros((len(test_X),num_classes))



for fold_n, (train_index, valid_index) in enumerate(skf.split(train.index, train.labels)):

    print('Fold', fold_n)

    trn_data = lgb.Dataset(train_X[train_index], label=y_train[train_index,0])

    val_data = lgb.Dataset(train_X[valid_index], label=y_train[valid_index,0])

    

    params = {'num_leaves': 128,

      'min_data_in_leaf': 79,

      'num_class': num_classes,

      'objective':'multiclass',

      'max_depth': -1,

      'learning_rate': config.learning_rate,

      "boosting": "gbdt",

      'boost_from_average': True,

      "feature_fraction": 0.9,

      "bagging_freq": 7,

      "bagging_fraction": 0.8126672064208567,

      "bagging_seed": SEED,

#       "metric": 'mae',

      "verbosity": -1,

      'max_bin': 500,

      'reg_alpha': 0.1302650970728192,

      'reg_lambda': 0.3603427518866501,

      'seed': SEED,

      'n_jobs': 4

      }



    clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)



    oof[valid_index] += clf.predict(train_X[valid_index], num_iteration=clf.best_iteration)

    prediction += clf.predict(test_X, num_iteration=clf.best_iteration)



prediction /= config.n_folds



# print('\nMAE: ', mean_absolute_error(y_train, oof))
# Make a submission file

test_df = pd.read_csv('../input/sample_submission.csv')

print(test_df.head())

test_df[LABELS] = prediction

test_df.to_csv('submission.csv', index=False)

print(test_df.head())