import scipy
import numpy as np
import pandas as pd
import librosa

from sklearn.model_selection import train_test_split
from scipy.io import wavfile

import os
from tqdm import tqdm, tqdm_pandas, tqdm_notebook
print(os.listdir("../input"))
INIT_PATH = '../input/'
TRAIN_PATH = os.path.join(INIT_PATH, 'audio_train')
TEST_PATH = os.path.join(INIT_PATH, 'audio_test')
SAMPLE_RATE = 44100
RANDOM_STATE = 131
audio_test_files = os.listdir(TEST_PATH)
train_df = pd.read_csv(os.path.join(INIT_PATH, 'train.csv'))
sub_df = pd.read_csv(os.path.join(INIT_PATH, 'sample_submission.csv'))
def get_intersection_ts_feuters(ts, smr, n=10, fname=''):
    try:
        x = []
        ts = ts - np.mean(ts)
        ts = np.abs(ts)
        dts = np.diff(ts)
        length_ts = len(ts)
        step = length_ts//(2*n)
        for i in range(step, length_ts, step):
            fts = ts[i-step:i+step]
            fdts = dts[i-step:i+step]
            x.append(np.mean(fdts))
            x.append(np.std(fdts))
            x.append(np.min(fdts))
            x.append(np.max(fdts))
            x.append(np.median(fdts))
            x.append(scipy.stats.skew(fdts))
            x.append(np.mean(fdts))
            x.append(np.std(fdts))
            x.append(np.min(fdts))
            x.append(np.max(fdts))
            x.append(np.median(fdts))
            x.append(scipy.stats.skew(fdts))
        x.append(length_ts/smr)
        return x
    except:
        print('bad file {0}'.format(fname))
        return [0]*(2*n)

def get_intersection_mfcc_feuters(ts, smr, n=10, fname=''):
    try:
        x = []
        ts = ts - np.mean(ts)
        mfcc = librosa.feature.mfcc(ts, sr = smr, n_mfcc=30)
        delta_mfcc  = librosa.feature.delta(mfcc)
        length_mfcc = len(mfcc)
        step = length_mfcc//(2*n)
        for i in range(step, length_mfcc, step):
            fmfcc = mfcc[:][i-step:i+step]
            x.extend(np.mean(fmfcc,axis=1).tolist())
            x.extend(np.std(fmfcc,axis=1).tolist())
            x.extend(np.min(fmfcc,axis=1).tolist())
            x.extend(np.max(fmfcc,axis=1).tolist())
            x.extend(np.median(fmfcc,axis=1).tolist())
            x.extend(scipy.stats.skew(fmfcc,axis=1).tolist())
            fdmfcc = delta_mfcc[:][i-step:i+step]
            x.extend(np.mean(fdmfcc,axis=1).tolist())
            x.extend(np.std(fdmfcc,axis=1).tolist())
            x.extend(np.min(fdmfcc,axis=1).tolist())
            x.extend(np.max(fdmfcc,axis=1).tolist())
            x.extend(np.median(fdmfcc,axis=1).tolist())
            x.extend(scipy.stats.skew(fdmfcc,axis=1).tolist())
        return x
    except:
        print('bad file {0}'.format(fname))
        return [0]*(2*n*20+1)
def get_feuters(name, path):
    ts, smr = librosa.core.load(os.path.join(path, name), sr=SAMPLE_RATE)
    fts = get_intersection_ts_feuters(ts, smr, n=2, fname=name)
    fmfcc = get_intersection_mfcc_feuters(ts, smr, n=2, fname=name)
    fmfcc.extend(fts)
    return pd.Series(fmfcc)
train_data = pd.DataFrame()
train_data['fname'] = train_df['fname']
test_data = pd.DataFrame()
test_data['fname'] = os.listdir(TEST_PATH)

tqdm_pandas(tqdm, desc='extraction train feature')
train_data = train_data['fname'].progress_apply(get_feuters, path=TRAIN_PATH)
tqdm_pandas(tqdm, desc='extraction test feature')
test_data = test_data['fname'].progress_apply(get_feuters, path=TEST_PATH)
train_data['fname'] = train_df['fname']
test_data['fname'] = os.listdir(TEST_PATH)
train_data['label'] = train_df['label']
test_data['label'] = np.zeros((len(test_data)))
train_data.head()
import catboost as cb
c2i = {}
i2c = {}
labels = np.sort(np.unique(train_data.label.values))
for i, c in enumerate(labels):
    c2i[c] = i
    i2c[i] = c
y = np.array([c2i[x] for x in train_data.label.values])
length_col = train_data.shape[1]-2
print(length_col)
train_data = train_data.fillna(0)
X = train_data[[i for i in range(length_col)]].values
test_data = test_data.fillna(0)
X_test = test_data[[i for i in range(length_col)]].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=131)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train = sc.transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)
train_pool = cb.Pool(X_train, y_train)
valid_pool = cb.Pool(X_valid, y_valid)
test_pool = cb.Pool(X_test)
clf = cb.CatBoostClassifier(iterations=2000,
                            learning_rate=0.05,
                            l2_leaf_reg=15,
                            depth = 6,
                            leaf_estimation_iterations=3,
                            border_count=64,
                            loss_function='MultiClass',
                            custom_metric=['Accuracy'],
                            eval_metric='Accuracy',
                            random_seed=RANDOM_STATE,
                            classes_count=41
                           ).fit(train_pool, eval_set=valid_pool, verbose=False, plot=True)
pred = clf.predict(test_pool)
test_data['label'] = [i2c[int(p[0])] for p in pred]
sub = test_data[['fname', 'label']]
sub.to_csv('sub_catboost.csv', index=False)
sub.head()
