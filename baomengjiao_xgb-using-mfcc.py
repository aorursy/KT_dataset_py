# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import librosa

#from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



from sklearn.metrics import accuracy_score

from scipy.stats import skew

SAMPLE_RATE = 45100



#from sklearn.model_selection import KFold, RepeatedKFold

from tqdm import tqdm, tqdm_pandas



tqdm.pandas()

import scipy

data_path = '../input/'

ss = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
#loading data

audio_train_files = os.listdir('../input/train_curated/')

audio_test_files = os.listdir('../input/test/')



train = pd.read_csv('../input/train_curated.csv')

submission = pd.read_csv('../input/sample_submission.csv')
#function from EDA kernel: https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis

def clean_filename(fname, string):   

    file_name = fname.split('/')[1]

    if file_name[:2] == '__':        

        file_name = string + file_name

    return file_name



#returns mfcc features with mean and standard deviation along time

def get_mfcc(name, path):

    b, _ = librosa.core.load(path + name, sr = SAMPLE_RATE)

    assert _ == SAMPLE_RATE

    try:

        ft1 = librosa.feature.mfcc(b, sr = SAMPLE_RATE, n_mfcc=20)

        ft2 = librosa.feature.zero_crossing_rate(b)[0]

        ft3 = librosa.feature.spectral_rolloff(b)[0]

        ft4 = librosa.feature.spectral_centroid(b)[0]

        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.min(ft1, axis = 1)))

        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.min(ft2)))

        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.min(ft3)))

        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.min(ft4)))

        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc)))

    except:

        print('bad file')

        return pd.Series([0]*115)
#preparing data

train_data = pd.DataFrame()

train_data['fname'] = train['fname']

test_data = pd.DataFrame()

test_data['fname'] = audio_test_files



train_data = train_data['fname'].apply(get_mfcc, path='../input/train_curated/')

print('done loading train mfcc')

test_data = test_data['fname'].apply(get_mfcc, path='../input/test/')

print('done loading test mfcc')



train_data['fname'] = train['fname']

test_data['fname'] = audio_test_files
train_data['label'] = train['labels'].apply(lambda x:x.split(',')[0])

test_data['label'] = np.zeros((len(audio_test_files)))
train_data.head()
def extract_features(files, path):

    features = {}

    cnt = 0

    for f in tqdm(files):

        features[f] = {}

        fs, data = scipy.io.wavfile.read(os.path.join(path, f))

        abs_data = np.abs(data)

        diff_data = np.diff(data)

        def calc_part_features(data, n=2, prefix=''):

            f_i = 1

            for i in range(0, len(data), len(data)//n):

                features[f]['{}mean_{}_{}'.format(prefix, f_i, n)] = np.mean(data[i:i + len(data)//n])

                features[f]['{}std_{}_{}'.format(prefix, f_i, n)] = np.std(data[i:i + len(data)//n])

                features[f]['{}min_{}_{}'.format(prefix, f_i, n)] = np.min(data[i:i + len(data)//n])

                features[f]['{}max_{}_{}'.format(prefix, f_i, n)] = np.max(data[i:i + len(data)//n])

        features[f]['len'] = len(data)

        if features[f]['len'] > 0:

            n = 1

            calc_part_features(data, n=n)

            calc_part_features(abs_data, n=n, prefix='abs_')

            calc_part_features(diff_data, n=n, prefix='diff_')



            n = 2

            calc_part_features(data, n=n)

            calc_part_features(abs_data, n=n, prefix='abs_')

            calc_part_features(diff_data, n=n, prefix='diff_')



            n = 3

            calc_part_features(data, n=n)

            calc_part_features(abs_data, n=n, prefix='abs_')

            calc_part_features(diff_data, n=n, prefix='diff_')

        cnt += 1

    features = pd.DataFrame(features).T.reset_index()

    features.rename(columns={'index': 'fname'}, inplace=True)

    return features



path = os.path.join(data_path, 'train_curated')

train_files = train.fname.values

train_features = extract_features(train_files, path)



path = os.path.join(data_path, 'test')

test_files = ss.fname.values

test_features = extract_features(test_files, path)
train_data = train_data.merge(train_features, on='fname', how='left')

test_data = test_data.merge(test_features, on='fname', how='left')

train_data.head()
#Functions from LightGBM baseline: https://www.kaggle.com/opanichev/lightgbm-baseline

# Construct features set

X = train_data.drop(['label', 'fname'], axis=1)

feature_names = list(X.columns)

X = X.values

labels = np.sort(np.unique(train_data.label.values))

num_class = len(labels)

c2i = {}

i2c = {}

for i, c in enumerate(labels):

    c2i[c] = i

    i2c[i] = c

y = np.array([c2i[x.split(',')[0]] for x in train_data.label.values])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=10, shuffle = True)

clf = LGBMClassifier(max_depth=5, learning_rate=0.05, n_estimators=1000,

                    n_jobs=-1, random_state=0, reg_alpha=0.2, 

                    colsample_bylevel=0.5, colsample_bytree=0.5)

clf.fit(X_train, y_train)

print(accuracy_score(clf.predict(X_val), y_val))

#more functions from LightGBM baseline: https://www.kaggle.com/opanichev/lightgbm-baseline
p = clf.predict_proba(test_data.drop(['label', 'fname'], axis =1).values)
p.shape
for i in range(len(labels)):

    submission[i2c[i]] = p[:, i]
submission.head()
submission.to_csv('submission.csv', index = False)
!ls