# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install numpy==1.16.0

#Machine learning

!pip install catboost
#data preprocessing

import pandas as pd

#math operations

import numpy as np

#machine learning

from catboost import CatBoostRegressor, Pool

#data scaling

from sklearn.preprocessing import StandardScaler

#hyperparameter optimization

from sklearn.model_selection import GridSearchCV

#support vector machine model

from sklearn.svm import NuSVR, SVR

#kernel ridge model

from sklearn.kernel_ridge import KernelRidge

#data visualization

import matplotlib.pyplot as plt



# Improve progress bar display

import tqdm

import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm
#lets create a function to generate some statistical features based on the training data

def gen_features(X):

    strain = []

    strain.append(X.mean())

    strain.append(X.std())

    strain.append(X.min())

    strain.append(X.max())

    strain.append(X.kurtosis())

    strain.append(X.skew())

    strain.append(np.quantile(X,0.01))

    strain.append(np.quantile(X,0.05))

    strain.append(np.quantile(X,0.95))

    strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)
train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

X_train = pd.DataFrame()

y_train = pd.Series()

pbar = tqdm.tqdm(total=100)

for df in train:

    ch = gen_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))

    pbar.update(1)

pbar.close()
X_train.head()
train_pool = Pool(X_train, y_train)

m = CatBoostRegressor(iterations=50000, loss_function='MAE', boosting_type='Ordered')

m.fit(X_train, y_train, silent=True)

m.best_score_
# submission format

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame()



# prepare test data

for seg_id in tqdm.tqdm(submission.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    ch = gen_features(seg['acoustic_data'])

    X_test = X_test.append(ch, ignore_index=True)



X_test.head()
y_hat = m.predict(X_test)
# write submission file

submission['time_to_failure'] = y_hat

submission.to_csv('submission.csv')