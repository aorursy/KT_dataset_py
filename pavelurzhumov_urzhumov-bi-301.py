# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pickle

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns
train_df = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv', index_col='session_id')

test_df = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv', index_col='session_id')
times = ['time%s' % i for i in range(1, 11)]

train_df[times] = train_df[times].apply(pd.to_datetime)

test_df[times] = test_df[times].apply(pd.to_datetime)
train_df = train_df.sort_values(by='time1')
train_df.head()
sites = ['site%s' % i for i in range(1, 11)]

train_df[sites] = train_df[sites].fillna(0).astype('int')

test_df[sites] = test_df[sites].fillna(0).astype('int')
with open(r"/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/site_dic.pkl", "rb") as input_file:

    site_dict = pickle.load(input_file)
sites_dict_df = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])

print(u'всего сайтов:', sites_dict_df.shape[0])

sites_dict_df.head()
y_train = train_df['target']
full_df = pd.concat([train_df.drop('target', axis=1), test_df])
idx_split = train_df.shape[0]
sites = ['site%d' % i for i in range(1, 11)]

sites
full_sites = full_df[sites]

full_sites.head()
sites_flatten = full_sites.values.flatten()
full_sites_sparse = csr_matrix(([1] * sites_flatten.shape[0], sites_flatten, range(0, sites_flatten.shape[0] + 10, 10)))[:, 1:]
full_sites_sparse.shape
X_train_sparse = full_sites_sparse[:idx_split]

X_test_sparse = full_sites_sparse[idx_split:]
X_train_sparse.shape, y_train.shape
X_test_sparse.shape
def get_auc_lr_valid(X, y, C=1.0, ratio = 0.9, seed=17):

    '''

    X, y – выборка

    ratio – в каком отношении поделить выборку

    C, seed – коэф-т регуляризации и random_state 

              логистической регрессии

    '''

    train_len = int(ratio * X.shape[0])

    X_train = X[:train_len, :]

    X_valid = X[train_len:, :]

    y_train = y[:train_len]

    y_valid = y[train_len:]

    

    logit = LogisticRegression(C=C, n_jobs=-1, random_state=seed)

    

    logit.fit(X_train, y_train)

    

    valid_pred = logit.predict_proba(X_valid)[:, 1]

    

    return roc_auc_score(y_valid, valid_pred)

    
def write_to_submission_file(predicted_labels, out_file,target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,index = np.arange(1, predicted_labels.shape[0] + 1), columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
new_feat_train = pd.DataFrame(index=train_df.index)

new_feat_test = pd.DataFrame(index=test_df.index)
new_feat_train['year_month'] = train_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)

new_feat_test['year_month'] = test_df['time1'].apply(lambda ts: 100 * ts.year + ts.month)
new_feat_train.head()
scaler = StandardScaler()

scaler.fit(new_feat_train['year_month'].values.reshape(-1, 1))





new_feat_train['year_month_scaled'] = scaler.transform(new_feat_train['year_month'].values.reshape(-1, 1))

new_feat_test['year_month_scaled'] = scaler.transform(new_feat_test['year_month'].values.reshape(-1, 1))
new_feat_train.head()
X_train_sparse_new = csr_matrix(hstack([X_train_sparse, 

                                        new_feat_train['year_month_scaled'].values.reshape(-1, 1)]))

X_test_sparse_new = csr_matrix(hstack([X_test_sparse, 

                                        new_feat_test['year_month_scaled'].values.reshape(-1, 1)]))
%%time

get_auc_lr_valid(X_train_sparse_new, y_train)
%%time

logit = LogisticRegression(n_jobs=-1, random_state=17)

logit.fit(X_train_sparse_new, y_train)
logit.predict_proba(X_test_sparse_new[:15, :])[: ,1]
test_pred = logit.predict_proba(X_test_sparse_new)[:, 1]
pd.Series(test_pred, index=range(1, test_pred.shape[0] + 1), name='target').to_csv('benchmark2.csv', header=True, index_label='session_id')
!head benchmark2.csv