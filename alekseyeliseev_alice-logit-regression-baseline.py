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
import numpy as np

import pandas as pd

import pickle
train_df = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv',

                      index_col='session_id')

test_df = pd.read_csv('/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv',

                     index_col='session_id')
train_df.head()
train_df.info()
# Меняем тип атрибутов site1, ..., site10 на целочисленный и заменяем отсутствующие значения нулями

sites = ['site%s' % i for i in range(1,11)]

train_df[sites] = train_df[sites].fillna(0).astype(int)

test_df[sites] = test_df[sites].fillna(0).astype(int)
train_df.head()
with open(r"/kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/site_dic.pkl", "rb") as input_file:

    site_dict = pickle.load(input_file)



sites_dict = pd.DataFrame(list(site_dict.keys()), index=list(site_dict.values()), columns=['site'])
sites_dict.head()
sites_dict.shape
train_df.shape, test_df.shape
train_df['target'].values
y_train = train_df['target'].values
idx = train_df.shape[0]

data = pd.concat([train_df, test_df], sort=False)
data[sites].to_csv('data_sessions_text.txt', 

                                 sep=' ', index=None, header=None)
!head data_sessions_text.txt
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(ngram_range=(1, 1), max_features=50000)

with open('data_sessions_text.txt') as inp_file:

    data = cv.fit_transform(inp_file)
X_train = data[:idx]

X_test = data[idx:]

print(X_train.shape, X_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

log_reg = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', max_iter=500)

X_train_log, X_valid_log, y_train_log, y_valid_log = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

log_reg.fit(X_train_log, y_train_log)
y_pred = log_reg.predict_proba(X_valid_log)

score = roc_auc_score(y_valid_log, y_pred[:,1])

score
log_reg.fit(X_train, y_train)
# Делаем предсказания

y_test = log_reg.predict_proba(X_test)
y_test[:5]
def write_to_submission_file(predicted_labels, out_file,

                             target='target', index_label="session_id"):

    predicted_df = pd.DataFrame(predicted_labels,

                                index = np.arange(1, predicted_labels.shape[0] + 1),

                                columns=[target])

    predicted_df.to_csv(out_file, index_label=index_label)
write_to_submission_file(y_test[:,1], 'baseline_1.csv')
param_grid = [

    {'penalty' : ['l1', 'l2'],

    'C' : [0.001,0.01,0.1,1,10,100,1000],

    'solver' : ['liblinear']}]
#from sklearn.model_selection import GridSearchCV

#grid_search = GridSearchCV(log_reg, scoring = 'roc_auc', param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)

#grid_search.fit(X_train, y_train)
#grid_search.best_params_
#y_test = grid_search.best_estimator_.predict_proba(X_test)
#write_to_submission_file(y_test[:,1], 'baseline_2.csv')