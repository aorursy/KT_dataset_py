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

        

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler



from scipy import sparse



# Any results you write to the current directory are saved as output.
#Load data

train = pd.read_csv('../../kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train_sessions.csv')

test = pd.read_csv('../../kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/test_sessions.csv')

sample = pd.read_csv('../..//kaggle/input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/sample_submission.csv')
train.head()
# a little features engineering

def get_features(df):

    df['day_of_month'] = pd.to_datetime(df['time1']).dt.day

    df['first_session_time'] = (pd.to_datetime(df['time2']) -\

                                pd.to_datetime(df['time1'])).dt.seconds



    df['first_session_time'] = df['first_session_time'].fillna(0)

    df['day_of_week'] = pd.to_datetime(df['time1']).dt.dayofweek

    return df



# processing sites

def process_sites(df):

    sites = [x for x in df.columns if 'site' in x]

    df['sites_str'] = df[sites].astype('str').apply(' '.join)



    str_list = []

    for ind, row in df[sites].iterrows():

        str_list.append(' '.join(row.values.astype('str')))



    df['sites'] = str_list

    

    return df
%%time

train = get_features(train)

train = process_sites(train)



test = get_features(test)

test = process_sites(test)
# apply tfidf

tfidf = TfidfVectorizer(analyzer='word')

tfidf = tfidf.fit(train['sites'])



train_tf_feats = tfidf.transform(train['sites'])

test_tf_feats = tfidf.transform(test['sites'])
features = ['first_session_time', 'day_of_month', 'day_of_week']



train_features = sparse.hstack([sparse.csr_matrix(train[features]), train_tf_feats])

test_features = sparse.hstack([sparse.csr_matrix(test[features]), test_tf_feats])
x_train, x_valid, y_train, y_valid = train_test_split(

    train_features,

    train['target'],

    test_size=0.2

)
# initialize and fit model

lr = LogisticRegression(

    random_state=17,

    verbose=0,

    C=1,

    solver='liblinear',

    penalty='l2'

)



lr = lr.fit(x_train, y_train)
# predict and estimate results

predict = lr.predict_proba(x_valid)[:, 1]

roc_auc_score(y_valid, predict)
predicts = lr.predict_proba(test_features)[:, 1]
# make submission file

pd.DataFrame({

    "session_id": test['session_id'],

    'target': predicts

}).set_index('session_id').to_csv('baseline1.csv')