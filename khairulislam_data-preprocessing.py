# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings; warnings.simplefilter('ignore')

from sklearn.preprocessing import StandardScaler

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv')

test = pd.read_csv('/kaggle/input/unsw-nb15/UNSW_NB15_testing-set.csv')

if train.shape[0]<100000:

    print("Train test sets are reversed. Fixing. ")

    train, test = test, train



# https://www.kaggle.com/khairulislam/unsw-nb15-feature-importance

drop_columns = ['attack_cat', 'id'] + ['response_body_len', 'spkts', 'ct_flw_http_mthd', 'trans_depth', 'dwin', 'ct_ftp_cmd', 'is_ftp_login']

for df in [train, test]:

    for col in drop_columns:

        if col in df.columns:

            print('Dropping '+col)

            df.drop([col], axis=1, inplace=True)
def feature_engineer(df):

    df.loc[~df['state'].isin(['FIN', 'INT', 'CON', 'REQ', 'RST']), 'state'] = 'others'

    df.loc[~df['service'].isin(['-', 'dns', 'http', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']), 'service'] = 'others'

    df.loc[df['proto'].isin(['igmp', 'icmp', 'rtp']), 'proto'] = 'igmp_icmp_rtp'

    df.loc[~df['proto'].isin(['tcp', 'udp', 'arp', 'ospf', 'igmp_icmp_rtp']), 'proto'] = 'others'

    return df



def get_cat_columns(train):

    categorical = []

    for col in train.columns:

        if train[col].dtype == 'object':

            categorical.append(col)

    return categorical
x_train, y_train = train.drop(['label'], axis=1), train['label']

x_test, y_test = test.drop(['label'], axis=1), test['label']



x_train, x_test = feature_engineer(x_train), feature_engineer(x_test)



categorical_columns = get_cat_columns(x_train)

non_categorical_columns = [x for x in x_train.columns if x not in categorical_columns]



scaler = StandardScaler()

x_train[non_categorical_columns] = scaler.fit_transform(x_train[non_categorical_columns])

x_test[non_categorical_columns] = scaler.transform(x_test[non_categorical_columns])





x_train = pd.get_dummies(x_train)

x_test = pd.get_dummies(x_test)

print("Column mismatch {0}, {1}".format(set(x_train.columns)- set(x_test.columns),  set(x_test.columns)- set(x_train.columns)))

features = list(set(x_train.columns) & set(x_test.columns))



print(f"Number of features {len(features)}")

x_train = x_train[features]

x_test = x_test[features]
x_train['label'] = y_train

x_test['label'] = y_test

x_train.to_csv('train.csv', index=False)

x_test.to_csv('test.csv', index=False)