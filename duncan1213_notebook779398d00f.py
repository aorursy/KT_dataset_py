import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss
datadir='../input'

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),index_col='device_id')

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),index_col='device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),parse_dates=['timestamp'],index_col='event_id')

app_events = pd.read_csv(os.path.join(datadir,'app_events.csv'),

                        usecols=['event_id','app_id','is_active'],

                        dtype={'is_active':bool})

app_labels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
##添加列

gatrain['trainrow'] = np.arange(gatrain.shape[0])

gatest['testrow'] = np.arange(gatest.shape[0])
brandEncode = LabelEncoder().fit(phone['phone_brand'])

phone['brand'] = brandEncode.transform(phone['phone_brand'])

##print(phone['brand'].head(10))

gatrain['brand'] = phone['brand']

gatest['brand'] = phone['brand']



Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),

                      (gatrain['trainrow'],gatrain['brand'])))



Xts_brand = csr_matrix((np.ones(gatest.shape[0]),

                      (gatest['testrow'],gatest['brand'])))



#print(Xtr_brand.toarray()[1:3])

print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xts_brand.shape))
