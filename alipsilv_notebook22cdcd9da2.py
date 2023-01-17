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



datadir = '../input/'

gatrain = pd.read_csv(datadir+'gender_age_train.csv'

                      #,nrows=1000

                      ,index_col='device_id')



gatest = pd.read_csv(datadir+'gender_age_test.csv'

                      #,nrows=1000

                      ,index_col = 'device_id')



phone = pd.read_csv(datadir+'phone_brand_device_model.csv'

                       #,nrows=1000

                       )



# Get rid of duplicate device ids in phone

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')



events = pd.read_csv(datadir+'events.csv'

                     ,parse_dates=['timestamp']

                     ,index_col='event_id'

                     #,nrows=1000

                     )

appevents = pd.read_csv(datadir+'app_events.csv'

                     #,nrows=1000

                     , usecols=['event_id','app_id','is_active']

                     , dtype={'is_active':bool})

applabels = pd.read_csv(datadir+'app_labels.csv'

                    #,nrows=1000

                    )
gatrain['trainrow'] = np.arange(gatrain.shape[0])

gatest['testrow'] = np.arange(gatest.shape[0])
brandencoder = LabelEncoder().fit(phone.phone_brand)

phone['brand'] = brandencoder.transform(phone['phone_brand'])

gatrain['brand'] = phone['brand']

gatest['brand'] = phone['brand']

Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.brand)))

Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.brand)))

print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))


m = phone.phone_brand.str.cat(phone.device_model)

modelencoder = LabelEncoder().fit(m)

phone['model'] = modelencoder.transform(m)

gatrain['model'] = phone['model']

gatest['model'] = phone['model']

Xtr_model = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.model)))

Xte_model = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.model)))

print('Model features: train shape {}, test shape {}'.format(Xtr_model.shape, Xte_model.shape))

appencoder = LabelEncoder().fit(appevents.app_id)

appevents['app'] = appencoder.transform(appevents.app_id)

napps = len(appencoder.classes_)

deviceapps = (appevents.merge(events[['device_id']], how='left',left_on='event_id',right_index=True)

                       .groupby(['device_id','app'])['app'].agg(['size'])

                       .merge(gatrain[['trainrow']], how='left', left_index=True, right_index=True)

                       .merge(gatest[['testrow']], how='left', left_index=True, right_index=True)

                       .reset_index())

deviceapps.head()



deviceapps.to_csv('C:/3-Kaggle/talkingdata/simples.csv', index=False, float_format='%.3f')