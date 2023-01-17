# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os #joining filepath

from sklearn.preprocessing import LabelEncoder #one-hot encoding

from scipy.sparse import csr_matrix, hstack

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def logloss_err(y,p):

    N=len(p)

    err=-1/N*np.sum(y*np.log(p))

    return err
p=np.array([[0.2,0.8],[0.7,0.3]])

y=np.array([[0,1],[1,0]])

logloss_err(y,p)
datadir = '../input'

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),

                      index_col='device_id')

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),

                     index_col = 'device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),

                     parse_dates=['timestamp'], index_col='event_id')

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 

                        usecols=['event_id','app_id','is_active'],

                        dtype={'is_active':bool})

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
gatrain['trainrow'] = np.arange(gatrain.shape[0])

brandencoder = LabelEncoder().fit(phone.phone_brand)

phone['brand'] = brandencoder.transform(phone['phone_brand'])

gatrain['brand'] = phone['brand']

gatest['brand'] = phone['brand']

Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]), 

                       (gatrain.trainrow, gatrain.brand)))

Xte_brand = csr_matrix((np.ones(gatest.shape[0]), 

                       (gatest.testrow, gatest.brand)))

print('Brand features: train shape {}, test shape {}'.format(Xtr_brand.shape, Xte_brand.shape))