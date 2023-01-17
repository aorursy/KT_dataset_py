# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import xgboost as xgb

import gc
train = pd.read_csv('../input/train_2016.csv')

properties = pd.read_csv('../input/properties_2016.csv')
sample = pd.read_csv('../input/sample_submission.csv')
properties.airconditioningtypeid = properties.airconditioningtypeid.fillna(0).astype(int)

properties.architecturalstyletypeid = properties.architecturalstyletypeid.fillna(0).astype(int)

properties.buildingclasstypeid = properties.buildingclasstypeid.fillna(0).astype(int)

properties.buildingqualitytypeid = properties.buildingqualitytypeid.fillna(0).astype(int)

properties.decktypeid = properties.decktypeid.fillna(0).astype(int)

properties.heatingorsystemtypeid = properties.heatingorsystemtypeid.fillna(0).astype(int)



properties.bathroomcnt = properties.bathroomcnt.fillna(0).astype(float)

properties.bedroomcnt = properties.bedroomcnt.fillna(0).astype(int)

properties.calculatedbathnbr = properties.calculatedbathnbr.fillna(0).astype(float)

properties.fireplacecnt = properties.fireplacecnt.fillna(0).astype(int)

properties.fullbathcnt = properties.fullbathcnt.fillna(0).astype(int)

properties.garagecarcnt = properties.garagecarcnt.fillna(0).astype(int)

properties.poolcnt = properties.poolcnt.fillna(0).astype(int)

properties.fips = properties.fips.fillna(0).astype(int)

properties.poolsizesum = properties.poolsizesum.fillna(0).astype(int)



properties.hashottuborspa = properties.hashottuborspa.fillna(False).astype(int)
sample.head()
print('Binding to float32')



for c, dtype in zip(properties.columns, properties.dtypes):

	if dtype == np.float64:

		properties[c] = properties[c].astype(np.float32)
print('Creating training set ...')



df_train = train.merge(properties, how='left', on='parcelid')



x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

y_train = df_train['logerror'].values

print(x_train.shape, y_train.shape)



train_columns = x_train.columns



for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)



del df_train; gc.collect()

split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



del x_train, x_valid; gc.collect()
print('Training ...')



params = {}

params['eta'] = 0.02

params['objective'] = 'reg:linear'

params['eval_metric'] = 'mae'

params['max_depth'] = 4

params['silent'] = 1



watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)



del d_train, d_validb
print('Building test set ...')



sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(properties, on='parcelid', how='left')



del properties; gc.collect()



x_test = df_test[train_columns]

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)



del df_test, sample; gc.collect()



d_test = xgb.DMatrix(x_test)



del x_test; gc.collect()



print('Predicting on test ...')



p_test = clf.predict(d_test)



del d_test; gc.collect()



sub = pd.read_csv('../input/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:

    sub[c] = p_test



print('Writing csv ...')

sub.to_csv('xgb_starter.csv', index=False, float_format='%.4f') # Thanks to @inversion