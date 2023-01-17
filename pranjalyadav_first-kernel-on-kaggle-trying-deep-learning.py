import pandas as pd

import numpy as np

import gc

import matplotlib.pyplot as plt



%matplotlib inline



pd.options.display.max_columns = 999





import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn import model_selection



from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler

from sklearn.neighbors import NearestNeighbors



import operator, random, pickle

import math, keras, datetime, keras.backend as K, tensorflow as tf, matplotlib.pyplot as plt, operator, random, pickle, glob, os, functools, itertools

from numpy.random import normal



from keras import initializers

from keras.models import Model, Sequential

from keras.layers import *

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.utils.data_utils import get_file

train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
prop = pd.read_csv("../input/properties_2016.csv", low_memory=False)
sample = pd.read_csv("../input/sample_submission.csv")
xls = pd.ExcelFile('../input/zillow_data_dictionary.xlsx')

sheets = {sh:xls.parse(sh) for sh in xls.sheet_names}
prop = prop.merge(sheets['AirConditioningTypeID'], on ='airconditioningtypeid', how='left')

prop = prop.merge(sheets['HeatingOrSystemTypeID'], on ='heatingorsystemtypeid', how='left')

prop = prop.merge(sheets['PropertyLandUseTypeID'], on ='propertylandusetypeid', how='left')

prop = prop.merge(sheets['StoryTypeID'], on ='storytypeid', how='left')

prop = prop.merge(sheets['ArchitecturalStyleTypeID'], on ='architecturalstyletypeid', how='left')

prop = prop.merge(sheets['TypeConstructionTypeID'], on ='typeconstructiontypeid', how='left')
prop['airconditioningdesc'].fillna('Other', inplace=True)

prop['heatingorsystemdesc'].fillna('Other', inplace=True)

prop['propertylandusedesc'].fillna('Other', inplace=True)

prop['storydesc'].fillna('Other', inplace=True)

prop['architecturalstyledesc'].fillna('Other', inplace=True)

prop['typeconstructiondesc'].fillna('Other', inplace=True)
waste_col = []



for col in prop.columns:

    if prop.ix[:,col].isnull().sum()/prop.shape[0] >0.9:

        waste_col.append(col)

    else:

        continue
prop.drop(waste_col,axis=1,inplace=True)
prop.fillna(prop.median(),inplace=True)
prop.drop(['propertyzoningdesc','propertycountylandusecode'], axis=1, inplace=True)
prop.select_dtypes(include=["object"]).columns
cat_var_dict = {'airconditioningdesc':10,'heatingorsystemdesc':10,'propertylandusedesc':10,

                'storydesc':2,'architecturalstyledesc':10,'typeconstructiondesc':10}
cat_vars = [o[0] for o in 

            sorted(cat_var_dict.items(), key=operator.itemgetter(1), reverse=True)]
contin_vars = prop.select_dtypes(include=['float64']).columns
cat_maps = [(o, LabelEncoder()) for o in cat_vars]

contin_maps = [([o], StandardScaler()) for o in contin_vars]
cat_mapper = DataFrameMapper(cat_maps)

cat_map_fit = cat_mapper.fit(prop)

cat_cols = len(cat_map_fit.features)

contin_mapper = DataFrameMapper(contin_maps)

contin_map_fit = contin_mapper.fit(prop)

contin_cols = len(contin_map_fit.features)

cat_map_fit.transform(prop)[0,:5], contin_map_fit.transform(prop)[0,:5]
# print('Creating training set ...')



df_train = train.merge(prop, how='left', on='parcelid')
# print('Creating test set ...')



sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(prop, on='parcelid', how='left')

def missmatch_col():

    extra_train_cols = []

    extra_test_cols = []



    for i in df_train.columns:

        if i in df_test.columns:

            continue

        else:

            extra_train_cols.append(i)



    for i in df_test.columns:

        if i in df_train.columns:

            continue

        else:

            extra_test_cols.append(i)

    print("Extra Columns in Train, ","Extra columns in Test")        

    return extra_train_cols,extra_test_cols
def add_datepart(df):

    df.transactiondate = pd.to_datetime(df.transactiondate)

    df["Year"] = df.transactiondate.dt.year

    df["Month"] = df.transactiondate.dt.month

    df["Week"] = df.transactiondate.dt.week

    df["Day"] = df.transactiondate.dt.day
y_train = df_train['logerror'].values
df_train.drop(missmatch_col()[0], axis=1, inplace=True)

df_test.drop(missmatch_col()[1], axis=1, inplace=True)
df_test.drop(['parcelid'], axis=1, inplace=True)

df_train.drop(['parcelid'], axis=1, inplace=True)
def cat_preproc(dat):

    return cat_map_fit.transform(dat).astype(np.int64)
def contin_preproc(dat):

    return contin_map_fit.transform(dat).astype(np.float32)
split = 80000

x_train, y_train, x_valid, y_valid = df_train[:split], y_train[:split], df_train[split:], y_train[split:]
cat_map_train = cat_preproc(x_train)

cat_map_valid = cat_preproc(x_valid)
contin_map_train = contin_preproc(x_train)

contin_map_valid = contin_preproc(x_valid)
cat_map_test = cat_preproc(df_test)

contin_map_test = contin_preproc(df_test)
def split_cols(arr): return np.hsplit(arr,arr.shape[1])
map_train = split_cols(cat_map_train) + [contin_map_train]

map_valid = split_cols(cat_map_valid) + [contin_map_valid]
map_test = split_cols(cat_map_test) + [contin_map_test]
def cat_map_info(feat): return feat[0], len(feat[1].classes_)
def emb_init(shape, dtype=None):

    return K.random_normal(shape, dtype=dtype)
def get_emb(feat):

    name, c = cat_map_info(feat)

    c2 = (c+1)//2

    inp = Input((1,), dtype='int64', name=name+'_in')

    # , W_regularizer=l2(1e-6)

    u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1, init=emb_init)(inp))

#    u = Flatten(name=name+'_flt')(Embedding(c, c2, input_length=1)(inp))

    return inp,u
def get_contin(feat):

    name = feat[0][0]

    inp = Input((1,), name=name+'_in')

    return inp, Dense(1, name=name+'_d', init=my_init(1.))(inp)
contin_inp = Input((contin_cols,), name='contin')

contin_out = Dense(contin_cols*10, activation='relu', name='contin_d')(contin_inp)

contin_out = BatchNormalization()(contin_out)
embs = [get_emb(feat) for feat in cat_map_fit.features]

x = merge([emb for inp,emb in embs] + [contin_out], mode='concat')

x = Dense(500, activation='relu', kernel_initializer='uniform')(x)

x = Dense(200, activation='relu', kernel_initializer='uniform')(x)

x = Dropout(0.2)(x)

x = Dense(1, activation='linear')(x)



model = Model([inp for inp,emb in embs] + [contin_inp], x)

model.compile('adam', 'mean_absolute_error')
histMax = model.fit(map_train, y_train, batch_size=64, epochs=5,

                 verbose=0, validation_data=(map_valid, y_valid))
ans = np.squeeze(model.predict(map_test))
sample = pd.read_csv('sample_submission.csv')

for c in sample.columns[sample.columns != 'ParcelId']:

    sample[c] = ans



print('Writing csv ...')

sample.to_csv('emb_model.csv', index=False, float_format='%.4f')