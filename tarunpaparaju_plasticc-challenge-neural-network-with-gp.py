# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data_df = pd.read_csv('../input/training_set.csv')

train_metadata_df = pd.read_csv('../input/training_set_metadata.csv')
train_data_df['flux_ratio_sq'] = np.power(train_data_df['flux'] / train_data_df['flux_err'], 2.0)

train_data_df['flux_by_flux_ratio_sq'] = train_data_df['flux'] * train_data_df['flux_ratio_sq']
data_features = train_data_df.columns[1:]

metadata_features = train_metadata_df.columns[1:]
groupObjects = train_data_df.groupby('object_id')[data_features]



print("Add constant object features")

features = train_metadata_df.drop(['target'], axis=1)



print("Add mean of mutable object features")

features = pd.merge(features, groupObjects.agg('mean'), how='right', on='object_id', suffixes=['', '_mean'])



print("Add sum of mutable object features")

features = pd.merge(features, groupObjects.agg('sum'), how='right', on='object_id', suffixes=['', '_sum'])



print("Add median of mutable features")

features = pd.merge(features, groupObjects.agg('median'), how='right', on='object_id', suffixes=['', '_median'])



print("Add minimum of mutable features")

features = pd.merge(features, groupObjects.agg('min'), how='right', on='object_id', suffixes=['', '_min'])



print("Add maximum of mutable features")

features = pd.merge(features, groupObjects.agg('max'), how='right', on='object_id', suffixes=['', '_max'])



print("Add range of mutable features")

features = pd.merge(features, groupObjects.agg(lambda x: max(x) - min(x)), how='right', on='object_id', suffixes=['', '_range'])



print("Add standard deviation of mutable features")

features = pd.merge(features, groupObjects.agg('std'), how='right', on='object_id', suffixes=['', '_stddev'])



print("Add skew of mutable features")

features = pd.merge(features, groupObjects.agg('skew'), how='right', on='object_id', suffixes=['', '_skew'])
features = features.fillna(features.mean())
features
import tensorflow as tf
import keras

from keras.utils import to_categorical
train_metadata_df['target'] = train_metadata_df.target.map({6:0, 15:1, 16:2, 42:3, 52:4, 53:5, 62:6, 64:7, 65:8, 67:9, 88:10, 90:11, 92:12, 95:13})

targets = train_metadata_df['target']
import gplearn

from gplearn.genetic import SymbolicTransformer
function_set = ['add', 'sub', 'mul', 'div',

                'sqrt', 'log', 'abs', 'neg', 'inv',

                'max', 'min']



gp = SymbolicTransformer(generations=100, population_size=2000,

                         hall_of_fame=100, n_components=10,

                         function_set=function_set,

                         parsimony_coefficient=0.0005,

                         max_samples=0.9, verbose=1,

                         random_state=0, n_jobs=3)



gp.fit(features.drop('object_id', axis=1).values, targets.values)
engineered_features = gp._programs



for i in range(len(engineered_features)):

    for engineered_feature in engineered_features[i]:

        if engineered_feature != None:

            print(engineered_feature)
new_features = pd.DataFrame(gp.transform(features.drop('object_id', axis=1).values))

features = pd.concat([features, new_features], axis=1, join_axes=[features.index])
features
targets = to_categorical(targets)
features = features.drop(['object_id'], axis=1).values
import sklearn

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features)

features = scaler.transform(features)
train_features = features [:np.int32(0.8*len(features))]

train_targets = targets [:np.int32(0.8*len(features))]



val_features = features[np.int32(0.8*len(features)):]

val_targets = targets[np.int32(0.8*len(features)):]
from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.regularizers import L1L2
model = Sequential()



model.add(Dense(30, activation='relu')) # kernel_regularizer=L1L2(l1=0.00, l2=0.01), bias_regularizer=L1L2(l1=0.00, l2=0.01)

model.add(BatchNormalization())

model.add(Dropout(0.2))





model.add(Dense(40, activation='relu')) # kernel_regularizer=L1L2(l1=0.00, l2=0.01), bias_regularizer=L1L2(l1=0.00, l2=0.01)

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(50, activation='relu')) # kernel_regularizer=L1L2(l1=0.00, l2=0.01), bias_regularizer=L1L2(l1=0.00, l2=0.01)

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(10, activation='relu')) # kernel_regularizer=L1L2(l1=0.00, l2=0.01), bias_regularizer=L1L2(l1=0.00, l2=0.01)

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Dense(targets.shape[1], activation='softmax')) # kernel_regularizer=L1L2(l1=0.00, l2=0.01), bias_regularizer=L1L2(l1=0.00, l2=0.01)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(train_features, train_targets, validation_data=(val_features, val_targets), epochs=500)
import time



test_metadata_df = pd.read_csv('../input/test_set_metadata.csv')



# print("Add constant object features")

test_metadata_df = test_metadata_df.fillna(test_metadata_df.mean())



predictions = []

object_ids = []



chunks = 5000000

total = 0



for i_c, test_data_df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):

    startTime = time.time()

    

    test_data_df['flux_ratio_sq'] = np.power(test_data_df['flux'] / test_data_df['flux_err'], 2.0)

    test_data_df['flux_by_flux_ratio_sq'] = test_data_df['flux'] * test_data_df['flux_ratio_sq']

    

    groupObjects = test_data_df.fillna(test_data_df.mean()).groupby('object_id')[data_features]



    # print("Add mean of mutable object features")

    features = groupObjects.agg('mean')

    

    # print("Add sum of mutable object features")

    features = pd.merge(features, groupObjects.agg('sum'), how='right', on='object_id', suffixes=['', '_sum'])



    # print("Add median of mutable features")

    features = pd.merge(features, groupObjects.agg('median'), how='right', on='object_id', suffixes=['', '_median'])



    # print("Add minimum of mutable features")

    features = pd.merge(features, groupObjects.agg('min'), how='right', on='object_id', suffixes=['', '_min'])



    # print("Add maximum of mutable features")

    features = pd.merge(features, groupObjects.agg('max'), how='right', on='object_id', suffixes=['', '_max'])



    # print("Add range of mutable features")

    features = pd.merge(features, groupObjects.agg(lambda x: max(x) - min(x)), how='right', on='object_id', suffixes=['', '_range'])



    # print("Add standard deviation of mutable features")

    features = pd.merge(features, groupObjects.agg('std'), how='right', on='object_id', suffixes=['', '_stddev'])

    

    # print("Add skew of mutable features")

    features = pd.merge(features, groupObjects.agg('skew'), how='right', on='object_id', suffixes=['', '_skew'])

    

    test_features = pd.merge(test_metadata_df, features, on='object_id')

    

    new_features = pd.DataFrame(gp.transform(test_features.drop('object_id', axis=1).values))

    test_features = pd.concat([test_features, new_features], axis=1, join_axes=[test_features.index])

    

    object_ids.extend(list(test_features['object_id']))

    test_features = test_features.drop(['object_id'], axis=1).values

    

    test_features = scaler.transform(test_features)

    total = total + len(test_features)

    

    predictions.extend(model.predict(test_features))



    endTime = time.time()

    

    print("Iteration : " + str(i_c))

    print("Time taken : " + str(endTime - startTime) + " s")

    print("Total objects predicted on : " + str(total))

    print("")
predictions = pd.DataFrame(predictions)
predictions.columns = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53', 'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90', 'class_92', 'class_95']
predictions['class_99'] = 1 - predictions.max(axis=1)

predictions['object_id'] = object_ids
predictions
predictions.to_csv('plasticc_submission_file.csv', index=False)