import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import math

import random
# load training and testing data

trainx = pd.read_csv('../input/X_train.csv')

trainy = pd.read_csv('../input/y_train.csv')

testx = pd.read_csv('../input/X_test.csv')
# Create a dataset with averages



# list of unique series_id's

series_ids = trainx.series_id.unique()



# create an empty dataframe

cols = trainx.columns.tolist()

cols.remove('row_id')

cols.remove('measurement_number')

shape = (len(series_ids),11)

data = np.empty(shape=shape)

trainx_base = pd.DataFrame(data=data,columns=cols)



# list of sensors

cols.remove('series_id')

sensors = cols



for id in series_ids:

    means = []

    for sensor in sensors:

        means.append(trainx[trainx.series_id==id][sensor].mean())

    means.insert(0, id)

    trainx_base.iloc[id] = means



# default data type of float64 is fine for all columns except for series_id - it needs to be int64

trainx_base = trainx_base.astype({'series_id':np.int64})



trainx_base.head()
# And another one for the test set



# list of unique series_id's

series_ids = testx.series_id.unique()



# create am empty dataframe

cols = testx.columns.tolist()

cols.remove('row_id')

cols.remove('measurement_number')

shape = (len(series_ids),11)

data = np.empty(shape=shape)

testx_base = pd.DataFrame(data=data,columns=cols)



# list of sensors

cols.remove('series_id')

sensors = cols



for id in series_ids:

    means = []

    for sensor in sensors:

        means.append(testx[testx.series_id==id][sensor].mean())

    means.insert(0, id)

    testx_base.iloc[id] = means



# default data type of float64 is fine for all columns except for series_id - it needs to be int64

testx_base = testx_base.astype({'series_id':np.int64})



testx_base.head()
# Min-max normalize all data, i.e. convert to number from 0 to 1.



scaler = MinMaxScaler(feature_range=(0,1))

sensors = trainx_base.columns.tolist()

sensors.remove('series_id')



trainx_base_normalized = pd.DataFrame(data=trainx_base)

trainx_base_normalized[sensors] = scaler.fit_transform(trainx_base[sensors])

trainx_base_normalized.head()
# And again for the test set



scaler = MinMaxScaler(feature_range=(0,1))

sensors = testx_base.columns.tolist()

sensors.remove('series_id')



testx_base_normalized = pd.DataFrame(data=testx_base)

testx_base_normalized[sensors] = scaler.fit_transform(testx_base[sensors])

testx_base_normalized.head()
# First, lets flag about 10% of the data for testing, working through each surface individually



# percent of sample to reserve for testing/validation:

test_part = 0.1



# for reproducability

random.seed(27)



trainx_base_normalized['test'] = 0



surfaces = trainy.surface.unique().tolist()



for surface in surfaces:

    #print('working on surface "{}"'.format(surface))

    surface_series_id = trainy[trainy.surface==surface].series_id.tolist()

    #print('  found {} ids in total'.format(len(surface_series_id)))

    test_part_cnt = math.floor(len(surface_series_id) * test_part)

    #print('  picked {} ids for testing'.format(test_part_cnt))

    test_series_id = random.sample(surface_series_id, test_part_cnt)

    trainx_base_normalized.loc[test_series_id, 'test'] = 1
print(len(trainx_base_normalized[trainx_base_normalized.test==0]), '<- for training')

print(len(trainx_base_normalized[trainx_base_normalized.test==1]), '<- for testing')
# Lets bring in the labels



basemodel = trainx_base_normalized.copy()

basemodel['surface'] = trainy.surface

basemodel.head()
# Now, separate out the 10% of the data that we flagged earlier for testing/validation



basemodel_train_X = basemodel.copy()

basemodel_train_X.drop(basemodel_train_X[basemodel_train_X['test']==1].index.tolist(), inplace=True)

basemodel_train_X.reset_index(drop=True, inplace=True)



basemodel_valid_X = basemodel.copy()

basemodel_valid_X.drop(basemodel_valid_X[basemodel_valid_X['test']==0].index.tolist(), inplace=True)

basemodel_valid_X.reset_index(drop=True, inplace=True)
# Finally, create labels for each of the two datasets above



basemodel_train_y = basemodel_train_X.copy()

basemodel_train_X.drop(['series_id','test','surface'], axis=1, inplace=True)

basemodel_train_y.drop(['series_id','orientation_X','orientation_Y','orientation_Z','orientation_W',

                        'angular_velocity_X','angular_velocity_Y','angular_velocity_Z','linear_acceleration_X',

                        'linear_acceleration_Y','linear_acceleration_Z','test'], axis=1, inplace=True)



basemodel_valid_y = basemodel_valid_X.copy()

basemodel_valid_X.drop(['series_id','test','surface'], axis=1, inplace=True)

basemodel_valid_y.drop(['series_id','orientation_X','orientation_Y','orientation_Z','orientation_W',

                        'angular_velocity_X','angular_velocity_Y','angular_velocity_Z','linear_acceleration_X',

                        'linear_acceleration_Y','linear_acceleration_Z','test'], axis=1, inplace=True)
# Use SVM classifier and fit the training data



from sklearn.svm import SVC



clf = SVC(kernel='rbf', gamma='scale')

clf.fit(basemodel_train_X, basemodel_train_y)
# Make predictions and compute accuracy score (on the validation set)

from sklearn.metrics import accuracy_score



y_true = basemodel_valid_y.surface.tolist()

y_pred = clf.predict(basemodel_valid_X)

acc_score = accuracy_score(y_true, y_pred)

print(acc_score)
# Make predictions on the test set



basemodel_test = testx_base_normalized.copy()

# we don't need the "series_id" columns because it is the same as the index

basemodel_test.drop(['series_id'], axis=1, inplace=True)



test_preds = clf.predict(basemodel_test)



# into a dataframe

test_preds_df = pd.DataFrame(data={'series_id':range(0,3816), 'surface':test_preds})



test_preds_df.head()
# Write to file (for submission)

# test_preds_df.to_csv('submission.csv', index=False)
# Use all training data (no validation)



basemodel_full_X = basemodel.copy()

basemodel_full_X.drop(['series_id','test'], axis=1, inplace=True)

basemodel_full_y = basemodel_full_X.copy()

basemodel_full_X.drop(['surface'], axis=1, inplace=True)

basemodel_full_y.drop(['orientation_X','orientation_Y','orientation_Z','orientation_W','angular_velocity_X','angular_velocity_Y','angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z'], axis=1, inplace=True)



# print(basemodel_full_X.shape)

# print(basemodel_full_y.shape)
# Fit the model

clf_full = SVC(kernel='rbf', gamma='scale')

clf_full.fit(basemodel_full_X, basemodel_full_y)



# And just because we can, let's predict on the validation set we still have from before

y_true = basemodel_valid_y.surface.tolist()

y_pred = clf_full.predict(basemodel_valid_X)

acc_score = accuracy_score(y_true, y_pred)

print(acc_score)
# Make predictions on the test set

test_preds_full = clf.predict(basemodel_test)

# into a dataframe

test_preds_full_df = pd.DataFrame(data={'series_id':range(0,3816), 'surface':test_preds_full})

test_preds_full_df.head()
# Write to file (for submission)

test_preds_full_df.to_csv('submission.csv', index=False)