import pandas as pd

import os

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


data_train_path = '../input/X_train.csv'

response_train_path = '../input/y_train.csv'

test_data_path = '../input/X_test.csv'

sub_data_path = '../input/sample_submission.csv'



data_train = pd.read_csv(data_train_path)

response_train = pd.read_csv(response_train_path)

data_test = pd.read_csv(test_data_path)

sub_data = pd.read_csv(sub_data_path)
data_train.head()
data_train.shape
print( 'Counts row_id: ' + str(len(data_train['row_id'].unique())),

      '\nCounts series_id: ' + str(len(data_train['series_id'].unique())),

      '\nCounts measurement_number: ' + str(len(data_train['measurement_number'].unique())) 

     )
#all values in 'series_id' hold the same number of measurements

sum(data_train.groupby('series_id')['measurement_number'].count() !=128)
response_train.head()
#73 numbers for group id

len(response_train['group_id'].unique())
series_bygroupid = response_train.groupby('series_id')['group_id'].max()

groupid_counts = response_train.groupby('series_id')['group_id'].count()



series_bygroupid.head()
len(groupid_counts == 1)
series_bygroupid.plot()
#we've got 9 different categories

len(response_train['surface'].unique())
#let's see their names

response_train['surface'].unique()
response_train.groupby('surface')['group_id'].count()
response_train.groupby('surface')['group_id'].nunique()
response_train['surface'] = response_train['surface'].astype('category')
sum(response_train.groupby('group_id')['surface'].nunique() != 1)
data_train.dtypes
data_train.describe()
unit_quat = (data_train['orientation_W']**2+

data_train['orientation_X']**2+  

data_train['orientation_Y']**2+

data_train['orientation_Z']**2)



unit_quat.head()
data_train['angular_velocity_Y'].plot()
data_train['linear_acceleration_X'].plot()
data_train['linear_acceleration_Y'].plot()
data_train['linear_acceleration_Z'].plot()
data_merge = pd.merge(data_train, response_train, on = 'series_id')
#we check all series have only one category assigned

sum(data_merge.groupby('series_id')['surface'].nunique() !=  1)
data_clus = data_train.drop(['row_id','measurement_number'], axis=1)
data_clusmean = data_clus.groupby('series_id').mean()
kmeans = KMeans(n_clusters=9, random_state=0).fit(data_clusmean)
#let's get the labels as a dataframe and create the column for series_id so we can merge it

labels_clus = pd.DataFrame(kmeans.labels_)

labels_clus['series_id'] = range(response_train.shape[0])
labels_clus.columns = ['labels', 'series_id']
response_labeled = pd.merge(response_train, labels_clus, on='series_id')
freq_catlabel = response_labeled.groupby('labels')['surface'].value_counts()
freq_catlabel
freq_labels_dict = ({0:'concrete', 1:'soft_pvc',2:'wood', 3:'concrete',

                    4:'soft_pvc',5:'concrete',6:'tiled',7:'soft_tiles', 8:'tiled'})
freq_labels_dict
data_test_cluster = data_test.drop(['measurement_number', 'row_id'],axis=1)

data_test_cluster_group = data_test_cluster.groupby('series_id').mean()
#predict fitted clusters on the test

y_pred_test = kmeans.predict(data_test_cluster_group)
y_pred_test = pd.DataFrame(y_pred_test)
y_pred_test['series_id']  = range(len(y_pred_test))
y_pred_test.columns = ['surface', 'series_id']
y_pred_test['surface']  = y_pred_test['surface'].map(freq_labels_dict)
y_pred_test = y_pred_test[['series_id', 'surface']]
#y_pred_test.to_csv()