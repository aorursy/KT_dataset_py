# Imports

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np

import os

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

!pip install category_encoders ### Uncomment this went running notebook for the first time ###

import category_encoders as ce
# Create the dataframes from the csv's

train = pd.read_csv('../input/train_features.csv')

test = pd.read_csv('../input/test_features.csv')

train_labels = pd.read_csv('../input/train_labels.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
test_ids = test['id']
def wrangle(X):

    """Wrangles train, validate, and test sets in the same way"""

    X = X.copy()



    # Convert date_recorded to datetime

    X['date_recorded'] = pd.to_datetime(X['date_recorded'], infer_datetime_format=True)

    

    # Extract components from date_recorded, then drop the original column

    X['year_recorded'] = X['date_recorded'].dt.year

    X['month_recorded'] = X['date_recorded'].dt.month

    X['day_recorded'] = X['date_recorded'].dt.day

    X = X.drop(columns='date_recorded')

    

    # Engineer feature: how many years from construction_year to date_recorded

    X['years'] = X['year_recorded'] - X['construction_year']    

    

    # Drop recorded_by (never varies) and id (always varies, random)

    unusable_variance = ['recorded_by', 'id', 'num_private', 

                         'wpt_name', 'permit', 'management_group',

                         'water_quality', 'year_recorded',

                         'extraction_type_group']

    X = X.drop(columns=unusable_variance)

    

    # Drop duplicate columns

    duplicate_columns = ['quantity_group']

    X = X.drop(columns=duplicate_columns)

    

    # About 3% of the time, latitude has small values near zero,

    # outside Tanzania, so we'll treat these like null values

    X['latitude'] = X['latitude'].replace(-2e-08, np.nan)

    

    # When columns have zeros and shouldn't, they are like null values

    cols_with_zeros = ['construction_year', 'longitude', 'latitude',

                       'gps_height', 'population']

    for col in cols_with_zeros:

        X[col] = X[col].replace(0, np.nan)

        

    return X
train['status_group'] = train_labels['status_group']
from sklearn.cluster import KMeans

train['longitude'].values

points = train[['longitude', 'latitude']].fillna(0).values

kmeans = KMeans(n_clusters=14, random_state=42)

kmeans.fit(points)

y_km = kmeans.fit_predict(points)



data = pd.DataFrame({'lon': points[:,0], 'lat': points[:,1], 'cluster':y_km})



train['cluster'] = data['cluster']



# color = data['cluster'].replace({0:'r', 1:'orange', 2:'y', 3:'g',

#                                  4:'b', 5:'indigo', 6:'violet', 7:'black',

#                                  8:'lightblue', 9:'lightgreen', 10:'grey', 

#                                  11:'r', 12:'brown', 13:'brown', 14:'purple',

#                                  15:'black'})





# plt.figure(figsize=(10, 10))

# plt.style.use('ggplot')

# plt.scatter(train['longitude'], train['latitude'], s=2, c = color)

# plt.xlabel('Longitude', fontsize=20)

# plt.ylabel('Latitude', fontsize=20)
test['longitude'].values

points = test[['longitude', 'latitude']].fillna(0).values

kmeans = KMeans(n_clusters=14)

kmeans.fit(points)

y_km = kmeans.fit_predict(points)



data = pd.DataFrame({'lon': points[:,0], 'lat': points[:,1], 'cluster':y_km})



test['cluster'] = data['cluster']
train, val = train_test_split(train, test_size= test.shape[0], stratify=train['status_group'])
train = wrangle(train)

val = wrangle(val)

test = wrangle(test)
X_train = train.drop('status_group', axis=1)

y_train = train['status_group']

X_val = val.drop('status_group', axis=1)

y_val = val['status_group']

X_test = test
rfc = RandomForestClassifier(n_estimators=1000,

                               min_samples_split=6,

                               max_depth = 23,

                               criterion='gini',

                               max_features='auto',

                               random_state=42,

                               n_jobs=-1)



pipe = make_pipeline(

    ce.OrdinalEncoder(),

    SimpleImputer(strategy='median'),

    rfc)



pipe.fit(X_train, y_train)

pipe.score(X_train, y_train)
y_pred = pipe.predict(X_val)

accuracy_score(y_pred, y_val)
importances = rfc.feature_importances_

features = X_train.columns

plt.style.use('ggplot')

plt.figure(figsize=(10, 10))

plt.barh(features, importances)

plt.axvline(.008, c='b')
y_pred = pipe.predict(X_test)
y_pred = pipe.predict(X_test)

sub = pd.DataFrame(data = {

    'id': test_ids,

    'status_group': y_pred

})

sub.to_csv('submission.csv', index=False)
color = train['status_group'].replace({'functional': 'g', 'functional needs repair': 'y', 'non functional': 'r'})

plt.figure(figsize=(10, 10))

plt.style.use('ggplot')

plt.scatter(train['longitude'], train['latitude'], s=2, color = color)

plt.xlabel('Longitude', fontsize=20)

plt.ylabel('Latitude', fontsize=20)

red_patch = mpatches.Patch(color='red', label='Non-Functional')

green_patch = mpatches.Patch(color='green', label='Functional')

yellow_patch = mpatches.Patch(color='yellow', label='Functional, Needs Work')

plt.legend(handles=[green_patch, yellow_patch, red_patch])

train.head()
plt.scatter(train['population'], train['quantity'], alpha=.1)
