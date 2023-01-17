# Load Python libraries

from sklearn import cross_validation, grid_search, metrics, ensemble

import xgboost as xgb

from catboost import CatBoostClassifier

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import warnings

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
# Load data

df = pd.read_csv('../input/train.csv')



# 1% sample of items

df = df.sample(frac=0.01)



# Load and join songs data

songs = pd.read_csv('../input/songs.csv')

df = pd.merge(df, songs, on='song_id', how='left')

del songs



# Load and join songs data

members = pd.read_csv('../input/members.csv')

df = pd.merge(df, members, on='msno', how='left')

del members



# Replace NA

for i in df.select_dtypes(include=['object']).columns:

    df[i][df[i].isnull()] = 'unknown'

df = df.fillna(value=0)



# Create Dates



# registration_init_time

df.registration_init_time = pd.to_datetime(df.registration_init_time, format='%Y%m%d', errors='ignore')

df['registration_init_time_year'] = df['registration_init_time'].dt.year

df['registration_init_time_month'] = df['registration_init_time'].dt.month

df['registration_init_time_day'] = df['registration_init_time'].dt.day



# expiration_date

df.expiration_date = pd.to_datetime(df.expiration_date,  format='%Y%m%d', errors='ignore')

df['expiration_date_year'] = df['expiration_date'].dt.year

df['expiration_date_month'] = df['expiration_date'].dt.month

df['expiration_date_day'] = df['expiration_date'].dt.day



# Select columns

df = df[['msno', 'song_id', 'source_screen_name', 'source_type', 'target',

       'song_length', 'artist_name', 'composer', 'bd',

       'registration_init_time', 'registration_init_time_month',

       'registration_init_time_day', 'expiration_date_day']]



# Dates to categoty

df['registration_init_time'] = df['registration_init_time'].astype('category')



# Object data to category

for col in df.select_dtypes(include=['object']).columns:

    df[col] = df[col].astype('category')

    

# Encoding categorical features

for col in df.select_dtypes(include=['category']).columns:

    df[col] = df[col].cat.codes



df.info()
# Train & Test split

target = df.pop('target')

train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(df, target, test_size = 0.3)



# Delete df

del df
%%time



# Create model

model1 = ensemble.RandomForestClassifier(n_estimators=350, max_depth=40)

model1.fit(train_data, train_labels)



# Predicting

predict_labels1 = model1.predict(test_data)
%%time



# Create model

model2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, min_child_weight=10, n_estimators=250)

model2.fit(train_data, train_labels)



# Predicting

predict_labels2 = model2.predict(test_data)
%%time



# Create model

model3 = CatBoostClassifier(learning_rate=0.1, depth=10, iterations=300,)

model3.fit(train_data, train_labels)



# Predicting

predict_labels3 = model3.predict(test_data)
print(metrics.classification_report(test_labels, predict_labels1))
print(metrics.classification_report(test_labels, predict_labels2))
print(metrics.classification_report(test_labels, predict_labels3))