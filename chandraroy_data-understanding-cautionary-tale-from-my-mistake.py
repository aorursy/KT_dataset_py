import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import plotly.express as px

import plotly.graph_objects as go

import json

import seaborn as sns
print(os.listdir('../input/data-science-bowl-2019/'))
df_train = pd.read_csv("../input/data-science-bowl-2019/train.csv")

df_train_labels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

df_specs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")

df_test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
print("Shape of training data:{}".format(df_train.shape))

print("Shape of training labels data:{}".format(df_train_labels.shape))

print("Shape of specs data:{}".format(df_specs.shape))

print("Shape of test data:{}".format(df_test.shape))
df_test.installation_id.unique()
df_test.query('installation_id=="0de6863d"').head(5)
df_test.query('installation_id=="0de6863d"').tail(5)
print(df_train.columns)

print(len(df_train.columns))
df_train.head(3)
#joing train and specs datafrmae to get train_data.

#result = pd.concat([df_train, df_specs], axis=1, sort=False)



#result = pd.merge(df_train, df_specs, how='outer', on='event_id', left_on=None, right_on=None,

#         left_index=False, right_index=False, sort=False,

#         suffixes=('_x', '_y'), copy=True, indicator=False,

#         validate=None)
#result_test = pd.merge(df_test, df_specs, how='outer', on='event_id', left_on=None, right_on=None,

#         left_index=False, right_index=False, sort=False,

#         suffixes=('_x', '_y'), copy=True, indicator=False,

#         validate=None)

#result_test.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(df_train)
missing_data(df_test)
missing_data(df_specs)
missing_data(df_train_labels)
for col in df_train.columns:

    print("Column name:", col)

    print("Unique values--->",df_train[col].nunique())
for col in df_train_labels.columns:

    print("Column Name:", col)

    print("Unique values--->", df_train_labels[col].nunique())
for col in df_specs.columns:

    print("Column Name:", col)

    print("Unique values--->",df_specs[col].nunique())
for col in df_test.columns:

    print("Column Name:", col)

    print("Unique values--->",df_test[col].nunique())
%%time

#extracted_event_data = pd.io.json.json_normalize(df_train.event_data.apply(json.loads))

#extracted_event_data = pd.io.json.json_normalize(train_df.event_data.apply(json.loads))
df_train_labels.columns
df_train_labels.head()
temp_accuracy_group = df_train_labels.accuracy_group



#sns.barplot(temp_accuracy_group.index, temp_accuracy_group)
df_test.columns
min_viable_col = ['event_id', 'game_session', 'timestamp', 'event_data',

       'installation_id', 'event_count', 'event_code', 'game_time', 'title',

       'type', 'world']
train_data = df_train[min_viable_col]

train_data.columns
train_data.drop(['event_data'], axis=1, inplace = True)

df_test.drop(['event_data'], axis=1, inplace= True)
train_data.head(10)
df_test.head(10)
def extract_time_features(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['year'] = df['timestamp'].dt.year

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df.drop(['timestamp'], axis=1, inplace=True)

    return df
train_data.shape, df_test.shape
train_tmp = extract_time_features(train_data)
test_tmp = extract_time_features(df_test)
train_tmp.head()
import matplotlib.pyplot as plt
#sns.barplot(x="month", y="world", data=train_tmp)



fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="world",ax=ax[0], data=train_tmp)

sns.countplot(x="world", ax=ax[1], data=df_test)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="type",ax=ax[0], data=train_tmp)

sns.countplot(x="type", ax=ax[1], data=df_test)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(18,12))

chart1 = sns.countplot(x="title",ax=ax[0], data=train_tmp)

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=90)

chart2 = sns.countplot(x="title", ax=ax[1], data=df_test)

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=90)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="month",ax=ax[0], data=train_tmp)

sns.countplot(x="month", ax=ax[1], data=df_test)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="year",ax=ax[0], data=train_tmp)

sns.countplot(x="year", ax=ax[1], data=df_test)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="hour",ax=ax[0], data=train_tmp)

sns.countplot(x="hour", ax=ax[1], data=df_test)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="dayofweek",ax=ax[0], data=train_tmp)

sns.countplot(x="dayofweek", ax=ax[1], data=df_test)

fig.show()
fig, ax =plt.subplots(1, 2, figsize=(12,8))

sns.countplot(x="weekofyear",ax=ax[0], data=train_tmp)

sns.countplot(x="weekofyear", ax=ax[1], data=df_test)

fig.show()
train_tmp.head()
test_tmp.head()
train_tmp.shape, test_tmp.shape
df_train_labels.head()
df_train_labels.shape
len(df_train_labels.game_session.unique())