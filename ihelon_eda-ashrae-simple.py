import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

print(building_df.shape)

building_df.head()
building_df.describe()
building_df.dtypes
building_df.isna().sum() / building_df.shape[0]
building_df['site_id'] = building_df['site_id'].astype(np.uint8)

building_df['building_id'] = building_df['building_id'].astype(np.uint16)

building_df['square_feet'] = building_df['square_feet'].astype(np.uint32)

building_df.dtypes
cat_columns = ['site_id', 'primary_use']

plt.figure(figsize=(15, 5))

for ind, col in enumerate(cat_columns):

    plt.subplot(1, len(cat_columns), ind+1)

    plt.title(col)

    building_df[col].value_counts(sort=False).plot(kind='bar')
plt.figure(figsize=(10,5))

y = building_df['year_built'].value_counts(sort=False)

plt.title('year_built')

plt.xticks(rotation=45)

y.index = pd.to_datetime(y.index.astype(int), format='%Y')

y = y.sort_index()

plt.grid()

plt.plot(y)

del y
le = LabelEncoder()

building_df["primary_use_enc"] = le.fit_transform(building_df["primary_use"]).astype(np.uint8)
plt.figure(figsize=(6,6))

sns.heatmap(building_df.corr(), square=True, annot=True)
plt.figure(figsize=(5, 5))

plt.xlabel('square_feet')

plt.ylabel('floor_count')

X = building_df[pd.notnull(building_df['floor_count'])]

plt.scatter(np.log1p(X['square_feet']), X['floor_count'])

del X
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

print(weather_train.shape)

weather_train.head()
weather_train.describe()
weather_train.dtypes
weather_train.isna().sum() / weather_train.shape[0]
float_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',

        'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
weather_train['site_id'] = weather_train['site_id'].astype(np.uint8)

weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])

for i in float_cols:

    weather_train[i] = weather_train[i].astype(np.float32)

weather_train.dtypes
plt.figure(figsize=(8, 8))

sns.heatmap(weather_train.corr(), square=True, annot=True)
tmp = weather_train[weather_train['site_id'] == 0]

plt.figure(figsize=(20, 15))

for ind, col in enumerate(float_cols):

    plt.subplot(4, 2, ind + 1)

    plt.xticks(rotation=30)

    plt.grid()

    plt.title(col)

    plt.plot(tmp['timestamp'], tmp[col])