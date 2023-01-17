# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plant1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

plant2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')

weather1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

weather2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
plant1.tail()
plant2.tail()
plant1['DATE_TIME'] = pd.to_datetime(plant1['DATE_TIME'], format = '%d-%m-%Y %H:%M')

weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

plant1.columns = plant1.columns.str.lower()

weather1.columns = weather1.columns.str.lower()

plant2['DATE_TIME'] = pd.to_datetime(plant2['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

weather2['DATE_TIME'] = pd.to_datetime(weather2['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

plant2.columns = plant2.columns.str.lower()

weather2.columns = weather2.columns.str.lower()
print(plant1.head())

print(plant2.head())
plant1['date'] = plant1['date_time'].dt.date

plant1['time'] = plant1['date_time'].dt.time

plant2['date'] = plant2['date_time'].dt.date

plant2['time'] = plant2['date_time'].dt.time
dc_plant1 = plant1.groupby('time')['dc_power'].sum()

ac_plant1 = plant1.groupby('time')['ac_power'].sum()

dc_plant2 = plant2.groupby('time')['dc_power'].sum()

ac_plant2 = plant2.groupby('time')['ac_power'].sum()
fig, ax = plt.subplots(1, 2, dpi=100, figsize=(20, 5))

dc_plant1.plot(ax=ax[0])

dc_plant2.plot(ax=ax[0])

ac_plant1.plot(ax=ax[1])

ac_plant2.plot(ax=ax[1])

ax[0].legend(['plant1', 'plant2'])

ax[1].legend(['plant1', 'plant2'])

ax[0].set_title('DC Power')

ax[1].set_title('AC Power')

plt.show()
loss_p1 = plant1.copy()

loss_p2 = plant2.copy()

loss_p1 = loss_p1.groupby('date').sum()

loss_p1['losses'] = loss_p1['ac_power'] / loss_p1['dc_power'] * 100

loss_p2 = loss_p2.groupby('date').sum()

loss_p2['losses'] = loss_p2['ac_power'] / loss_p2['dc_power'] * 100



#Plot the losses

fig, ax = plt.subplots(2, 1, sharex = True, dpi=100, figsize=(13,7))

loss_p1['losses'].plot(style='o--', ax=ax[0])

loss_p2['losses'].plot(style='o--', ax=ax[1])

ax[0].set_title('Percentage of DC Power converted into AC Power for Plant 1')

ax[1].set_title('Percentage of DC Power converted into AC Power for Plant 2')

plt.xticks(rotation=45)

plt.show()
unique_keyes = set(plant1['source_key'])

total_dc_power_p1 = {}

for key in unique_keyes:

    dc_power = plant1[plant1['source_key'] == key]['total_yield'].iloc[-1] - plant1[plant1['source_key'] == key]['total_yield'].iloc[0]

    total_dc_power_p1[key] = dc_power

print(total_dc_power_p1)

fig, ax = plt.subplots(figsize = (17, 5))

ax.plot(list(total_dc_power_p1.values()), marker = '^', linestyle = '-.')

ax.set(xlabel = 'Source', ylabel='kW', title='Total yielded power by different solar battery for plant 1')

plt.xticks(range(0, 22), list(total_dc_power_p1.keys()), rotation = 90)

plt.show()
unique_keyes2 = set(plant2['source_key'])

total_dc_power_p2 = {}

for key in unique_keyes2:

    dc_power = plant2[plant2['source_key'] == key]['total_yield'].iloc[-1] - plant2[plant2['source_key'] == key]['total_yield'].iloc[0]

    total_dc_power_p2[key] = dc_power

print(total_dc_power_p2)

fig, ax = plt.subplots(figsize = (17, 5))

ax.plot(list(total_dc_power_p2.values()), marker = '^', linestyle = '-.')

ax.set(xlabel = 'Source', ylabel='kW', title='Total yielded power by different solar battery for plant 2')

plt.xticks(range(0, 22), list(total_dc_power_p2.keys()), rotation = 90)

plt.show()
worst_inverter_p1 = plant1[plant1['source_key'] == 'bvBOhCH3iADSZry'].reset_index(drop=True)



ax=worst_inverter_p1.groupby(['time', 'date'])['dc_power'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))

worst_inverter_p1.groupby(['time', 'date'])['daily_yield'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')

cols=worst_inverter_p1.groupby(['time', 'date'])['dc_power'].mean().unstack().columns

a=0

for i in range(len(ax)):

    for j in range(len(ax[i])):

        ax[i,j].set_title(cols[a], size=15)

        ax[i,j].legend(['dc_power','daily_yield'])

        a=a+1

plt.tight_layout()

plt.show()
best_inverter_p1 = plant1[plant1['source_key'] == 'adLQvlD726eNBSB'].reset_index(drop=True)



ax=best_inverter_p1.groupby(['time', 'date'])['dc_power'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))

best_inverter_p1.groupby(['time', 'date'])['daily_yield'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')

cols=best_inverter_p1.groupby(['time', 'date'])['dc_power'].mean().unstack().columns

a=0

for i in range(len(ax)):

    for j in range(len(ax[i])):

        ax[i,j].set_title(cols[a], size=15)

        ax[i,j].legend(['dc_power','daily_yield'])

        a=a+1

plt.tight_layout()

plt.show()
worst_inverter_p2 = plant2[plant2['source_key'] == 'Quc1TzYxW2pYoWX'].reset_index(drop=True)



ax=worst_inverter_p2.groupby(['time', 'date'])['dc_power'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30))

worst_inverter_p2.groupby(['time', 'date'])['daily_yield'].mean().unstack().plot(sharex=True,subplots=True,layout=(17,2),figsize=(20,30),ax=ax,style='-.')

cols=worst_inverter_p2.groupby(['time', 'date'])['dc_power'].mean().unstack().columns

a=0

for i in range(len(ax)):

    for j in range(len(ax[i])):

        ax[i,j].set_title(cols[a], size=15)

        ax[i,j].legend(['dc_power','daily_yield'])

        a=a+1

plt.tight_layout()

plt.show()
df_plant1 = plant1.merge(weather1, on='date_time', suffixes=['', '_w'])

df_plant1['hour'] = df_plant1['date_time'].dt.hour

df_plant1 = df_plant1.drop(['source_key_w', 'plant_id', 'plant_id_w', 'date', 'time', 'date_time'], axis = 1)

df_to_corr = df_plant1.drop('source_key', axis=1)

df_to_corr.corr()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
df1 = plant1.merge(weather1, on='date_time', suffixes=['', '_w'])

columns_to_drop = ['date_time', 'date', 'time', 'dc_power', 'daily_yield', 'total_yield', 'module_temperature', 'irradiation', 'source_key_w', 'plant_id', 'plant_id_w']

df_p1 = df1.drop(columns_to_drop, axis=1)

df_p1['hour'] = df1['date_time'].dt.hour
values = np.array(df_p1['source_key'])

#Encoding

label_enc = LabelEncoder()

integer_enc = label_enc.fit_transform(values)

onehot_enc = OneHotEncoder(sparse=False)

enc_keys = onehot_enc.fit_transform(integer_enc.reshape(-1, 1))

enc_keys_df = pd.DataFrame(enc_keys)

# Create dictionary with categories of encoded feature, to use them in plotting

keys = label_enc.classes_

values = label_enc.transform(label_enc.classes_)

dictionary = dict(zip(values, keys))
df_p1 = pd.concat([df_p1, enc_keys_df], axis=1).drop('source_key', axis=1)

df_p1.head()
train_columns = df_p1.columns[1:]

X_train = df_p1[df1['date_time'] < '2020-06-16'][train_columns].reset_index().drop('index', axis=1)

X_test = df_p1[df1['date_time'] < '2020-06-16']['ac_power'].reset_index().drop('index', axis=1)

y_test = df_p1[df1['date_time'] >= '2020-06-16']['ac_power'].reset_index().drop('index', axis=1)

y_train = df_p1[df1['date_time'] >= '2020-06-16'][train_columns].reset_index().drop('index', axis=1)
reg = RandomForestRegressor()

reg.fit(X_train, X_test)

y_pred_rf = reg.predict(y_train)

print(mean_absolute_error(y_pred_rf, y_test))

print(r2_score(y_pred_rf, y_test))
gbr = GradientBoostingRegressor(learning_rate = 0.05, n_estimators=200)

gbr.fit(X_train, X_test)

y_pred = gbr.predict(y_train)

print(mean_absolute_error(y_pred, y_test))

print(r2_score(y_pred, y_test))
fig = plt.figure(figsize=(20, 20))

fig.subplots_adjust(wspace=0.2, hspace=0.6)

for i in range(0, 22):

    ax = fig.add_subplot(6, 4, i+1)

    index = df1[(df1['source_key'] == dictionary[i])&(df1['date_time'] >= pd.to_datetime('2020-06-16'))]['date_time'] # just for index

    ax.plot(pd.DataFrame(y_test[y_train[y_train.columns[i+2]] == 1]).set_index(index))

    ax.plot(pd.DataFrame(y_pred[y_train[y_train.columns[i+2]] == 1]).set_index(index), color='darkorange')

    ax.set_title('"{}" source, Plant 1'.format(dictionary[i]))

    ax.legend(['real', 'predicted'])

    plt.xticks(rotation = 45)
df2 = plant2.merge(weather2, on='date_time', suffixes=['', '_w'])

columns_to_drop = ['date_time', 'date', 'time', 'dc_power', 'daily_yield', 'total_yield', 'module_temperature', 'irradiation', 'source_key_w', 'plant_id', 'plant_id_w']

df_p2 = df2.drop(columns_to_drop, axis=1)

df_p2['hour'] = df2['date_time'].dt.hour
values = np.array(df_p2['source_key'])



#Encoding



label_enc = LabelEncoder()

integer_enc = label_enc.fit_transform(values)

onehot_enc = OneHotEncoder(sparse=False)

enc_keys = onehot_enc.fit_transform(integer_enc.reshape(-1, 1))

enc_keys_df = pd.DataFrame(enc_keys)



# Create dictionary with categories of encoded feature, to use them in plotting



keys = label_enc.classes_

values = label_enc.transform(label_enc.classes_)

dictionary = dict(zip(values, keys))
df_p2 = pd.concat([df_p2, enc_keys_df], axis=1).drop('source_key', axis=1)

df_p2.head()
train_columns = df_p2.columns[1:]

X_train2 = df_p2[df2['date_time'] < '2020-06-16'][train_columns].reset_index(drop=True)

X_test2 = df_p2[df2['date_time'] < '2020-06-16']['ac_power'].reset_index(drop=True)

y_test2 = df_p2[df2['date_time'] >= '2020-06-16']['ac_power'].reset_index(drop=True)

y_train2 = df_p2[df2['date_time'] >= '2020-06-16'][train_columns].reset_index(drop=True)
gbr = GradientBoostingRegressor(learning_rate = 0.05, n_estimators=200)

gbr.fit(X_train2, X_test2)

y_pred2 = gbr.predict(y_train2)

print(mean_absolute_error(y_pred2, y_test2))

print(r2_score(y_pred2, y_test2))
fig = plt.figure(figsize=(20, 20))

fig.subplots_adjust(wspace=0.2, hspace=0.6)

for i in range(0, 22):

    ax = fig.add_subplot(6, 4, i+1)

    index = df2[(df2['source_key'] == dictionary[i])&(df2['date_time'] >= pd.to_datetime('2020-06-16'))]['date_time'] # just for index

    ax.plot(pd.DataFrame(y_test2[y_train2[y_train2.columns[i+2]] == 1]).set_index(index))

    ax.plot(pd.DataFrame(y_pred2[y_train2[y_train2.columns[i+2]] == 1]).set_index(index), color='darkorange')

    ax.set_title('"{}" source, Plant 2'.format(dictionary[i]))

    ax.legend(['real', 'predicted'])

    plt.xticks(rotation = 45)