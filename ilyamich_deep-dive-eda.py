import pandas as pd

import numpy as np



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

matplotlib.rcParams['figure.dpi'] = 100

%matplotlib inline
data_path = '../input/ashrae-energy-prediction/'



building_path = data_path + 'building_metadata.csv'

weather_train_path = data_path + 'weather_train.csv'

train_path = data_path + 'train.csv'

test_path = data_path + 'test.csv'

weather_test_path = data_path + 'weather_test.csv'

sample_path = data_path + 'sample_submission.csv'
building_df = pd.read_csv(building_path)

weather_train_df = pd.read_csv(weather_train_path)

# train_df = pd.read_csv(train_path)

# test_df = pd.read_csv(test_path)

# weather_test_df = pd.read_csv(weather_test_path)

# sample_df = pd.read_csv(sample_path)
# check and print if there are empty values in a dataset

def check_for_nulls(dataset, name):

    print('There are {} {}'.format(len(dataset), name))

    for idx, key in enumerate(dataset.count()):

        if len(dataset) > key:

            print("There are {0:.2f}% ({1}) missing values in colum '{2}'".format((len(dataset)-key)/len(dataset)*100, len(dataset)-key, dataset.columns[idx]))



# plot using boxplot and scatterplot

def plot_moxes(x, y, title, dataframe):

    plt.subplots(figsize=[20,6])

    sns.boxplot(x=x, y=y, data=building_df)

    sns.stripplot(x=x, y=y, data=dataframe)

    plt.title(title)



# calculate humidity

def calc_humidity(temperature, dew_point):

    Es = 6.11 * 10**((7.5*temperature) / (237.3+temperature))

    E = 6.11 * 10**((7.5*dew_point) / (237.3+dew_point))

    

    return (E/Es) * 100
building_df.head()
check_for_nulls(building_df, 'buildings')
building_df['building_volume'] = building_df['square_feet'] * building_df['floor_count']
building_count = building_df.groupby('primary_use')['primary_use'].count()



keys = building_count.keys()

values = building_count.values

idxs = np.flip(np.argsort(values))

keys = keys[idxs]

values = values[idxs]



buildings_n = len(building_df)



# print number of building types

for key, value in zip(keys, values):

    print("There are {0:.2f}% ({1}) buildings of type '{2}'".format(100-(buildings_n-value)/buildings_n*100, value, key))



# plot

fig, ax = plt.subplots(figsize=[20,12])

sns.barplot(ax=ax, x=values, y=keys)

plt.title('Number of building types');
plot_moxes('square_feet', 'primary_use', 'Building area distrebution for each building type', building_df)
plt.subplots(figsize=[20,6])

sns.distplot(building_df['square_feet'], bins=50, kde=False)

plt.title('Building area distrebution');
plot_moxes('building_volume', 'primary_use', 'Building volume distrebution for each building type', building_df)
plot_moxes('year_built', 'primary_use', 'Building building year distrebution for each building type', building_df)
weather_train_df.head()
check_for_nulls(weather_train_df, 'weather timestamps')
weather_train_df['humidity'] = calc_humidity(weather_train_df['air_temperature'], weather_train_df['dew_temperature'])

weather_train_df.head()
def time2num(time_s):

    time = time_s.split(':')

    return int(time[0]) * 60 * 60 + int(time[1]) * 60 + int(time[0])
timestamp_df = pd.DataFrame()

timestamp_df['site_id'] = weather_train_df['site_id']

timestamp_df['date'] = weather_train_df['timestamp'].apply(lambda x: x.split(' ')[0])

timestamp_df['year'] = timestamp_df['date'].apply(lambda x: int(x.split('-')[0]))

timestamp_df['month'] = timestamp_df['date'].apply(lambda x: int(x.split('-')[1]))

timestamp_df['day'] = timestamp_df['date'].apply(lambda x: int(x.split('-')[2]))

timestamp_df['time'] = weather_train_df['timestamp'].apply(lambda x: time2num(x.split(' ')[1]))



timestamp_df.head()
site_n = weather_train_df['site_id'].unique()



for site_id in site_n:

    times = timestamp_df[timestamp_df['site_id'] == site_id]['time'].to_numpy()



    print('Site number {} has unique time jumps: {}'.format(site_id, np.unique(times[1:] - times[:-1])))