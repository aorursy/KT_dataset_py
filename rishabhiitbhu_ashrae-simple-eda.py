!ls -lh ../input/ashrae-energy-prediction/
import os

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import altair as alt

from altair.vega import v5

from IPython.display import HTML



%matplotlib inline

plt.rc('figure', figsize=(15.0, 8.0))
%%time

root = '../input/ashrae-energy-prediction/'

train_df = pd.read_csv(root + 'train.csv')

weather_train_df = pd.read_csv(root + 'weather_train.csv')

test_df = pd.read_csv(root + 'test.csv')

weather_test_df = pd.read_csv(root + 'weather_test.csv')

building_meta_df = pd.read_csv(root + 'building_metadata.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
train_df.head()
print(f'There are {train_df.shape[0]} rows in train data.')

print(f"There are {train_df['meter'].nunique()} distinct meters in train data.")

print(f"There are {train_df['building_id'].nunique()} distinct buildings in train data, same as the number of rows in building_meta data")
building_meta_df.head()
print(f'There are {building_meta_df.shape[0]} rows in building meta data.')
weather_train_df.head()
print(f'There are {weather_train_df.shape} rows in weather train data.')
test_df.head()
print(f'There are {test_df.shape[0]} rows in test data.')

print(f"There are {test_df['meter'].nunique()} distinct meters in test data.")

print(f"There are {test_df['building_id'].nunique()} distinct buildings in test data.")
sorted(train_df['building_id'].unique()) == sorted(test_df['building_id'].unique())
weather_test_df.head()
print(f'There are {weather_test_df.shape} rows in weather test data.')
sample_submission.head()
print(f'There are {sample_submission.shape} rows in sample submission file.')
train_df = train_df.merge(building_meta_df, on='building_id', how='left')

train_df = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')
train_df.describe()
fig, ax = plt.subplots(figsize=(10, 10))

plot = sns.countplot(y="meter", data=train_df, palette=['navy', 'darkblue', 'blue', 'dodgerblue']).set_title('Meter count', fontsize=16)

plt.yticks(fontsize=14)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Meter number", fontsize=15)

plt.show(plot)
fig, ax = plt.subplots(figsize = (18, 6))

plt.subplot(1, 2, 1);

plt.hist(train_df['meter_reading']);

plt.title('Basic meter_reading histogram');

plt.subplot(1, 2, 2);

sns.violinplot(x='meter', y='meter_reading', data=train_df);

plt.title('Violinplot of meter_reading by meter');
train_df.query('meter_reading==0').shape, train_df.shape
def plot_dist(meter, clip_from):

    '''Plots distribution of non zero train data vs meter number''' 

    df = train_df.query(f'meter=={meter} and meter_reading!=0')

    fig, ax = plt.subplots(figsize = (18, 6))

    plt.subplot(1, 3, 1);

    plt.hist(df['meter_reading']);

    plt.title('Basic meter_reading histogram');

    plt.subplot(1, 3, 2);

    sns.violinplot(x='meter', y='meter_reading', data=df);

    plt.title('Violinplot of meter_reading by meter');

    plt.subplot(1, 3, 3);

    sns.violinplot(x='meter', y='meter_reading', data=df.query(f'meter_reading < {clip_from}'));

    plt.title('*Clipped* Violinplot of meter_reading by meter');
plot_dist(0, 2000)
plot_dist(1, 10000)
plot_dist(2, 10000)
plot_dist(3, 2000)
# Compute the correlation matrix

corr = train_df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
from statsmodels.tsa.seasonal import seasonal_decompose
building_id = 400

building = train_df.query(f'building_id == {building_id} and meter==0')
decomposition = seasonal_decompose(building['meter_reading'].values, freq=24)

decomposition.plot();
weather_train_df.columns
def plot_kde(column):

    plot = sns.jointplot(x=train_df[column][:10000], y=train_df['meter_reading'][:10000], kind='kde', color='blueviolet')

    plot.set_axis_labels('meter', 'meter_reading', fontsize=16)

    plt.show()
def plot_dist_col(column):

    '''plot dist curves for train and test weather data for the given column name'''

    fig, ax = plt.subplots(figsize=(10, 10))

    sns.distplot(weather_train_df[column].dropna(), color='darkorange', ax=ax).set_title(column, fontsize=16)

    sns.distplot(weather_test_df[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

    plt.xlabel(column, fontsize=15)

    plt.legend(['train', 'test'])

    plt.show()
plot_dist_col('air_temperature')
plot_dist_col('cloud_coverage')
plot_dist_col('dew_temperature')
plot_dist_col('precip_depth_1_hr')
plot_dist_col('sea_level_pressure')
plot_dist_col('wind_direction')
plot_dist_col('wind_speed')
building_meta_df.head()
fig, ax = plt.subplots(figsize=(10, 10))

plot = sns.countplot(y="primary_use", data=building_meta_df, 

                     palette='YlGn').set_title('Primary category of activities for the building', fontsize=16)

plt.yticks(fontsize=14)

plt.xlabel("Count", fontsize=15)

plt.ylabel("Building use type", fontsize=15)

plt.show(plot)
fig, ax = plt.subplots(figsize=(10, 10))

sns.distplot(building_meta_df['square_feet'], color='indigo', 

             ax=ax).set_title('Gross floor area of the building', fontsize=16)

plt.xlabel('square_feet', fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

sns.distplot(building_meta_df['year_built'].dropna(), color='crimson', 

             ax=ax).set_title('Year building was opened', fontsize=16)

plt.xlabel('year_built', fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize=(10, 10))

sns.distplot(building_meta_df['floor_count'].dropna(), color='crimson', 

             ax=ax).set_title('Number of floors of the building', fontsize=16)

plt.xlabel('floor_count', fontsize=15)

plt.show()