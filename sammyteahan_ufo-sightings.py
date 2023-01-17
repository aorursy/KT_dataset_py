# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Data
import numpy as np
import pandas as pd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
ufo_sightings_path = '../input/scrubbed.csv'
ufo_data = pd.read_csv(ufo_sightings_path)
ufo_data.head()
ufo_data.columns
ufo_data.shape
"""
Let's take a quick look at our null values
"""
null_values = ufo_data.isnull().sum() # sum null vals for each column
print('Null Values:')
print(null_values.sort_values(ascending=False))

print('\n')
print('Null Value Percentages:')
null_percentages = (null_values / len(ufo_data)) * 100
print(round(null_percentages.sort_values(ascending=False), 2))
ufo_data['country'].value_counts()
us_ufo_data = ufo_data[ufo_data.country == 'us']
us_ufo_data.head()
state_sightings = us_ufo_data.state.value_counts() # grouped by state
state_names = state_sightings.index # grab index for x vals (first col values are state names)

us_ufo_data.state.value_counts()
sighting_frequencies = state_sightings.get_values() # y vals

def plot_state_frequencies():
    with sns.axes_style('darkgrid'):
        f, ax = plt.subplots(figsize=(20, 10))
        plt.xticks(rotation = 60)
        x = state_names
        y = sighting_frequencies
        
        sns.barplot(x=x, y=y, palette="GnBu_d", ax=ax)
        
plot_state_frequencies()
"""
First we need to clean our `datetime` column
and do some conversions so we can work with the values
"""
# extract dates to a separate thing for use later
ufo_dates = ufo_data.datetime.str.replace('24:00', '00:00') # a python datetime value needs 24:00 to be mapped to 00:00
ufo_dates = pd.to_datetime(ufo_dates, format='%m/%d/%Y %H:%M') # now mapping our datetime column will work

# remap dataframe values
ufo_data['datetime'] = ufo_data.datetime.str.replace('24:00', '00:00') # a python datetime value needs 24:00 to be mapped to 00:00
ufo_data['datetime'] = pd.to_datetime(ufo_data['datetime'], format='%m/%d/%Y %H:%M') # now mapping our datetime column will work

ufo_data.head()
"""
we created a separate variable above with just the dates so
it's easier to get at our `dt` object which consequently
makes it much easier to get `year`, `month`, or `day` data.
We could still also use `ufo_data['datetime'].dt.year` here
"""
ufo_year = ufo_dates.dt.year # exclusively get year Series from ufo_dates

year_counts = ufo_year.value_counts() # group data by year
year_labels = year_counts.index # x labels
year_frequencies =  year_counts.get_values() # y values

def plot_yearly_frequency():
    with sns.axes_style('darkgrid'):
        f, ax = plt.subplots(figsize=(20, 10))
        plt.title('UFO Sightings by Year')
        plt.xticks(rotation = 60)
        x = year_labels[:60] # last 60 values seem to be more relevant, the aliens didn't start visiting earth until the 50's
        y = year_frequencies[:60] # last 60 values seem to be more relevant, the aliens didn't start visiting earth until the 50's
        sns.barplot(x=x, y=y, palette="GnBu_d", ax=ax)

plot_yearly_frequency()
ufo_months = ufo_dates.dt.month # exclusively get months in a Series

month_data = (ufo_months.value_counts()).sort_index() # sort the index to put back into order for months in a year
months_labels = month_data.index # x labels
months_frequencies = month_data.get_values() # y vals

def plot_month_frequencies():
    with sns.axes_style('darkgrid'):
        f, ax = plt.subplots(figsize=(20, 10))
        x = months_labels
        y = months_frequencies
        ax.set_title('Global UFO Sightings by Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('# Sightings')

        sns.barplot(x=x, y=y, palette="GnBu_d", ax=ax)

plot_month_frequencies()
