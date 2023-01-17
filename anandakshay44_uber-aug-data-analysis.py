# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import cm

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import datetime
file = '../input/uber-raw-data-aug14.csv'

data = pd.read_csv(file)

data.head()
data.info()
data['Date/Time'] = pd.to_datetime(data['Date/Time'], format="%m/%d/%Y %H:%M:%S")

data['dayofweek'] = data['Date/Time'].dt.dayofweek

data['weekday_name'] = data['Date/Time'].dt.weekday_name

data['day'] = data['Date/Time'].dt.day

data['hour'] = data['Date/Time'].dt.hour

data.head()
weekday_details = data.pivot_table(index=['dayofweek','weekday_name'],

                                  values='Base',

                                  aggfunc='count')

weekday_details.plot(kind='bar', figsize=(8,6))

plt.ylabel('Total no of Journeys')

plt.title('Journeys vs Week Day');
#journey per day of week

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('journey per day of week')

sns.countplot(data['dayofweek'])
#journey per day of whole month

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('journey per day of month')

sns.countplot(data['day'])
#journey per hour in a day

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('journey per hour in a day for month')

sns.countplot(data['hour'])
#grouping data by Weekday and hour

data_total = data.groupby(['day', 'hour'])['Date/Time'].count()

data_total = data_total.reset_index()

#converting to dataframe

data_total = pd.DataFrame(data_total)

data_total.head()
data_total=data_total.rename(columns = {'Date/Time':'Counts'})
sns.set_style('whitegrid')

ax = sns.pointplot(x="day", y="Counts", hue="hour", data=data_total)

handles,labels = ax.get_legend_handles_labels()

#reordering legend content

handles = [handles[1], handles[5], handles[6], handles[4], handles[0], handles[2], handles[3]]

labels = [labels[1], labels[5], labels[6], labels[4], labels[0], labels[2], labels[3]]

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Hour', fontsize = 12)

ax.set_ylabel('Count of Uber Pickups', fontsize = 12)

ax.set_title('Hourly Uber Pickups By Day of the Week in NYC (Aug 2014)', fontsize=16)

ax.tick_params(labelsize = 8)

ax.legend(handles,labels,loc=0, title="Legend", prop={'size':8})

ax.get_legend().get_title().set_fontsize('8')

plt.show()