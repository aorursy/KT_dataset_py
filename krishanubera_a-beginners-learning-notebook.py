# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Imports
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
# This code is copied from another notebook (https://www.kaggle.com/thedatabeast/making-perfect-chai-and-other-tales )
# loading data

try:
    home = '/kaggle/input/chai-time-data-science'
    ds_episods = pd.read_csv(os.path.join(home, "Episodes.csv"))
    ds_description = pd.read_csv(os.path.join(home, "Description.csv"))
    ds_y_thumb = pd.read_csv(os.path.join(home, "YouTube Thumbnail Types.csv"))
    ds_anchor_thumb = pd.read_csv(os.path.join(home, "Anchor Thumbnail Types.csv"))
except:
    print("File names have been changed. Have a look at the dataset home page.")
ds_episods.head()
ds_episods.describe()
ds_episods.dtypes
df = ds_episods.describe().select_dtypes(['number']);
ax = df.loc[['mean','std']].T.plot.bar(figsize=(20,5));
ax.set_title('Episodes Statistics');
# Divide episode duration into minute range
duration='duration_minute_range'
minute_range=15
ds_episods[duration]=((ds_episods["episode_duration"]/(60*minute_range)).apply(np.ceil))*minute_range

# sort by duration_minute_range
ds_episods_by_duration = ds_episods.sort_values(by=[duration])
# create a display function
def display_by_duration(platform, excluded_cols, ax):

    display_cols = [col for col in ds_episods_by_duration if col.startswith(platform) and col not in excluded_cols ]
    # print(f" Columns to be displayed:{display_cols}")

    # display various response in bar
    ds_episods_by_duration.groupby(duration).sum().plot(y=display_cols, kind='bar', ax=ax);

fig,ax   = plt.subplots(nrows=5,figsize=(20,20))
fig.subplots_adjust(hspace=0.5)    

# Display episode count
ds_episods_by_duration[duration].value_counts().sort_index().plot(ax=ax[0], kind='bar');
ax[0].set_title('Count of Videos')

# Display youtube fields
excluded_cols = [duration, 'youtube_impressions', 'youtube_thumbnail_type', 'youtube_ctr', 'youtube_dislikes', 'youtube_comments', 'youtube_url']
display_by_duration('youtube', excluded_cols, ax[1])
ax[1].set_title('Youtube')
ax[1].xaxis.label.set_visible(False)

# Display Apple fields
excluded_cols = [duration]
display_by_duration('apple', excluded_cols, ax[2])
ax[2].set_title('Apple')
ax[2].xaxis.label.set_visible(False)


# Display spotify fields
excluded_cols = [duration]
display_by_duration('spotify', excluded_cols, ax[3])
ax[3].set_title('Spotify')
ax[3].set_xlabel('Duration')


# Display episode id by duration
ds_episods_by_duration.plot.bar('episode_id', duration, ax=ax[4])
ax[4].set_yticks(ds_episods[duration])
for a_duration in ds_episods[duration]:
    ax[4].axhline(a_duration, color='grey');
m_series = ds_episods[ds_episods['episode_id'].str.startswith('M')].notna().sum()
e_series = ds_episods[ds_episods['episode_id'].str.startswith('E')].notna().sum()

df = pd.DataFrame({'M':m_series, 'E':e_series} )
df.plot.bar(figsize=(10,5));