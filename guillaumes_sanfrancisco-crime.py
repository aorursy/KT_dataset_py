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
path = '/kaggle/input/sanfranciso-crime-dataset/Police_Department_Incidents_-_Previous_Year__2016_.csv'

df = pd.read_csv(path)

df.head()
df.Date.min(), df.Date.max()
df.shape
df.Descript.nunique()
from collections import Counter

Counter(df.Category).most_common()
dict_day = dict(Counter(df.DayOfWeek).most_common())

import matplotlib.pyplot as plt
x_axis = [d[:3] for d in dict_day.keys()]
plt.bar(x_axis, dict_day.values())
BBox = [df.Y.min(), df.Y.max(), df.X.min(), df.X.max()]

BBox
sf_map = plt.imread('/kaggle/input/sanfranciscomappng/map.png')
def plot_crime(df):

    """df should contain X and Y columns"""

    fig, ax = plt.subplots(figsize = (8,7))

    ax.scatter(df.Y, df.X, zorder=1, alpha= 0.2, c='b', s=10)

    ax.set_title('Plotting Spatial Data on San Francisco Map')

    ax.set_xlim(BBox[0],BBox[1])

    ax.set_ylim(BBox[2],BBox[3])

    ax.imshow(sf_map, zorder=0, extent=BBox, aspect='equal')

plot_crime(df)
df['day'] = [date[3:5] for date in df['Date']]

df['month'] = [date[:2] for date in df['Date']]

df['year'] = [date[6:10] for date in df['Date']]
df.head()
df_jan = df[df['month'] == '01']
plot_crime(df_jan)
df_jan['day'] = df_jan['day'].astype('int')

df_jan_1_15 = df_jan[df_jan['day'] <= 15]
plot_crime(df_jan_1_15)
df_suicide = df[df['Category'] == 'SUICIDE']

plot_crime(df_suicide)