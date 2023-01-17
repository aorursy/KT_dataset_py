# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
from datetime import datetime


df = pd.read_csv('../input/procurement-notices.csv')
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# number of calls currently out
# cells with NA deadline are currently out
df['deadline_date'] = pd.to_datetime(df['deadline_date'])
print((df[df['deadline_date'] > datetime.now()] | df[df['deadline_date'].isna()]).count()['deadline_date'])

current_calls = df[df['deadline_date'] > datetime.now()]
    






# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

% matplotlib inline
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

def hex_to_rgb(h):
    """ helper function converts hex string into rgb np.array """
    h = h.lstrip('0x')
    array = np.array(tuple(int(h[i:i+2], 16) for i in (0, 2 ,4)))
    return array / 255


def get_color(value, max_value):
    """ helper function return the color for the map """
    red = '0xff0000'
    white = '0xffffff'
    if value == 1:
        return hex_to_rgb(white)
    color = hex(int((value / max_value) * int(red, 16) + (1 - (value / max_value)) * int(white, 16)))
    return hex_to_rgb(color)


# dist. by country
calls_by_country = pd.DataFrame(current_calls.groupby('country_code').count()['id'])
calls_by_country.rename(columns={'id': 'number_of_bids'}, inplace=True)

# join data to a map and plot the map
countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
max_value = calls_by_country['number_of_bids'].max()
plt.rcParams["figure.figsize"] = (20, 10)
ax = plt.axes(projection=ccrs.PlateCarree())
for country in shpreader.Reader(countries_shp).records():
    code = country.attributes['ISO_A2']
    if not code in calls_by_country.index:
        color = get_color(1, 0)
    else:
        value = calls_by_country.loc[calls_by_country.index == code, 'number_of_bids'][0]
        color = get_color(value, max_value)
    ax.add_geometries([country.geometry], ccrs.PlateCarree(), facecolor=color, label=country.attributes['ISO_A2'],
                      linewidth=0.3, edgecolor='black')

# dist. of due dates
due_dates = df[df['deadline_date'] > datetime.now()].groupby('deadline_date').count().dropna()

# plot number of bids due per date
due_dates.rename(columns={'id': 'n'}, inplace=True)
plt.plot(due_dates.index, due_dates['n'])