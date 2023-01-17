import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from datetime import datetime

import seaborn as sns 



%matplotlib inline 



sns.set_context('notebook',font_scale=1.5)

sns.set_style('ticks')
data = pd.read_csv('../input/party_in_nyc.csv')
data.head()
import folium

from folium.plugins import MarkerCluster



bars_map = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()],zoom_start=10)

mc = MarkerCluster()

for ind,row in bar_locs.iterrows():

    mc.add_child(folium.CircleMarker(location=[row['Latitude'],row['Longitude']],

                        radius=1,color='#3185cc'))

bars_map.add_child(mc)

bars_map
complaints_by_city = data['Borough'].value_counts()

plt.figure(figsize=(8,6))

sns.barplot(y=complaints_by_city.index,x=complaints_by_city.values,palette=sns.color_palette('Blues_d'))

plt.xticks(rotation=45);
complaints_by_loc_type = data['Location Type'].value_counts()

plt.figure(figsize=(8,6))

sns.barplot(y=complaints_by_loc_type.index,x=complaints_by_loc_type.values,palette=sns.color_palette('Blues_d'))

plt.xticks(rotation=45);
# Remove the data where closed date is not available



dt = data[~(data['Closed Date'].isnull())]



dt['Created Date'] = dt['Created Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

dt['Closed Date'] = dt['Closed Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

dt['dtime'] = dt['Closed Date'] - dt['Created Date']

dt['dtime'] = dt['dtime'].apply(lambda x: x.total_seconds())



# There are some rows where closed date is weird (i.e. year is 1900), remove them 

dt = dt[dt['dtime'] >= 0]
sns.distplot(dt['dtime'].apply(np.log10))

plt.xlabel('$log_{10}(\Delta seconds)$')

plt.axvline(dt['dtime'].apply(np.log10).median(),color='grey',linestyle='--')

print("Median amount of hours to close the call: %.1f" % (np.power(10,dt['dtime'].apply(np.log10).median())/3600.))
dtlog = dt.copy()

dtlog['dtime'] = dt['dtime'].apply(np.log10)

plt.figure(figsize=(8,6))

sns.boxplot(x='dtime',y='Borough',data=dtlog,palette=sns.color_palette('Blues_d'))

plt.xlabel('$log_{10}(\Delta seconds)$')
