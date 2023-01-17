# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

import datetime

from mpl_toolkits.basemap import Basemap





%matplotlib inline

plt.rcParams['figure.figsize'] = (8.0, 6.0)



# Any results you write to the current directory are saved as output.
# Load data



tapsi_data = pd.read_csv('../input/tap30/tap30-sample.csv')



tapsi_data.head()
tapsi_data.info()
# Clean Data



# Shwo sum of NaN Values

tapsi_data.isnull().sum()



# Clean Data

tapsi_data.dropna()



# Show Sample of data

tapsi_data.sample(5)
# number of rows in the dataset



len(tapsi_data)
# Checking the maximum date in the dataset



tapsi_data.createdAt.max()
# converting pickup_date column to datetime

tapsi_data.createdAt = pd.to_datetime(tapsi_data.createdAt,format = '%Y-%m-%d %H:%M:%S')





# extracting the hour from "createAt" column and putting it into a new column name 'Hour'

tapsi_data['Hour'] = tapsi_data.createdAt.apply(lambda x: x.hour)



# extracting the minute from "createAt" column and putting in into a new column name 'minute'

tapsi_data['Minute'] = tapsi_data.createdAt.apply(lambda x: x.minute)



# extracting the Month from "createAt" column and putting it into a new column named 'Month'

tapsi_data['Month'] = tapsi_data.createdAt.apply(lambda x:x.month)



# extracting the day from "creatAt" column and putting it into a new column named 'month'

tapsi_data['Day'] = tapsi_data.createdAt.apply(lambda x:x.day)
#browse updated dataset

tapsi_data.head()
# Tapsi pickups by the month



sns.set(style="white", color_codes=True)

ax = sns.countplot(x="Month", data = tapsi_data)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Month',fontsize = 12)

ax.set_ylabel('Count of tapsi pickups',fontsize = 12)

ax.set_title('Tapsi pickups by the Month',fontsize = 16)

ax.tick_params(labelsize = 8)

plt.show()
# tapsi pickups by the hour



sns.set(style="white", color_codes=True)

ax = sns.countplot(x = 'Hour',data = tapsi_data )



ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Hour of the Day',fontsize = 12)

ax.set_ylabel('Count of Pickup',fontsize = 12)

ax.set_title('tapsi pickups By the Hour',fontsize = 16)

ax.tick_params(labelsize = 8)

plt.show()
tapsi_data['Hour'].hist(bins = 24,color = 'k',alpha = 0.5)

plt.xlim(0,23)

plt.title('pickup Hours')

plt.ylabel('Total')

plt.xlabel('Time in Hours')

plt.axvline(8,color = 'r',linestyle = '--')

plt.axvline(17,color='r',linestyle = '--')
tapsi_data['DayOfWeek'] = tapsi_data['createdAt'].dt.weekday_name

tapsi_weekdays = tapsi_data.pivot_table(index = ['DayOfWeek'],

                                       values='id',

                                       aggfunc='count')

tapsi_weekdays.plot(kind='bar',figsize=(8,6))

plt.ylabel('total pickup')

plt.title('Week of day')
tapsi_data.head()
# group The data By Weekday and hour



summary = tapsi_data.groupby(['DayOfWeek','Hour'])['createdAt'].count()
# reset index

summary = summary.reset_index()



#convert to dataframe

summary = pd.DataFrame(summary)



#browse data

summary.head()

#rename last column

summary = summary.rename(columns = {'createdAt':'Counts'})
sns.set_style('whitegrid')

ax = sns.pointplot(x="Hour", y="Counts", hue="DayOfWeek", data=summary)

handles,labels = ax.get_legend_handles_labels()

#reordering legend content

handles = [handles[1], handles[5], handles[6], handles[4], handles[0], handles[2], handles[3]]

labels = [labels[1], labels[5], labels[6], labels[4], labels[0], labels[2], labels[3]]

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax.set_xlabel('Hour', fontsize = 12)

ax.set_ylabel('Count of tapsi Pickups', fontsize = 12)

ax.set_title('Hourly tapsi Pickups By Day of the Week', fontsize=16)

ax.tick_params(labelsize = 8)

ax.legend(handles,labels,loc=0, title="Legend", prop={'size':8})

ax.get_legend().get_title().set_fontsize('8')

plt.show()
print(tapsi_data.originLat.max(),tapsi_data.originLat.min())

print(tapsi_data.originLon.max(),tapsi_data.originLon.min())
plt.scatter(range(tapsi_data.shape[0]),np.sort(tapsi_data.originLat.values))

plt.title("Latitude distribution")

plt.show()
plt.scatter(range(tapsi_data.shape[0]),np.sort(tapsi_data.originLon.values))

plt.title('Longitude distribution')

plt.show()
import folium

tapsi_weekday_map = tapsi_data.loc[tapsi_data['DayOfWeek']!='x']

tapsi_weekday_map.head()
tapsi_time_map = tapsi_weekday_map.loc[tapsi_weekday_map['Hour'] != 0]

tapsi_time_map.shape
lat = 30.066936

lon = 57.5

ir_map  = folium.Map(location=[lat,lon],zoom_start=5)



# add markers to map



for lat,lng in zip(tapsi_time_map['originLat'],tapsi_time_map['originLon']):

    

    folium.CircleMarker([lat,lng],

                       radius = 5,

                       color = 'blue',

                       fill = True,

                       fill_color = '#3186cc',

                       fill_opacity = 0.7,

                       parse_html = False).add_to(ir_map)

ir_map
session_labels = ['Late Night','Early Morning','Late Morning','Afternoon','Evening','Night']



df_tapsi = tapsi_data[['rideStatus','createdAt']]

df_tapsi = df_tapsi.assign(session = pd.cut(df_tapsi.createdAt.dt.hour,[-1,4,8,12,16,20,24],labels = session_labels))

df_tapsi.sample(5)
plt.style.use('ggplot')

colors = ['#CC2529','#8E8D8D','008000']

df_tapsi.groupby(['session','rideStatus']).rideStatus.count().unstack().plot.bar(legend = True,figsize = (15,10))

plt.title('Total Count Of all trip Status')

plt.xlabel('Sessions')

plt.ylabel('Total Count of trip Status')

plt.show()
# tapsi_data.head()

tapsi_data['Day'] = tapsi_data['createdAt'].dt.weekday_name

fig,ax = plt.subplots(figsize = (15,8))

ax = sns.barplot(x = 'Day',y = 'driverETA' , data = tapsi_data,ci = False)
#TODO: Make Distance Column