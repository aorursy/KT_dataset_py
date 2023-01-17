# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data(2).csv')

data['from_date']=pd.to_datetime(data['from_date'])

data['to_date']=pd.to_datetime(data['to_date'])

data['booking_created']=pd.to_datetime(data['booking_created'])

data.set_index('booking_created')

data.sample(5)

data.info()
import folium

from folium.plugins import HeatMap,HeatMapWithTime

m=folium.Map([12.9,77.6],zoom_start=11)

HeatMap(data[['from_lat','from_long']].dropna().groupby(['from_lat', 'from_long']).sum().reset_index().values.tolist(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)

plt.figure(figsize=(40,40))

plt.subplot(2,2,1)

sns.countplot(data.package_id);plt.xticks(fontsize=18);plt.yticks(fontsize=18);plt.xlabel('Package type',fontsize=18)

plt.subplot(2,2,2)

sns.countplot(data.travel_type_id);plt.xticks(fontsize=18);plt.yticks(fontsize=18);plt.xlabel('Travel type',fontsize=18)

plt.subplot(2,2,3)

sns.countplot(data.Car_Cancellation);plt.xticks(fontsize=18);plt.yticks(fontsize=18);plt.xlabel('Cancellation',fontsize=18)

plt.subplot(2,2,4)

sns.countplot(data.vehicle_model_id);plt.xticks(fontsize=18);plt.yticks(fontsize=18);plt.xlabel('Vehicle model',fontsize=18)

plt.show()

data['booking_monthly']=data['booking_created'].dt.to_period('M')

data['booking_daily']=data['booking_created'].dt.to_period('D')

data.head()

plt.figure(figsize=(18,10))

sns.countplot(data.sort_values('booking_monthly').booking_monthly,

              hue=data.sort_values('booking_monthly').Car_Cancellation,

              palette='Set2')

plt.xlabel('Month')

plt.ylabel('Bookings done')

plt.title('Monthly demand of cabs')

plt.legend(['Completed','Cancelled'],loc='best')

plt.show()

data['booking_daily']=data['booking_daily'].astype(str)

new=data['booking_daily'].str.split('-',n=2,expand=True)

#data['booking_daily']=new[2]

#data['booking_daily']

plt.figure(figsize=(18,10))

sns.countplot(new[2])

plt.title('Daily demand of cabs')

plt.xlabel('Day of the month')

plt.ylabel('Bookings done')

plt.show()

data['booking_daily']=pd.to_datetime(data['booking_daily'])

data['booking_daily'].sample(5)

plt.figure(figsize=(18,10))

sns.countplot(data['booking_daily'].dt.day_name(),hue=data['Car_Cancellation'])

plt.title('Day wise demand of cabs')

plt.xlabel('Day of the week')

plt.ylabel('Bookings done')

plt.legend(['Completed','Cancelled'],loc='best')

plt.show()

new2=data['booking_created'].astype(str)

new2=new2.str.extract(pat='(..:..:..)')

new2=new2[0].str.split(':',n=3,expand=True)

plt.figure(figsize=(18,10))

sns.countplot(new2[0])

plt.xlabel('Hour')

plt.ylabel('Bookings done')

plt.title('Hourly demand of cabs')

plt.show()

m=folium.Map([12.9,77.6],zoom_start=11)

data['hour']=new2[0]

hour_list=[]

for hour in data.hour.sort_values().unique():

    hour_list.append(data.loc[data.hour==hour,['from_lat','from_long']].groupby(['from_lat','from_long']).sum().reset_index().values.tolist())

HeatMapWithTime(hour_list,radius=5,gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1: 'red'},

                min_opacity=0.5,max_opacity=0.8,use_local_extrema=True).add_to(m)

display(m)

x=pd.DataFrame(data.from_area_id.value_counts())

#data[data.from_area_id==393]

x=x.index[:20]

x

map=folium.Map([12.9,77.6],zoom_start=11)

for loc in x:

    folium.Marker([data[data.from_area_id==loc]['from_lat'].head(1).item(),

                   data[data.from_area_id==loc]['from_long'].head(1).item()],

                  popup='Area id: '+str(loc)).add_to(map)

display(map)
