# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")
print(data.head(5))
print(list(data.columns))
print("hotel value counts\n",data['hotel'].value_counts())



print("Reservation value counts\n",data['reservation_status'].value_counts())



print("meal value counts\n", data['meal'].value_counts())

print("country value counts\n", data['country'].value_counts())
data.describe()
data.info()
data.isnull().sum()
# fill nan values

nan_fields = {"children": 0.0,"country": "Unknown", "agent": 0, "company": 0}

data_wo_na = data.fillna(nan_fields)

print(data_wo_na.info())
guests_0 = list(data_wo_na.loc[(data_wo_na['adults']==0)& (data_wo_na['children']==0)&(data_wo_na['babies']==0)].index)

data_wo_guests = data_wo_na.drop(data_wo_na.index[guests_0])

print(data_wo_guests.isnull().sum())

data_wo_guests.info()
# dropping 'is_canceled' field

is_cancel = list(data_wo_guests.loc[(data_wo_guests['is_canceled']==0)].index)

data_wo_cancel = data_wo_guests.drop(is_cancel)

country_df = pd.DataFrame(data_wo_cancel["country"].value_counts())

guest_percent = country_df['country']/country_df['country'].sum()

# percentage of guests from each country

country_df['guest_percent'] = guest_percent*100 

print(country_df)
label = list(country_df['country'].index.unique())

print(label)
import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

#

plt.figure(figsize=(10,7))

plt.pie(country_df['guest_percent'],labels = label, autopct='%1.1f%%', shadow=True)
#guest_map = px.choropleth(country_df['guest_percent'],

#                     locations=country_df['country'],

#                     color=country_df["guest_percent"], 

#                     hover_name=country_df['country'], 

#                     color_continuous_scale=px.colors.sequential.Jet,

#                     title="Home country of guests")

# guest_map.show()
guest = data_wo_cancel['adults']+data_wo_cancel['children']

data_wo_cancel['guest_total'] = guest

 

resort = data_wo_cancel[data_wo_cancel['hotel']=='Resort Hotel']

city = data_wo_cancel[data_wo_cancel['hotel']=='City Hotel']



print(resort.info())

# plt.figure(figsize=(10,7))

# plt.plot(resort['arrival_date_month'], resort['guest_total'], '.r', label='resort hotel')

# plt.plot(city['arrival_date_month'], city['guest_total'], '.c', label='city hotel')

# plt.legend(loc=1)

# plt.grid()

# #plt.ylim([0, 10])

# plt.show()
# create plot

fig, ax = plt.subplots()

month = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

index = resort['arrival_date_month'].map(month)

bar_width = 0.35

opacity = 0.5



b1 = plt.bar(index,resort['guest_total'],0.35,alpha=opacity,color='b',label='resort')

b2 = plt.bar(index+bar_width,city['guest_total'],0.35,alpha=opacity,color='g',label='city')



plt.xlabel('Month')

plt.ylabel('guest')

plt.title('Number of guest in hotels')

# plt.xticks(index+bar_width)

plt.legend()



plt.tight_layout()

plt.show()
# create plot

fig, ax = plt.subplots()

month = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

index = resort['arrival_date_month'].map(month)

bar_width = 0.35

opacity = 0.5



b1 = plt.bar(index,resort['previous_bookings_not_canceled'],0.35,alpha=opacity,color='b',label='Non-cancellations')

b2 = plt.bar(index+bar_width,resort['previous_cancellations'],0.35,alpha=opacity,color='g',label='Cancellations')



plt.xlabel('Month')

plt.ylabel('status')

plt.title('Cancellation stats of resort')

plt.xticks(index+bar_width,('1','2','3','4','5','6','7','8','9','10','11','12'))

plt.legend()



plt.tight_layout()

plt.show()
# create plot

fig, ax = plt.subplots()

month = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

index = city['arrival_date_month'].map(month)

bar_width = 0.35

opacity = 0.5



b1 = plt.bar(index,city['previous_bookings_not_canceled'],0.35,alpha=opacity,color='b',label='Non-cancellations')

b2 = plt.bar(index+bar_width,city['previous_cancellations'],0.35,alpha=opacity,color='g',label='Cancellations')



plt.xlabel('Month')

plt.ylabel('status')

plt.title('Cancellation stats of city')

plt.xticks(index+bar_width,('1','2','3','4','5','6','7','8','9','10','11','12'))

plt.legend()



plt.tight_layout()

plt.show()
plt.plot(city['reserved_room_type'],city['guest_total'],'.r',alpha=0.2,markersize=20,label='city-reserved')

plt.plot(city['assigned_room_type'],city['guest_total'],'.g',alpha=0.2,markersize=15,label='city-assigned')

plt.plot(resort['reserved_room_type'],resort['guest_total'],'.b',alpha=0.2,markersize=8,label='resort-reserved')

plt.plot(resort['assigned_room_type'],resort['guest_total'],'.c',alpha=0.2,markersize=4,label='resort-assigned')

plt.legend(loc=1, markerscale=4)

plt.grid()
resort.info()
##  

# city['pp']   = city['adr']/(city['adults']+city['children'])

# resort['pp'] = resort['adr']/(resort['adults']+resort['children'])

#

# boxplot:

plt.figure(figsize=(12, 8))

sns.boxplot(x="reserved_room_type",

            y="adr",

            hue="hotel",

            data=city, 

            hue_order=["City Hotel", "Resort Hotel"],

            fliersize=0)

plt.title("Price of room types per night and person", fontsize=16)

plt.xlabel("Room type", fontsize=16)

plt.ylabel("Price [EUR]", fontsize=16)

plt.legend(loc="upper right")

plt.ylim(0,500)

plt.show()