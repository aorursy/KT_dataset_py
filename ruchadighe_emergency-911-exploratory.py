# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import folium

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
emergency_calls= pd.read_csv('../input/911.csv')
emergency_calls['timeStamp']=pd.to_datetime(emergency_calls['timeStamp'])
emergency_calls[['Title Type', 'Title Detail']] = emergency_calls.title.str.split(': ', expand = True)
emergency_calls['year']=emergency_calls['timeStamp'].dt.year
emergency_calls['month']=emergency_calls['timeStamp'].dt.month
emergency_calls['hours']=emergency_calls['timeStamp'].dt.hour
emergency_calls['Time of day']= np.where(emergency_calls['hours']<=6,'morning',np.where(emergency_calls['hours']<= 12,'noon',np.where(emergency_calls['hours']<= 18,'evening','night')))

plt.hist(emergency_calls.lat)
plt.title("Histogram of Latitute")
plt.xlabel("Latitute")
plt.ylabel("Frequency")
plt.show()
plt.hist(emergency_calls.lng)
plt.title("Histogram of Longitude")
plt.xlabel("Longitude")
plt.ylabel("Frequency")
plt.show()
tab_zip=pd.crosstab(index=emergency_calls['zip'], columns='count', colnames=[''])
#print(tab_zip)
#tab_zip.plot.bar()
tab_title_type=pd.crosstab(index=emergency_calls['Title Type'], columns='count', colnames=['count'])
#tab_title_type.plot.bar()
tab_title_type.reset_index(inplace = True)
#print(tab_title_type)
plt.pie(tab_title_type['count'],labels=tab_title_type['Title Type'], shadow=False, colors=["#E13F29", "#D69A80", "#D63B59"],
         explode=(0, 0, 0.15),startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()
#plt.show()

tab_title_detail=pd.crosstab(index=emergency_calls['Title Detail'], columns='count', colnames=[''])
#tab_title_detail
#tab_title_detail.plot.bar()
tab_year=pd.crosstab(index=emergency_calls['year'], columns='count', colnames=['count'])
#tab_year.plot.bar()
tab_year.reset_index(inplace = True)
#print(tab_year)
plt.pie(tab_year['count'],labels=tab_year['year'], shadow=False, colors=["#E13F29", "#D69A80", "#D63B59", "#AE5552"],
         explode=(0, 0,0, 0.15),startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()
#plt.show()
tab_month=pd.crosstab(index=emergency_calls['month'], columns='count', colnames=[''])
#print(tab_month)
#tab_month.plot.bar()
tab_time=pd.crosstab(index=emergency_calls['Time of day'], columns='count', colnames=['count'])
#tab_time.plot.bar()
tab_time.reset_index(inplace = True)
#print(tab_time)
plt.pie(tab_time['count'],labels=tab_time['Time of day'], shadow=False, colors=["#E13F29", "#D69A80", "#D63B59", "#AE5552"],
         explode=(0, 0,0, 0.15),startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.tight_layout()
#plt.show()
pd.crosstab(emergency_calls['Title Type'],emergency_calls['year'])
pd.crosstab(emergency_calls['Title Detail'],emergency_calls['Title Type'])
colors = {'EMS' : 'red', 'Fire' : 'blue','Traffic':'green'}

map_osm = folium.Map(location=[40.742, -73.956], zoom_start=11)

emergency_calls.apply(lambda row:folium.CircleMarker(location=[row["lat"], row["lng"]], 
                                              radius=10, fill_color=colors[row['Title Type']])
                                             .add_to(map_osm), axis=1)
map_osm