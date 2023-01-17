import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import os
import folium
print(os.listdir("../input"))
File = pd.read_csv("../input/Attacks on Political Leaders in Pakistan.csv", encoding = "ISO-8859-1")
File['City'] = File['City'].replace(['ATTOCK'],'Attock')
data = File.loc[:,['City','Latitude','Longititude']]
data.loc[2,:] = data.loc[15]
data.loc[48,:] = data.loc[15]
data.loc[43,:] = data.loc[9]
# Make an empty map
#m = folium.Map(location=[37.76, -122.45], tiles="Mapbox Bright", zoom_start=2)
m = folium.Map(location=[30, 70], tiles='Mapbox Bright', zoom_start=5) 
# I can add marker one by one on the map
# Latitude
# Longititude
for i in range(0,len(data)):
    folium.Marker([data.iloc[i]['Latitude'], data.iloc[i]['Longititude']], popup=data.iloc[i]['City']).add_to(m)
m
data = data.round(2)
Count = data.groupby(['City'])['City'].count()
data_dup_rem = data.drop_duplicates()
sorted_data = data_dup_rem.sort_values(by=['City'], ascending = True)
sorted_data['Count'] = np.array(Count)
sorted_data
#print(sorted_data.shape)
#print(Count.shape)
Heat_map_data = sorted_data.drop(['City'], axis = 1)
np.array(Heat_map_data).tolist() 
from folium.plugins import HeatMap

m = folium.Map([30., 70.], tiles='stamentoner', zoom_start=5)

HeatMap(np.array(Heat_map_data).tolist()).add_to(m)
m
Day = (File.groupby(['Day']).Day.count())
plt.figure(figsize=(10,5))
plt.title('No. of incidents w.r.t Day of the week')
Day.plot(kind='bar',fontsize=15)
plt.rc('axes',labelsize=20)
plt.rc('axes',titlesize=20) 
plt.grid(True)
plt.show()
Time = (File.groupby(['Time']).Time.count())
plt.figure(figsize=(10,5))
plt.title('No. of incidents w.r.t Time of the day')
Time.plot(kind='bar',fontsize=15)
plt.rc('axes',labelsize=20)
plt.rc('axes',titlesize=20) 
plt.grid(True)
plt.show()
File = File.rename(columns = {'Location Category': 'Location_Category'})
Loc = (File.groupby(['Location_Category']).Location_Category.count())
plt.figure(figsize=(10,5))
plt.title('No. of incidents w.r.t Location Category')
Loc.plot(kind='bar',fontsize=15)
plt.rc('axes',labelsize=20)
plt.rc('axes',titlesize=20) 
plt.grid(True)
plt.show()
Causality = File.loc[:,['Politician','Date','City','Latitude','Longititude','Killed','Injured']]
Causality['Total'] = Causality['Killed']+ Causality['Injured']
Causality.sort_values(by=['Total'], ascending=False)
#tempdata = np.array(Causality.loc[:,['Latitude','Longititude','Total']]).tolist()
#m = folium.Map([30., 70.], tiles='stamentoner', zoom_start=6)

#HeatMap(tempdata).add_to(m)
#m