import numpy as np
import pandas as pd

import ast
import time
import datetime

import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
vacant = pd.read_csv('../input/combined_complaints.csv')
vacant['Latitude/Longitude'].isnull().sum()
def lat(point):
    try:
        lat = ast.literal_eval(point)[0]
    except:
        lat = np.nan
    return lat
def lon(point):
    try:
        lat = ast.literal_eval(point)[1]
    except:
        lat = np.nan
    return lat
vacant['lon'] = vacant['Latitude/Longitude'].transform(lon)
vacant['lat'] = vacant['Latitude/Longitude'].transform(lat)
clean = vacant[(~vacant['lon'].isnull()) | (~vacant['lat'].isnull())]
clean = clean[clean['CSR Priority']<='4']
clean['CSR Priority'] = clean['CSR Priority'].astype(int)
hmap = folium.Map(location=[34, -118],tiles='Stamen Toner', zoom_start=10)
hmap
heat_data = [[row['lat'],row['lon'],5-row['CSR Priority']] for index, row in clean.iterrows()]
h = HeatMap(heat_data, radius=10, max_val=3, min_opacity=0.2).add_to(hmap)
h.save('Heatmap.html')
clean[clean['Date Received'].isnull()]
def datest_to_timestamp(series):
    res = []
    for st in series:
        res.append(datetime.datetime.strptime(st, "%m/%d/%Y"))
    return res
clean['Begin_date'] = clean['Date Received'].transform(datest_to_timestamp)
all_dates = clean['Begin_date'].values
all_dates = list(set(all_dates))
all_dates.sort()
all_dates[:4]
time_labels = [str(x)[:10] for x in all_dates]
data = [[[row['lat'], row['lon'], 5-row['CSR Priority']] for index, row in clean[clean['Begin_date'] == i].iterrows()] for i in all_dates]
def heat_map_time(start_date, end_date):
    # min_day is the date representation of the first index of the data array
    min_day = datetime.datetime.strptime(time_labels[0], "%Y-%m-%d")
    s_day = datetime.datetime.strptime(start_date, "%m/%d/%Y")
    e_day = datetime.datetime.strptime(end_date, "%m/%d/%Y")
    
    # the number of days relative to min_day are the indices in the data array we want
    s_idx = (s_day-min_day).days
    e_idx = (e_day-min_day).days
    
    # the data and times passed to HeatMapWithTime
    d = data[s_idx:e_idx+1]
    times = time_labels[s_idx:e_idx+1]
    
    #the Stamen Toner tile looks cool
    hmaptime = folium.Map(location=[34, -118], tiles='Stamen Toner',zoom_start=10)
    
    HeatMapWithTime(data=d,index=times).add_to(hmaptime)
    display(hmaptime)
heat_map_time('06/12/2013','08/11/2013')