import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import ast
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.basemap import Basemap

%matplotlib inline
colnames = ["name", "desc", "fee", "type", "subject", "age", "service_area", "sponsor",
            "dt_start", "dt_end", "loc_name", "loc_address", "contact_name", "contact_number", 
            "contact_email", "web", "official_office", "official_name", "council", "ref_id"]
events_data = pd.read_csv("../input/whats-happening-la-calendar-dataset.csv", names=colnames, header=0, usecols=range(18))
events_data["dt_start"] = pd.to_datetime(
    events_data["dt_start"], infer_datetime_format=True, errors="coerce")
events_data["dt_end"] = pd.to_datetime(
    events_data["dt_end"], infer_datetime_format=True, errors="coerce")
events_data.loc[events_data["age"] == "All ages", "age"] = "All"
events_data.head()
events_data_fee = events_data["fee"].apply(lambda x: str(x).strip().lower())
events_data_fee.replace("nan", "unknown", inplace=True)
events_data_fee.replace("none", "unknown", inplace=True)
counts = events_data_fee.value_counts()
# filter fee values (show only ones with more than 1 occurence)
fee_counts = events_data_fee[events_data_fee.isin(counts[counts > 1].keys())].value_counts()
# Fee Required
fig = plt.figure(figsize=(8, 8))
plt.pie(fee_counts.values, labels=fee_counts.keys(), autopct='%1.2f%%', pctdistance=1.2, labeldistance=1.35, radius=0.9)
plt.title("Fee required")
plt.show()
# parse addresses, longitude and latitude
parsed_addrs = events_data["loc_address"].fillna("{}").apply(ast.literal_eval)
human_addrs = parsed_addrs.apply(lambda x: x.get("human_address", "{}"))
human_addrs = human_addrs.apply(ast.literal_eval)
addr_lat = pd.to_numeric(parsed_addrs.apply(lambda x: x.get("latitude", np.nan)), errors="coerce")
addr_long = pd.to_numeric(parsed_addrs.apply(lambda x: x.get("longitude", np.nan)), errors="coerce")
# TOP 20 cities (by number of events)

cities = human_addrs.apply(lambda x: x["city"] if "city" in x else "")
cities.value_counts()[1:21].plot("bar", title="TOP 20 cities (by number of events)", figsize=(20,8))
plt.show()
# Year when event starts

dt_start = events_data[~events_data["dt_start"].isnull()]["dt_start"]
val_counts = dt_start.dt.year.astype("int").value_counts()
print("Years when events start:")
print(val_counts)
plt.figure(figsize=(12,6))
plt.bar(val_counts.index, val_counts.values)
plt.title("Year when events start")
plt.show()
# TOP 10 city councils with highest number of events
events_data["official_office"].value_counts()[:10].plot.bar(title="Top 10 city councils (by number of events)", figsize=(9, 5))
plt.show()
# events by age
events_data["age"].value_counts()[:10].plot.bar(title="Events by age", figsize=(9, 5))
plt.show()
# At first I wanted to create graphs for:
# - number of events scheduled for today
# - number of events currently happening
# - distribution of events in next 7 days
sum(events_data["dt_end"] > datetime.now())

# but as you can see, there is no events that finish in the future -> no current events -> no nice graphs -> sad :(
# draw events on a map

lowerleft = (33.46, -118.9)
upperright = (34.49, -117.81)
fig = plt.figure(figsize=(10, 10))
m = Basemap(projection='cyl', resolution='i', 
            llcrnrlat=lowerleft[0], urcrnrlat=upperright[0],
            llcrnrlon=lowerleft[1], urcrnrlon=upperright[1])
m.drawcoastlines(color='black')
m.drawstates(color='gray')
m.fillcontinents(color="lightgreen", lake_color='#DDEEFF')
m.drawrivers()
m.drawmapboundary()

# plot Los Angeles Downtown, for easier orientation
lon = -118.12986
lat = 34.043352
x,y = m(lon, lat)
m.plot(x, y, 'bo', markersize=10)
plt.text(x, y, "Los Angeles Downtown")

# plot events
m.scatter(addr_long, addr_lat, latlon=True, marker='o', color='r', zorder=5, s=3)
plt.title("Locations of events")
plt.show()
