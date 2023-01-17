import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
import math

df = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')
df.replace(['-1'], [np.nan], inplace=True)
df.drop(columns = ['Unnamed: 0'], inplace=True)

def get_salary_lb(text):
    if text != text:
        lb = np.nan
    else:
        text = text.replace("$", "")
        lower, _ = text.split('-')
        lb = int(lower.replace("K", ""))
    return lb


def get_salary_ub(text):
    if text != text:
        ub = np.nan
    else:
        text = text.replace("$", "")
        _, upper = text.split('-')
        upper, _ = upper.split("(")
        upper = upper.replace("K", "")
        ub = int(upper.strip())
    return ub


def clean_name(text):
    name = np.nan
    if text == text:
        if '\n' in text:
            name, _ = text.split('\n')
    return name
    
df['salary_lb'] = df['Salary Estimate'].apply(lambda x: get_salary_lb(x))
df['salary_ub'] = df['Salary Estimate'].apply(lambda x: get_salary_ub(x))
col_salary = df.loc[: , ["salary_lb", "salary_ub"]]
df['salary_mean'] = col_salary.mean(axis=1)

df['company_name_clean'] = df['Company Name'].apply(lambda x: clean_name(x))
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')

plt.figure(figsize=(15,5))
g = sns.countplot(x='Industry',data=df, order=df.Industry.value_counts().iloc[:50].index)
g.set_xticklabels(g.get_xticklabels(), rotation=40, ha="right")
#g.fig.set_size_inches(20, 10)
plt.title('Top 50 Industries by Job Posts')
plt.ylabel('')
plt.xlabel('')
plt.show()
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')


g = sns.catplot(x="salary_lb", y="Industry", kind="box", data=df, order=df.Industry.value_counts().iloc[:20].index)
g.set(xlim=(20, 125))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([25,50,75,100,125], minor=True)
plt.title("Salary Lower Bound - Ordered by Most Common Industries")
plt.ylabel("")
plt.xlabel("")
plt.show()
df_sal = df.groupby("Industry").median().sort_values(by = 'salary_lb', ascending=False)
g = sns.catplot(x="salary_lb", y="Industry", kind="box", data=df, order=df_sal.iloc[:25].index)
g.set(xlim=(20, 125))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([25,50,75,100,125], minor=True)
plt.title("Salary Lower Bound - Ordered by Median Salary")
plt.ylabel("")
plt.xlabel("")
plt.show()
industry_counts = df.Industry.value_counts()
CUTOFF = 10

droprows = []
for i in range(df.shape[0]):
    industry = str(df.loc[i, "Industry"])
    if industry != 'nan':
        val = industry_counts[industry]
        if val <= CUTOFF:
            droprows.append(i)
        
df_trim = df.drop(droprows)

df_sal = df_trim.groupby("Industry").median().sort_values(by = 'salary_lb', ascending=False)
g = sns.catplot(x="salary_lb", y="Industry", kind="box", data=df, order=df_sal.iloc[:25].index)
g.set(xlim=(20, 125))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([25,50,75,100,125], minor=True)
plt.title("Salary Lower Bound - Ordered by Median Salary")
plt.ylabel("")
plt.xlabel("")
plt.show()
industry_counts = df.Industry.value_counts()
CUTOFF = 10

droprows = []
for i in range(df.shape[0]):
    industry = str(df.loc[i, "Industry"])
    if industry != 'nan':
        val = industry_counts[industry]
        if val <= CUTOFF:
            droprows.append(i)
        
df_trim = df.drop(droprows)

df_sal = df_trim.groupby("Industry").median().sort_values(by = 'salary_ub', ascending=False)
g = sns.catplot(x="salary_ub", y="Industry", kind="box", data=df, order=df_sal.iloc[:25].index)
g.set(xlim=(30, 180))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([25,50,75,100,125], minor=True)
plt.title("Salary Upper Bound - Ordered by Median Salary")
plt.ylabel("")
plt.xlabel("")
plt.show()
from geopy.geocoders import Nominatim
from tqdm.notebook import tqdm

geolocator = Nominatim(user_agent = "jb_kaggle")
unique_locs = df["Location"].unique()
locs_dict = {}

df.replace('Northbrook, IL', 'Deerfield, IL', inplace=True)

for loc_name in tqdm(unique_locs):
    if str(loc_name) != 'nan':
        if 'CA' in loc_name:
            loc_name = loc_name.replace("CA", "California")
        if 'IL' in loc_name:
            loc_name = loc_name.replace("IL", "Illinois")
        if 'PA' in loc_name:
            loc_name = loc_name.replace("PA", "Pennsylvania")
        if 'Monaco,' in loc_name:
            loc_name = 'Lawndale, California, United States'
        location = geolocator.geocode(loc_name)
        if location is not None:
            lat = location.latitude
            long = location.longitude
            locs_dict[loc_name] = [lat, long]

import folium

df.reset_index(inplace=True)

for i in range(df.shape[0]):
    long = np.nan
    lat = np.nan
    my_loc = str(df.loc[i, 'Location'])
    if 'CA' in my_loc:
        my_loc = my_loc.replace('CA', 'California')
    if 'IL' in my_loc:
        my_loc = my_loc.replace("IL", "Illinois")
    if 'PA' in my_loc:
        my_loc = my_loc.replace("PA", "Pennsylvania")
    if 'Monaco,' in my_loc:
        my_loc = 'Lawndale, California, United States'
    if my_loc != 'nan' and my_loc in locs_dict.keys():
        coords = locs_dict[my_loc]
        long = coords[1]
        lat = coords[0]
    df.loc[i, 'long'] = long
    df.loc[i, 'lat'] = lat
    
from folium import plugins

df_loc = df.dropna(subset=['long'])
heat_map = folium.Map([41.8781, -87.6298], zoom_start=3)
locationArr = df_loc[['lat', 'long']]
heat_map.add_child(plugins.HeatMap(locationArr, radius=15))
heat_map
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer

X = df_loc[['lat', 'long']]

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,20))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 

kmeans = KMeans(n_clusters = 8, random_state=0).fit(X)
kmeans.cluster_centers_

cluster_map = folium.Map([41.8781, -87.6298], zoom_start=3)
for i in range(kmeans.cluster_centers_.shape[0]):
    num = sum(kmeans.labels_ == i)
    folium.CircleMarker([kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1]],
                        radius=15,
                        popup=str(num) + ' Jobs Associated with this Cluster',
                        fill_color="#3db7e4", # divvy color
                        ).add_to(cluster_map)
cluster_map

my_colors = ['#E6B0AA', '#EC7063', '#AF7AC5', '#7D3C98', '#5499C7', '#AED6F1 ', '#A3E4D7', '#16A085', '#229954', '#58D68D', '#F7DC6F', '#F5B041', '#AF601A', '#6E2C00', '#7F8C8D', '#D3D633', '#751693', '#684050', '#8493C6', '#9948CD', '#1A3F71', '#5BD474', '#044E13', '#DA7E1D', '#EEACE6']
kmeans = KMeans(n_clusters = 24, random_state=0).fit(X)
kmeans.cluster_centers_

m = folium.Map([41.8781, -87.6298], zoom_start=3)
for i in range(kmeans.cluster_centers_.shape[0]):
    num = sum(kmeans.labels_ == i)
    folium.CircleMarker([kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1]],
                        radius=30,
                        popup=str(num) + ' Jobs Associated with Cluster ' + str(i),
                        fill_color=my_colors[i],
                        fill_opacity = 0.8,# divvy color
                        ).add_to(m)
for i in range(df_loc.shape[0]):
    folium.CircleMarker([df_loc.loc[i, 'lat'], df_loc.loc[i, 'long']],
                        radius=15,
                        popup=df_loc.loc[i, 'Location'],
                        fill_color=my_colors[kmeans.labels_[i]],
                        fill_opacity = 1,# divvy color
                        ).add_to(m)
m
metro_areas = ['Chicago, IL',
               'Los Angeles, CA',
               'New York City, NY',
               'Dallas, TX',
               'SF Bay (San Francisco), CA',
               'Denver, CO',
               'Charlotte, NC',
               'Seattle, WA',
               'Phoenix, AZ',
               'Austin, TX',
               'Salt Lake City, UT',
               'Norfolk, VA',
               'Columbus, OH',
               'Jacksonville, FL',
               'Houston, TX',
               'Philadelphia, PA',
               'San Diego, CA',
               'Indianapolis, IN',
               'Topeka, KS',
               'SF Bay (San Jose), CA',
               'San Antonio, TX',
               'Atlanta, GA',
               'Hanford, CA',
               'Gainesville, FL']

for i in range(df_loc.shape[0]):
    cluster = kmeans.labels_[i]
    df_loc.loc[i, 'metro_area'] = metro_areas[cluster]

    
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')

g = sns.catplot(x="salary_lb", y="metro_area", kind="box", data=df_loc, order=df_loc['metro_area'].value_counts().iloc[:25].index)
g.set(xlim=(25, 150))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150], minor=True)
plt.title("Salary Lower Bound - Ordered by Most Common Metro Areas")
plt.show()

df_sal = df_loc.groupby("metro_area").median().sort_values(by = 'salary_lb', ascending=False)
g = sns.catplot(x="salary_lb", y="metro_area", kind="box", data=df_loc, order=df_sal.iloc[:25].index)
g.set(xlim=(25, 150))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150], minor=True)
plt.title("Salary Lower Bound - Ordered by Median Salary")
plt.show()

               
               
               
               
               
               
               
               
               
               
               
df_loc.metro_area.value_counts()
