# Main imports
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns


# Read data into pandas dataframe
df_jobs = pd.read_csv("/kaggle/input/data-scientist-jobs/DataScientist.csv")
print(df_jobs.head())
print(df_jobs.info())
# Remove Unnamed: 0 Column
df_jobs = df_jobs.drop(columns = "Unnamed: 0")


# Parse Salary Info, Company Name, Size and Revenue
HOURS_PER_WEEK = 40
WEEKS_PER_YEAR = 52

for i in range(df_jobs.shape[0]):
    salary_estimate = df_jobs.loc[i, "Salary Estimate"]
    salary_estimate = salary_estimate.replace("$", "")
    if 'Per Hour' in salary_estimate:
        lower, upper = salary_estimate.split('-')
        upper, _ = upper.split("Per")
        upper = upper.strip()
        lower = int(lower) * HOURS_PER_WEEK * WEEKS_PER_YEAR * (1/1000)
        upper = int(upper) * HOURS_PER_WEEK * WEEKS_PER_YEAR * (1/1000)
        df_jobs.loc[i, "Salary Estimate Extrapolate"] = True
    else:
        lower, upper = salary_estimate.split('-')
        lower = lower.replace("K", "")
        upper, _ = upper.split("(")
        upper = upper.replace("K", "")
        upper = upper.strip()
        df_jobs.loc[i, "Salary Estimate Extrapolate"] = False
    lower = int(lower)
    upper = int(upper)
    df_jobs.loc[i, "Salary Estimate Lower Bound"] = lower
    df_jobs.loc[i, "Salary Estimate Upper Bound"] = upper
    name = df_jobs.loc[i, "Company Name"]
    if '\n' in name:
        name, _ = name.split('\n')
    df_jobs.loc[i, "Company Name"] = name
    size = df_jobs.loc[i, "Size"]
    if 'to' in size:
        lower, upper = size.split('to')
        lower = lower.strip()
        _, upper, _ = upper.split(' ')
        upper = upper.strip()
        lower = int(lower)
        upper = int(upper)
    elif '+' in size:
        lower, _ = size.split('+')
        lower = int(lower)
        upper = np.inf
    else:
        lower = np.nan
        upper = np.nan
    df_jobs.loc[i, "Size Lower Bound"] = lower
    df_jobs.loc[i, "Size Upper Bound"] = upper
    revenue = str(df_jobs.loc[i, "Revenue"])
    if "$" in revenue:
        LOWER_MULT = 1
        UPPER_MULT = 1
        revenue = revenue.replace('$', "")
        if 'to' in revenue:
            lower, upper = revenue.split('to')
            if 'million' in lower:
                UPPER_MULT = 1000
                lower = lower.strip(" million")
            else:
                if 'billion' in upper:
                    UPPER_MULT = 1000
                    LOWER_MULT = 1000
            #lower, _ = lower.split(" ")
            _, upper, _, _ = upper.split(" ")
            upper = int(upper.strip()) * UPPER_MULT
            lower = int(lower.strip()) * LOWER_MULT 
        elif 'Less' in revenue:
            lower = 0
            _, upper = revenue.split('than')
            upper, _ = upper.split('million')
            upper = int(upper)
        else:
            lower,_ = revenue.split('+')
            lower = int(lower.strip())
            upper = np.inf
    else:
        lower = np.nan
        upper = np.nan
    df_jobs.loc[i, "Revenue Lower Bound"] = lower
    df_jobs.loc[i, "Revenue Upper Bound"] = upper
    
    
df_jobs = df_jobs.drop(columns = "Salary Estimate")
df_jobs = df_jobs.drop(columns = "Size")
df_jobs = df_jobs.drop(columns = "Revenue")

# Replace -1's
df_jobs = df_jobs.replace([-1,'-1'], np.nan)

col_salary = df_jobs.loc[: , ["Salary Estimate Lower Bound", "Salary Estimate Upper Bound"]]
df_jobs['Salary Estimate Median'] = col_salary.mean(axis=1)

col_size = df_jobs.loc[: , ["Size Lower Bound", "Size Upper Bound"]]
df_jobs['Size Median'] = col_size.mean(axis=1)

col_revenue = df_jobs.loc[: , ["Revenue Lower Bound", "Revenue Upper Bound"]]
df_jobs['Revenue Median'] = col_revenue.mean(axis=1)
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')


g = sns.catplot(x="Salary Estimate Lower Bound", y="Industry", kind="box", data=df_jobs, order=df_jobs.Industry.value_counts().iloc[:25].index)
g.set(xlim=(0, 250))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250], minor=True)
plt.title("Salary Lower Bound - Ordered by Most Common Industries")
plt.show()
df_sal = df_jobs.groupby("Industry").median().sort_values(by = 'Salary Estimate Lower Bound', ascending=False)
g = sns.catplot(x="Salary Estimate Lower Bound", y="Industry", kind="box", data=df_jobs, order=df_sal.iloc[:25].index)
g.set(xlim=(0, 250))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250], minor=True)
plt.title("Salary Lower Bound - Ordered by Median Salary")
plt.show()
industry_counts = df_jobs.Industry.value_counts()
CUTOFF = 20

droprows = []
for i in range(df_jobs.shape[0]):
    industry = str(df_jobs.loc[i, "Industry"])
    if industry != 'nan':
        val = industry_counts[industry]
        if val < CUTOFF:
            droprows.append(i)
        
df_trim = df_jobs.drop(droprows)

df_sal = df_trim.groupby("Industry").median().sort_values(by = 'Salary Estimate Lower Bound', ascending=False)
g = sns.catplot(x="Salary Estimate Lower Bound", y="Industry", kind="box", data=df_jobs, order=df_sal.iloc[:25].index)
g.set(xlim=(0, 250))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250], minor=True)
plt.title("Salary Lower Bound - Ordered by Median Salary (Excluding Low-Frequency Industries)")
plt.show()
g = sns.catplot(x="Salary Estimate Upper Bound", y="Industry", kind="box", data=df_jobs, order=df_jobs.Industry.value_counts().iloc[:25].index)
g.set(xlim=(50, 300))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250,275,300], minor=True)
plt.title("Salary Upper Bound - Ordered by Most Common Industries")
plt.show()


df_sal = df_jobs.groupby("Industry").median().sort_values(by = 'Salary Estimate Upper Bound', ascending=False)
g = sns.catplot(x="Salary Estimate Upper Bound", y="Industry", kind="box", data=df_jobs, order=df_sal.iloc[:25].index)
g.set(xlim=(50, 300))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250,275,300], minor=True)
plt.title("Salary Upper Bound - Ordered by Median Salary")
plt.show()

industry_counts = df_jobs.Industry.value_counts()
CUTOFF = 20

droprows = []
for i in range(df_jobs.shape[0]):
    industry = str(df_jobs.loc[i, "Industry"])
    if industry != 'nan':
        val = industry_counts[industry]
        if val < CUTOFF:
            droprows.append(i)
        
df_trim = df_jobs.drop(droprows)

df_sal = df_trim.groupby("Industry").median().sort_values(by = 'Salary Estimate Upper Bound', ascending=False)
g = sns.catplot(x="Salary Estimate Upper Bound", y="Industry", kind="box", data=df_jobs, order=df_sal.iloc[:25].index)
g.set(xlim=(50, 300))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250,275,300], minor=True)
plt.title("Salary Upper Bound - Ordered by Median Salary (Excluding Low-Frequency Industries)")
plt.show()
from tqdm.notebook import tqdm
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="jeff_braun")
unique_locs = df_jobs["Location"].unique()
locs_dict = {}

df_jobs.replace('Northbrook, IL', 'Deerfield, IL')

for loc_name in unique_locs:
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
            
for i in range(df_jobs.shape[0]):
    long = np.nan
    lat = np.nan
    my_loc = str(df_jobs.loc[i, 'Location'])
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
    df_jobs.loc[i, 'Location Longitude'] = long
    df_jobs.loc[i, 'Location Latitude'] = lat

import folium
from folium import plugins

df_loc = df_jobs.copy()
df_loc = df_loc.dropna(subset=['Location Latitude'])
df_loc = df_loc.reset_index()

# mark each station as a point
m = folium.Map([41.8781, -87.6298], zoom_start=3)
#for index, row in df_loc.iterrows():
    #if str(row['Location Latitude']) != 'nan':
        #folium.CircleMarker([row['Location Latitude'], row['Location Longitude']],
                            #radius=15,
                            #popup=row['Location'],
                            #fill_color="#3db7e4", # divvy color
                           #).add_to(m)
# convert to (n, 2) nd-array format for heatmap
locationArr = df_loc[['Location Latitude', 'Location Longitude']]

# plot heatmap
m.add_child(plugins.HeatMap(locationArr, radius=15))
m
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer

X = df_loc[['Location Latitude', 'Location Longitude']]

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,20))

visualizer.fit(X)        # Fit the data to the visualizer
visualizer.show() 

kmeans = KMeans(n_clusters = 8, random_state=0).fit(X)
kmeans.cluster_centers_

m = folium.Map([41.8781, -87.6298], zoom_start=3)
for i in range(kmeans.cluster_centers_.shape[0]):
    num = sum(kmeans.labels_ == i)
    folium.CircleMarker([kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1]],
                        radius=15,
                        popup=str(num) + ' Jobs Associated with this Cluster',
                        fill_color="#3db7e4", # divvy color
                        ).add_to(m)
m
my_colors = ['#E6B0AA', '#EC7063', '#AF7AC5', '#7D3C98', '#5499C7', '#AED6F1 ', '#A3E4D7', '#16A085', '#229954', '#58D68D', '#F7DC6F', '#F5B041', '#AF601A', '#6E2C00', '#7F8C8D' ]

kmeans = KMeans(n_clusters = 14, random_state=0).fit(X)
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
    folium.CircleMarker([df_loc.loc[i, 'Location Latitude'], df_loc.loc[i, 'Location Longitude']],
                        radius=15,
                        popup=df_loc.loc[i, 'Location'],
                        fill_color=my_colors[kmeans.labels_[i]],
                        fill_opacity = 1,# divvy color
                        ).add_to(m)
m
metro_areas = ['San Diego, CA', 'Chicago, IL', 'Philadelphia, PA', 'Austin, TX', 'London, UK',  'San Francisco Bay, CA', 'Jacksonville, FL', 'Phoenix, AZ', 'Columbus, OH', 'Dallas, TX', 'Houston, TX',  'Los Angeles, CA', 'New York, NY', 'San Antonio, TX']
for i in range(df_loc.shape[0]):
    cluster = kmeans.labels_[i]
    df_loc.loc[i, 'Metro Area'] = metro_areas[cluster]
    
sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')

g = sns.catplot(x="Salary Estimate Lower Bound", y="Metro Area", kind="box", data=df_loc, order=df_loc['Metro Area'].value_counts().iloc[:15].index)
g.set(xlim=(0, 250))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250], minor=True)
plt.title("Salary Lower Bound - Ordered by Most Common Metro Areas")
plt.show()

df_sal = df_loc.groupby("Metro Area").median().sort_values(by = 'Salary Estimate Lower Bound', ascending=False)
g = sns.catplot(x="Salary Estimate Lower Bound", y="Metro Area", kind="box", data=df_loc, order=df_sal.iloc[:25].index)
g.set(xlim=(0, 250))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250], minor=True)
plt.title("Salary Lower Bound - Ordered by Median Salary")
plt.show()
    
g = sns.catplot(x="Salary Estimate Upper Bound", y="Metro Area", kind="box", data=df_loc, order=df_loc['Metro Area'].value_counts().iloc[:15].index)
g.set(xlim=(50, 300))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250,275,300], minor=True)
plt.title("Salary Upper Bound - Ordered by Most Common Metro Areas")
plt.show()

df_sal = df_loc.groupby("Metro Area").median().sort_values(by = 'Salary Estimate Upper Bound', ascending=False)
g = sns.catplot(x="Salary Estimate Upper Bound", y="Metro Area", kind="box", data=df_loc, order=df_sal.iloc[:25].index)
g.set(xlim=(50, 300))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250,275,300], minor=True)
plt.title("Salary Upper Bound - Ordered by Median Salary")
plt.show()


    
pol = [0.77, 0.91, 1, 1.1, 0.62, 0.55, 1.22, 1.12, 1.22, 1, 1.16, 0.75, 0.44, 1.23]

for i in range(df_loc.shape[0]):
    ma = df_loc.loc[i, 'Metro Area']
    index = metro_areas.index(ma)
    pol_adjustment = pol[index]
    df_loc.loc[i, 'Salary Lower Bound COL Adjusted'] = df_loc.loc[i, 'Salary Estimate Lower Bound'] * pol_adjustment
    df_loc.loc[i, 'Salary Upper Bound COL Adjusted'] = df_loc.loc[i, 'Salary Estimate Upper Bound'] * pol_adjustment
    

df_sal = df_loc.groupby("Metro Area").median().sort_values(by = 'Salary Lower Bound COL Adjusted', ascending=False)
g = sns.catplot(x="Salary Lower Bound COL Adjusted", y="Metro Area", kind="box", data=df_loc, order=df_sal.iloc[:25].index)
g.set(xlim=(0, 250))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250], minor=True)
plt.title("Salary Lower Bound COL Adjusted - Ordered by Median Salary")
plt.show()

df_sal = df_loc.groupby("Metro Area").median().sort_values(by = 'Salary Upper Bound COL Adjusted', ascending=False)
g = sns.catplot(x="Salary Upper Bound COL Adjusted", y="Metro Area", kind="box", data=df_loc, order=df_sal.iloc[:25].index)
g.set(xlim=(50, 300))
g.fig.set_size_inches(30, 10)
g.ax.set_xticks([50,75,100,125,150,175,200,225,250,275,300], minor=True)
plt.title("Salary Upper Bound COL Adjusted - Ordered by Median Salary")
plt.show()
