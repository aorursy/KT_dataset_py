import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import scipy.cluster.hierarchy as shc
import os

# https://github.com/pandas-profiling/pandas-profiling
from pandas_profiling import ProfileReport
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head(10)
df.info()
df.describe(include = 'all')
profile = ProfileReport(df, title="Pandas Profiling Report")
# profile = ProfileReport(df, minimal=True)
profile.to_widgets()
profile.to_notebook_iframe()
sns.countplot(df['room_type'])
df = df[(df['room_type'] == 'Private room') | (df['room_type'] == 'Shared room')]
print('Ahha! There are {} accommodations left here.'.format(len(df)))
df = df[df['availability_365'] >= 5]
print('There are {} accommodations available here.'.format(len(df)))
df = df[df['minimum_nights'] <= 30]
plt.figure(figsize=(17, 8))

sns.countplot(df['minimum_nights'])
print('Lucky! I still have {} rooms left in my choice.'.format(len(df)))
df_filter = df.reset_index(drop=True)

df_filter.head()
df_filter.info()
profile_filter = ProfileReport(df_filter, title="Pandas Profiling Report")
# profile = ProfileReport(df, minimal=True)
profile_filter.to_widgets()
profile_filter.to_notebook_iframe()
print('There are {} that have 0 review'.format(sum(df_filter['number_of_reviews'] == 0)))
zero_review = (df_filter['number_of_reviews'] == 0)

null_last_review = (df_filter['last_review'].isna())

null_reviews_per_month = (df_filter['reviews_per_month'].isna())

print('We are now sure that {} records have null value because there is no review.\nMaybe this is the new accommodation.'.format(len(df_filter[zero_review & null_last_review & null_reviews_per_month])))
df_filter = df_filter[df_filter['number_of_reviews'] != 0]
len(df_filter)
selected_col = 'number_of_reviews last_review reviews_per_month price'.split()

df_selected = df_filter[selected_col]

df_selected.head()
df_selected.info()
df_selected.head()
df_selected['last_review'] = df_selected['last_review'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))
df_selected['last_review_day_ago'] = datetime.now() - df_selected['last_review']

df_selected['last_review_day_ago'] = df_selected['last_review_day_ago'].apply(lambda x: x.days)
df_selected.head()
df_selected.info()
df_cluster = df_selected[['number_of_reviews','reviews_per_month','price','last_review_day_ago']]
df_cluster.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(scaler.fit_transform(df_cluster), columns=df_cluster.columns)
df_minmax.head()
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")
plt.xlabel('Accommodations')
plt.ylabel('Euclidean distances')

dend = shc.dendrogram(shc.linkage(df_minmax, method='ward'))
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")
plt.xlabel('Accommodations')
plt.ylabel('Euclidean distances')

dend = shc.dendrogram(shc.linkage(df_minmax, method='ward'))
plt.axhline(y=7, color='r', linestyle='--')
# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
group = cluster.fit_predict(df_minmax)
unique, counts = np.unique(group, return_counts=True)
dict(zip(unique, counts))
df_minmax['group'] = group

df_minmax.head()
df_radar = df_minmax.groupby('group').mean()

df_radar = df_radar.reset_index()

# df_radar['group'] = ['A','B','C']
df_radar['group'] = df_radar['group'].astype('str')

df_radar

# Scaling for visualization
col_visual = 'number_of_reviews reviews_per_month price last_review_day_ago'.split()

scaler_visual = MinMaxScaler()
df_radar[col_visual] = pd.DataFrame(scaler_visual.fit_transform(df_radar.iloc[:,1:]), columns=col_visual)
df_radar.head()
from math import pi
 
# PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider(row, title, color):
 
    # number of variable
    categories=list(df_radar)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(10,10))
    # Initialise the spider plot
    ax = plt.subplot(2,2,row+1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=15)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8,1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=10)
    plt.ylim(0,1)

    # Ind1
    values=df_radar.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color=color, y=1.1)

    
# PART 2: Apply to all individuals
# initialize the figure

my_dpi=96
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df_radar.index))
 
# Loop to plot
for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group '+df_radar['group'][row], color=my_palette(row))

import folium
from folium.plugins import MarkerCluster

data = df_filter[:3000]
details_col = 'name'.split()
x = 'latitude'
y = 'longitude'

world_map_final = folium.Map(location=[40.7128,-74.0060 ],tiles='cartodbpositron',zoom_start=11) 
# world_map= folium.Map(tiles="OpenStreetMap")

for i in range(len(data)):
    lat = data.iloc[i][x]
    long = data.iloc[i][y]
    radius = 4
    popup_text = """{}<br>"""
    popup_text = popup_text.format(df_filter[details_col].iloc[i])

    if df_minmax['group'][i] == 0:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='green').add_to(world_map_final)
    elif df_minmax['group'][i] == 1:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='lightgreen').add_to(world_map_final)
    else:
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, color='gray').add_to(world_map_final)

world_map_final
# world_map_final.save('airbnb_map.html')