### ALL LIBRARIES WILL BE IMPORTED HERE IN ONE PLACE FOR EASE TO MANAGE

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
plt.style.use('ggplot')
%matplotlib inline

import seaborn as sns
import folium

from bs4 import BeautifulSoup
import requests
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
import re # import regular expression to help with web scraping

from sklearn.cluster import KMeans

print('All libraries imported successfully!')
population_table = pd.read_html('https://en.wikipedia.org/wiki/Research_Triangle')[2].set_index('Rank')
population_table = population_table.sort_values(by='2019 estimate', ascending=False)
print('Total population: {:,.0f}'.format(population_table['2019 estimate'].sum()))
print('Top 5 cities population: {:,.0f}({:.0f}%)'.format(population_table['2019 estimate'].head(5).sum(),
                                            100 * population_table['2019 estimate'].head(5).sum() / population_table['2019 estimate'].sum()))
df = pd.read_csv('../input/rtp-nc-neighborhood-data/main_df.csv')
print('{} neighborhoods retreived for the top 5 cities in the Triangle'.format(df.shape[0]))

df_nbh = pd.read_csv('../input/rtp-nc-neighborhood-data/neighborhood_df.csv')
print('{} neighborhoods whose coordinates can be identified. Those without coordinates will be dropped from this analysis.'.format(df_nbh.shape[0]))

triangle_venues = pd.read_csv('../input/rtp-nc-neighborhood-data/triangle_venues.csv')
print('{} venues retreived for {} neighborhoods with coordinates.'.format(triangle_venues.shape[0], triangle_venues.groupby('Neighborhood').count().shape[0]))

# Define a function to retrieve latitude and longitudes of the neighborhoods
def get_coordinates(neighborhood, city):
    address = str(neighborhood) + ', ' + str(city) + ', NC'

    location = Nominatim(user_agent="hkhuu@elon.edu").geocode(address)
    if location == None:
        return [np.nan, np.nan]
    else:
        return [location.latitude, location.longitude]

    
# Try out the function to obtain the coordinates of Research Triangle Park area in Durham
rtp_coordinates = get_coordinates('Research Triangle Park', 'Durham')
LATITUDE, LONGITUDE = rtp_coordinates[0], rtp_coordinates[1]
print(LATITUDE, LONGITUDE)
# create map of Research Triangle Park area using latitude and longitude values
map_triangle = folium.Map(location=[LATITUDE, LONGITUDE], zoom_start=9.5)

# add markers to map
for lat, lng, city, neighborhood in zip(df_nbh['Latitude'], df_nbh['Longitude'], df_nbh['City'], df_nbh['Neighborhood']):
    label = '{}, {}'.format(neighborhood, city)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=3,
        popup=label,
        fill=True,
        fill_color='cyan',
        fill_opacity=0.5,
        parse_html=False).add_to(map_triangle)  
    
map_triangle
# Let's create 2 plots to answer these questions
fig = plt.figure(figsize=[20, 6])
ax1 = fig.add_subplot(1, 2, 1, facecolor='w')
ax2 = fig.add_subplot(1, 2, 2, facecolor='w', sharey=ax1)

# Add 1st plot to show number of neighborhoods by city
nbh_by_city = df_nbh['Neighborhood'].groupby(df_nbh['City']).count().sort_values()
nbh_by_city.plot(kind='barh',
                 title='Number of Neighborhoods by City',
                 color='teal',
                 alpha=0.6,
                 ax=ax1)
yticklabels = nbh_by_city.index

# Add 2nd plot to show number of venues by city
df_merged = triangle_venues.merge(df_nbh[['City', 'Neighborhood']], on='Neighborhood')
venues_by_city = df_merged['Venue'].groupby(df_merged['City']).count().reindex(yticklabels)
venues_by_city.plot(kind='barh',
                    title='Number of Venues by City',
                    color='red',
                    alpha=0.6,
                    ax=ax2)

# Define function to anotate the labels of each plot
def auto_annotate(series, size, axes=plt, color='grey', dist=0):
    for index, value in enumerate(series): 
        label = format(int(value), ',') # format int with commas

        # place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
        axes.annotate(label, xy=(value + dist, index), color=color, ha='left', fontsize=size)
        
        
auto_annotate(nbh_by_city,
              axes=ax1,
              dist=3,
              size=12)

auto_annotate(venues_by_city,
              axes=ax2,
              dist=10,
              size=12)

# Improve the asthetic of the plot
ax1.set_xlim(0, 180) # Increase plot limit to make sure annotations are included
ax1.set_xticks([]) # Clean x tick marks
ax1.set_ylabel(None) # Clean y field name 'City'
ax1.set_yticklabels(labels=ax1.get_yticklabels(), fontdict={'fontsize': 12}, color='black') # Make city name pops

ax2.set_xlim(0, 1200)
ax2.set_xticks([])

plt.show()
print('There are {} unique types of venues.'.format(np.unique(triangle_venues['Venue Category']
                                                             ).shape[0]
                                                   )
     )
top10venues = triangle_venues['Venue'].groupby(triangle_venues['Venue Category']
                                              ).count().sort_values(ascending=False
                                                                   ).head(10)
print('Among them, the top 10 venue types with highest count are:', ', '.join(top10venues.index.values))

grouped_venues = triangle_venues['Venue'].groupby(triangle_venues['Neighborhood']).count()
count, bin_edges = np.histogram(grouped_venues) # Creating 10 bins of equal distance
grouped_venues.plot(kind='hist', figsize=(10, 6), xticks=bin_edges, facecolor='limegreen', grid=False, alpha=0.6)

plt.title('Histogram of Number of Neighborhoods per Venue count') # add a title to the histogram
plt.ylabel('Number of Neighborhood') # add y-label
plt.xlabel('Venues count') # add x-label

print('Number of neighborhoods having 5 venues of less: %.0i'%(grouped_venues < 6).sum())
print('Number of neighborhoods having 10 venues or more: %.0i'%(grouped_venues >= 10).sum())

plt.show()
# Use get_dummies method to turn categorical data for venues category to dummy variables
triangle_onehot = pd.get_dummies(triangle_venues[['Venue Category']], prefix="", prefix_sep="").drop('Neighborhood', axis=1)

# add neighborhood column back to dataframe
triangle_onehot.insert(0, 'Neighborhood', triangle_venues['Neighborhood'])
print('Dimensions: ', triangle_onehot.shape)
triangle_onehot.head()
triangle_grouped = triangle_onehot.groupby('Neighborhood').mean().reset_index()
print('Dimensions: ', triangle_grouped.shape)
triangle_grouped.head()
# Write a function to sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Create the new dataframe and display the top 10 venues for each neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = triangle_grouped['Neighborhood']

for ind in np.arange(triangle_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(triangle_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
# set number of clusters
kclusters = 6

triangle_grouped_clustering = triangle_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(triangle_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# Add cluster labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

triangle_merged = df_nbh

# Merge dataframes to a new one listing each neighborhood with latitudes, longitudes, and top 10 venue types
triangle_merged = triangle_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood', how='inner')
triangle_merged.head()
# create map
map_clusters = folium.Map(location=[LATITUDE, LONGITUDE], zoom_start=9.5)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.Set1(np.linspace(0, 1, len(ys)))
markers_colors = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
for lat, lon, poi, cluster in zip(triangle_merged['Latitude'], triangle_merged['Longitude'], triangle_merged['Neighborhood'], triangle_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster+1), parse_html=True)
    folium.CircleMarker([lat, lon],
                        radius=4,
                        popup=label,
                        color=markers_colors[cluster-1],
                        fill=True,
                        fill_color=markers_colors[cluster-1],
                        fill_opacity=0.5).add_to(map_clusters)
       
map_clusters
c1 = triangle_merged.loc[triangle_merged['Cluster Labels'] == 0, triangle_merged.columns[[0, 1] + list(range(5, triangle_merged.shape[1]))]]
c1.head(3)
c2 = triangle_merged.loc[triangle_merged['Cluster Labels'] == 1, triangle_merged.columns[[0, 1] + list(range(5, triangle_merged.shape[1]))]]
c2.head(3)
c3 = triangle_merged.loc[triangle_merged['Cluster Labels'] == 2, triangle_merged.columns[[0, 1] + list(range(5, triangle_merged.shape[1]))]]
c3.head(3)
c4 = triangle_merged.loc[triangle_merged['Cluster Labels'] == 3, triangle_merged.columns[[0, 1] + list(range(5, triangle_merged.shape[1]))]]
c4.head(3)
c5 = triangle_merged.loc[triangle_merged['Cluster Labels'] == 4, triangle_merged.columns[[0, 1] + list(range(5, triangle_merged.shape[1]))]]
c5.head(3)
c6 = triangle_merged.loc[triangle_merged['Cluster Labels'] == 5, triangle_merged.columns[[0, 1] + list(range(5, triangle_merged.shape[1]))]]
c6.head(3)
# Let's create 2 plots to answer these questions
fig = plt.figure(figsize=[20, 6])
ax1 = fig.add_subplot(1, 2, 1, facecolor='w')
ax2 = fig.add_subplot(1, 2, 2, facecolor='w')
fig.suptitle('COMPARING CLUSTERS', fontsize=16)

# Bar chart for count of neighborhood per cluster
triangle_merged.groupby('Cluster Labels')['Neighborhood'].count().plot(kind='bar',
                                                                             color='limegreen',
                                                                             title='Neighborhood by Cluster',
                                                                             grid=False,
                                                                             ax=ax1)
xlabels = triangle_merged.groupby('Cluster Labels')['Neighborhood'].count().index
ax1.set_xticklabels(['Cluster ' + str(i+1) for i in xlabels], rotation=30)
ax1.set_xlabel(None)
ax1.set_ylabel('Number of Neighborhoods')

for index, value in enumerate(triangle_merged.groupby('Cluster Labels')['Neighborhood'].count()):
    ax1.annotate(value, xy=[index, value], ha='center', va='bottom', fontsize=12)

# Percentage stacked bar chart
df_grouped = triangle_merged.groupby(['Cluster Labels', 'City']).agg({'Neighborhood':'count'})
df_grouped = df_grouped.groupby(level=0).apply(lambda x: 100*x / float(x.sum())).reset_index()
df_grouped = df_grouped.pivot(columns='City', index='Cluster Labels', values='Neighborhood')
df_grouped.plot(kind='bar',
                stacked=True,
                legend=False,
                grid=False,
                title='Percentage of Cities by Cluster',
                ax=ax2)
ax2.legend(loc='lower right')
ax2.set_xlabel(None)
ax2.set_ylabel('Percentage')
ax2.set_xticklabels(['Cluster ' + str(int(item.get_text())+1) for item in ax2.get_xticklabels()], rotation=30)

plt.show()
