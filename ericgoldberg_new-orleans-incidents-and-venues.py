!pip install lxml

import lxml

import numpy as np

import pandas as pd

!pip install shapely



#from shapely.geometry import Polygon, Point, MultiPolygon



%matplotlib inline 

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

mpl.style.use('ggplot') # optional: for ggplot-like style

!pip install folium

import folium

# import k-means and evaluation for clustering stage

from sklearn.cluster import KMeans

from sklearn import metrics 

from scipy.spatial.distance import cdist 



import json # library to handle JSON files

import requests # library to handle requests

from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
#please note you'll need to download the csv from 

#https://data.nola.gov/api/views/9san-ivhk/rows.csv?accessType=DOWNLOAD

callsdf = pd.read_csv('Calls_for_Service_2018.csv')

callsdf.head()


#drop unnecessary columns

callsdf.drop(columns=['NOPD_Item', 'Priority', 'InitialType', 'InitialTypeText', 'InitialPriority', 'MapX', 'MapY', 'TimeDispatch', 'TimeArrive', 'TimeClosed', 'Disposition', 'DispositionText', 'SelfInitiated'], inplace=True) #drop unnecessary columns



#drop unnecessary types

types = callsdf['TypeText'].unique().tolist()

types = sorted(types)

removes =[]

rs = [0, 1, 18, 19, 24, 25, 26, 27, 28, 29, 38, 39, 42, 51, 52, 55, 60, 67, 68, 88, 92, 93, 94, 96, 101, 105, 109, 119, 121, 122, 128, 153, 159, 160, 161, 162, 163, 169, 173, 174]   # non-crime incidents     



for i in rs:

    removes.append(types[i]) #make traffic, medical, and misc incidents



removes #list of traffic, medical, and misc incidents

civildf = callsdf[~callsdf.TypeText.isin(removes)] #get df without traffic, medical, and miscillaneous incidents



#drop rows with nan and 0, 0 values in Location

civildf.Location.replace(to_replace='(0.0, 0.0)', value=np.nan, inplace=True)

civildf.dropna(subset=['Location'], inplace=True)



#Reset the index

civildf.reset_index(inplace=True)

civildf.drop(columns='index', inplace=True)



#Change our location values from string type into a latitude, longitude tuple 

civildf['Location'] = civildf['Location'].str.replace('[()]', '', regex=True)

for i in range(civildf.shape[0]):

    b = civildf.loc[i, 'Location'].split(",") #split into a list

    for j in range(len(b)):

        b[j] = float(b[j])

    civildf.at[i, 'Location'] = b #reassign poly value into list of x, y tuples

    

#rename columns

civildf.columns=['Code', 'Type', 'Time', 'Beat', 'Address', 'Zip', 'District', 'Coordinates']

civildf.head()
tables = pd.read_html('https://en.wikipedia.org/wiki/Neighborhoods_in_New_Orleans', header=0)

neighborhoods = tables[0]

neighborhoods.head()
#please note you'll need to download the csv from 

#https://data.nola.gov/api/views/abhb-x4ch/rows.csv?accessType=DOWNLOAD

boundariesdf = pd.read_csv('Boundary_data.csv')

boundariesdf.head()
boundariesdf.columns=['Poly', 'id', 'Neighborhood', 'length', 'area'] #Change column names

boundariesdf.drop(columns=['id', 'length', 'area'], inplace=True)

df1 = boundariesdf

df1['Poly'] = df1['Poly'].str.replace('[MULTIPOLYGON((()))]', '', regex=True) #remove the multiploy and stuff

for hood in range(72):

    x = df1.loc[hood, 'Poly'].split(",") #split into a list

    for i in range(0, len(x)):

        x[i] = x[i].split() #split list items into tuples

        for j in range(len(x[i])):

            x[i][j] = float(x[i][j]) #convert tuple items into floats

    df1.at[hood, 'Poly'] = x #reassign poly value into list of x, y tuples

df1['Poly'] = df1['Poly'].apply(Polygon) # convert our list into polygon objects

boundariesdf.replace(to_replace='ST.  ANTHONY', value='ST. ANTHONY', inplace=True)

boundariesdf.head()
#add in a neighborhood column

civildf['Neighborhood']=civildf['Coordinates']

#add neighborhood field to each data point in 

for c in range(0, civildf.shape[0]):

    p = Point(civildf.loc[c, 'Coordinates'][1], civildf.loc[c, 'Coordinates'][0])

    for h in range(boundariesdf.shape[0]):

        if boundariesdf.loc[h, 'Poly'].contains(p):

            civildf.loc[c, 'Neighborhood']=boundariesdf.loc[h, 'Neighborhood']

            break

        else:

            continue



project.save_data(file_name = "civildf.csv",data = civildf.to_csv(index=False), overwrite=True) #saves the file on IBM Watson Studio
# The code was removed by Watson Studio for sharing.
#rows that didn't receive a proper neighborhood label

civildf[~civildf['Neighborhood'].isin(boundariesdf['Neighborhood'])].shape
#drop rows that didn't receive a proper neighborhood label

civildf=civildf[civildf['Neighborhood'].isin(boundariesdf['Neighborhood'])]

civildf.reset_index(inplace=True)

civildf.drop(columns='index', inplace=True)

civildf.head()
#load saved data frame 

civildf = pd.read_csv('civildffinal.csv')

civildf.head()
#convert the time information to a pandas timestamp

civildf['Time'] = pd.to_datetime(civildf['Time'])
# The code was removed by Watson Studio for sharing.
#Dataframe of total incidents per incident type

typetotals = civildf.Type.value_counts().to_frame()

typetotals.columns=['Count']

typetotals.head()
#barchart of top 20 Incident Types

typetotals.head(20).plot(kind='bar', figsize=(10, 6), legend=None)

plt.ylabel('Incident Count')

plt.xlabel('Incident Type')

plt.title('Top 20 Neighborhood Safety Related Incident Types 2018')

plt.show
for t in typetotals.head(20).index:

    civildf[civildf['Type']==t].Neighborhood.value_counts().to_frame().head(20).plot(kind='bar', figsize=(10,6), legend=None)

    plt.title('Top 20 Neighborhoods for {}'.format(t))

    plt.figtext(0.75,0.5, 'Descriptive Statistics \n {}'.format(civildf[civildf['Type']==t].Neighborhood.value_counts().describe().to_string()))

    plt.show
# histogram of incident type occurrence

count1, bin_edges1 = np.histogram(typetotals)

typetotals.plot(kind='hist', figsize=(10, 6), xticks = bin_edges1, legend=None)

plt.ylabel('Number of Incident Types')

plt.xlabel('Incident Count')

plt.figtext(0.75,0.5, 'Descriptive Statistics \n {}'.format(typetotals.describe().to_string()))

plt.title('Frequency of Incident Total by Type')

plt.show
count2, bin_edges2 = np.histogram(typetotals[(typetotals['Count'] >= 28) & (typetotals['Count'] <= 771)])

typetotals[(typetotals['Count'] >= 28) & (typetotals['Count'] <= 771)].plot(kind='hist', figsize=(10, 6), xticks = bin_edges2, legend=None)

plt.ylabel('Number of Incident Types')

plt.xlabel('Incident Count')

plt.title('Quartiles 2 & 3 of Frequency Dist. of Total by Type')

plt.figtext(0.75,0.5, 'Descriptive Statistics \n {}'.format(typetotals[(typetotals['Count'] >= 28) & (typetotals['Count'] <= 771)].describe().to_string()))

plt.show
#Dataframe of total incidents per Neighborhood

hoodtotals = civildf.Neighborhood.value_counts().to_frame()

hoodtotals.columns=['Incident Count']

hoodtotals.head()
# Barchart of neighborhoods with the top 20 incident occurrence

%matplotlib inline

hoodtotals.head(20).plot(kind='bar', figsize=(10, 6), legend=None)

plt.xlabel('Neighborhood')

plt.ylabel('Incident Count')

plt.title('Neighborhood-Safety/Crime-related 911 Calls for top 20 Neighborhoods, New Orleans 2018')

plt.show
#top 20 incidents by neighborhood and their descriptive stats 

for h in hoodtotals.head(20).index:

    civildf[civildf['Neighborhood']==h].Type.value_counts().to_frame().head(20).plot(kind='bar', figsize=(10,6), legend=None)

    plt.title('Top 20 Incidents for {}'.format(h))

    plt.figtext(0.75,0.5, 'Descriptive Statistics \n {}'.format(civildf[civildf['Neighborhood']==h].Type.value_counts().describe().to_string()))

    plt.show
#histogram, showing distribution of incident count among neighborhoods

count, bin_edges = np.histogram(hoodtotals)

hoodtotals.plot(kind='hist', figsize=(10, 6), xticks = bin_edges, legend=None)

plt.ylabel('Number of Neighborhoods')

plt.xlabel('Incident Count')

plt.figtext(0.70,0.5, 'Descriptive Statistics \n {}'.format(hoodtotals.describe().to_string()))

plt.title('Frequency of Incident Count by Neighborhoods')

plt.show
# The code was removed by Watson Studio for sharing.
#corrected geojson available in github project folder

nola_geo = r'nola_geojson.json'
#join our neighborhoods dataframe to the hoodtotals

hoodts = neighborhoods.join(hoodtotals, on='Neighborhood')

hoodts.head()
# Map of New Orleans Neighborhoods

# New Orleans latitude and longitude values

latitude = 29.951065

longitude = -90.071533



# create map and display it

nola_map = folium.Map(location=[latitude, longitude], zoom_start=11)



# loop through the neighborhoods and add each to the map

for lat, lng, label in zip(neighborhoods.Latitude, neighborhoods.Longitude, neighborhoods.Neighborhood):

    folium.CircleMarker(

        [lat, lng],

        radius=5, # define how big you want the circle markers to be

        color='yellow',

        fill=True,

        popup=label,

        fill_color='blue',

        fill_opacity=0.6

    ).add_to(nola_map)





# display the map of New Orleans

nola_map
nola_map = folium.Map(location=[latitude, longitude], zoom_start=11)





# generate choropleth map using the neighborhood-safety related incidents in nola by neighborhood

nola_map.choropleth(

    geo_data=nola_geo,

    data=hoodts,

    columns=['Neighborhood', 'Incident Count'],

    key_on='feature.properties.gnocdc_lab',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Neighborhood Safety Related Incidents 2018',

    reset=True

)



# loop through the neighborhoods and add each to the map

for lat, lng, hood, count in zip(hoodts.Latitude, hoodts.Longitude, hoodts.Neighborhood, hoodts['Incident Count']):

    folium.CircleMarker(

        [lat, lng],

        radius=5, # define how big you want the circle markers to be

        color='yellow',

        fill=True,

        popup='{}, {} incidents'.format(hood, count),

        fill_color='blue',

        fill_opacity=0.6

    ).add_to(nola_map)



# display map

nola_map
#make a list of neighborhoods in each quartile

tophoods = []

upperhoods =[]

lowerhoods = []

bottomhoods = []

for i in civildf.Neighborhood.value_counts()[civildf.Neighborhood.value_counts() > 2522].index:

    tophoods.append(i)

    

for i in civildf.Neighborhood.value_counts()[(civildf.Neighborhood.value_counts() > 1316.5) & (civildf.Neighborhood.value_counts() <= 2522)].index:

    upperhoods.append(i)

    

for i in civildf.Neighborhood.value_counts()[(civildf.Neighborhood.value_counts() > 740) & (civildf.Neighborhood.value_counts() <= 1316.5)].index:

    lowerhoods.append(i)

    

for i in civildf.Neighborhood.value_counts()[civildf.Neighborhood.value_counts() <= 740].index:

    bottomhoods.append(i)
nola_map = folium.Map(location=[latitude, longitude], zoom_start=11)





# generate choropleth map using the neighborhood-safety related incidents in nola by neighborhood

nola_map.choropleth(

    geo_data=nola_geo,

    data=hoodts[~hoodts['Neighborhood'].isin(tophoods)],

    columns=['Neighborhood', 'Incident Count'],

    key_on='feature.properties.gnocdc_lab',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Neighborhood Safety Related Incidents 2018, top quartile excluded',

    reset=True

)



# loop through the neighborhoods and add each to the map

for lat, lng, hood, count in zip(hoodts.Latitude, hoodts.Longitude, hoodts.Neighborhood, hoodts['Incident Count']):

    folium.CircleMarker(

        [lat, lng],

        radius=5, # define how big you want the circle markers to be

        color='yellow',

        fill=True,

        popup='{}, {} incidents'.format(hood, count),

        fill_color='blue',

        fill_opacity=0.6

    ).add_to(nola_map)



# display map

nola_map
# monthly time series by neighborhood

df3 = civildf.groupby([civildf.Time.map(lambda t: t.month), 'Neighborhood']).count() #group by month and sum incidents by neighborhood

df3.drop(columns=['Code', 'Type', 'Time', 'Beat', 'Address', 'Zip', 'District'], inplace=True) #drop unnecessary columns

df3.reset_index(inplace=True)

df3.columns=['Month', 'Neighborhood', 'Count'] #rename columns

df3 = df3.pivot(index='Neighborhood', columns='Month', values='Count') #pivot

df3.fillna(value=0, inplace=True) #fill nan values

df3['Total'] = df3.sum(axis=1)

df3.sort_values(['Total'], ascending=False, axis=0, inplace=True)

mhood = df3.drop(columns='Total').transpose()

mhood['Total']=mhood.sum(axis=1)

mhood.head()



#df3.groupby('Month')['Count'].sum().plot(kind=) #get  total incidents by month

#time series grouped by incident type

df4 = civildf.groupby([civildf.Time.map(lambda t: t.month), 'Type']).count() #group by month and sum incidents by neighborhood

df4.drop(columns=['Code', 'Neighborhood', 'Time', 'Beat', 'Address', 'Zip', 'District'], inplace=True) #drop unnecessary columns

df4.reset_index(inplace=True)

df4.columns=['Month', 'Type', 'Count'] #rename columns

df4 = df4.pivot(index='Type', columns='Month', values='Count') #pivot to get months as columns

df4.fillna(value=0, inplace=True) #fill nan values

df4['Total'] = df4.sum(axis=1)

df4.sort_values(['Total'], ascending=False, axis=0, inplace=True)

mincident = df4.drop(columns='Total').transpose()

mincident['Total']=mincident.sum(axis=1)

mincident.head()
mincident.Total.plot(kind='line', figsize=(10, 6), xticks=np.linspace(1, 12, num=12))

plt.ylabel('Incident Count')

plt.title('Total Neighborhood-Safety Incidents by Month, New Orleans 2018')

plt.show
# what was higher than average, what was lower by 1.25 std deviations 

mincident[(mincident['Total']>(mincident.describe().loc['mean', 'Total']+1.25*(mincident.describe().loc['std', 'Total']))) | (mincident['Total']<(mincident.describe().loc['mean', 'Total']-1.25*(mincident.describe().loc['std', 'Total'])))]
mincident.iloc[:, 1:6].plot(kind='line', figsize=(10,6),xticks=np.linspace(1, 12, num=12))

plt.ylabel('Incident Count')

plt.title('Selected Incident Types Monthly Totals, NOLA 2018')

plt.show
#time series of median neighborhoods per quartile

mhood[[tophoods[9], upperhoods[9], lowerhoods[8], bottomhoods[10]]].plot(kind='area', figsize=(10, 8), xticks=np.linspace(1, 12, num=12), stacked=False, alpha=0.25)

plt.ylabel('Incident Count')

plt.title('Timeseries of Incidents for Quartile Median Neighborhoods')

plt.show
mincident.describe()
mhood.describe()
# The code was removed by Watson Studio for sharing.
#function for getting nearby venues

LIMIT = 100

def getNearbyVenues(names, latitudes, longitudes, radius=750):

    

    venues_list=[]

    for name, lat, lng in zip(names, latitudes, longitudes):

        print(name)

            

        # create the API request URL

        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(

            CLIENT_ID, 

            CLIENT_SECRET, 

            VERSION, 

            lat, 

            lng, 

            radius, 

            LIMIT)

            

        # make the GET request

        results = requests.get(url).json()["response"]['groups'][0]['items']

        

        # return only relevant information for each nearby venue

        venues_list.append([(

            name, 

            lat, 

            lng, 

            v['venue']['name'], 

            v['venue']['location']['lat'], 

            v['venue']['location']['lng'],  

            v['venue']['categories'][0]['name']) for v in results])



    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])

    nearby_venues.columns = ['Neighborhood', 

                  'Neighborhood Latitude', 

                  'Neighborhood Longitude', 

                  'Venue', 

                  'Venue Latitude', 

                  'Venue Longitude', 

                  'Venue Category']

    

    return(nearby_venues)
nola_venues = getNearbyVenues(names=neighborhoods['Neighborhood'],

                                   latitudes=neighborhoods['Latitude'],

                                   longitudes=neighborhoods['Longitude']

                                  )
#neighborhoods with no venues retrieved by Foursquare

neighborhoods[~neighborhoods['Neighborhood'].isin(nola_venues.groupby('Neighborhood').count().index)]
nola_venues.shape
nola_venues['Neighborhood'].value_counts().to_frame().head()
nola_venues['Neighborhood'].value_counts().to_frame().tail()
venue_types = nola_venues.groupby('Venue Category').count()

venue_types.drop(columns=['Neighborhood', 'Neighborhood Latitude', 'Neighborhood Longitude', 'Venue', 'Venue Latitude'], inplace=True)

venue_types.columns=['Count']

venue_types = venue_types.sort_values(by='Count', ascending=False)

venue_types.head()
venue_types.head(20).plot(kind='bar', figsize = (10,6), legend=None)

plt.ylabel('Count')

plt.title('Top Venue Types New Orleans from Foursquare')

plt.show
venue_types.plot(kind='hist', figsize = (10,6), legend=None)

plt.figtext(0.75,0.5, 'Descriptive Statistics \n {}'.format(venue_types.describe().to_string()))

plt.title('Frequency of Venue Types by Count')

plt.show
#nola_venues[nola_venues['Neihborhood'] 

ven_counts = nola_venues['Neighborhood'].value_counts().to_frame()

ven_counts.columns=['Count']

ven_counts[ven_counts['Count'] < 4].index

nola_venues = nola_venues[~nola_venues['Neighborhood'].isin(ven_counts[ven_counts['Count'] < 5].index)]

nola_venues.reset_index(inplace=True)

nola_venues.drop(columns='index', inplace=True)

nola_venues.tail()
# one hot encoding

venues_onehot = pd.get_dummies(nola_venues[['Venue Category']], prefix="", prefix_sep="")



# add neighborhood column back to dataframe, notice that neighborhood was a venue category so we need a different name

venues_onehot['Neighborhoods'] = nola_venues['Neighborhood'] 



# move neighborhood column to the first column

fixed_columns = [venues_onehot.columns[-1]] + list(venues_onehot.columns[:-1])

venues_onehot = venues_onehot[fixed_columns]



#group by neighborhood and get venue type frequency

venues_grouped = venues_onehot.groupby('Neighborhoods').mean().reset_index()

venues_grouped.head()

def return_most_common_venues(row, num_top_venues):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_venues]



num_top_venues = 10



indicators = ['st', 'nd', 'rd']



# create columns according to number of top venues

columns = ['Neighborhoods']

for ind in np.arange(num_top_venues):

    try:

        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common Venue'.format(ind+1))



# create a new dataframe

neighborhoods_venues_sorted = pd.DataFrame(columns=columns)

neighborhoods_venues_sorted['Neighborhoods'] = venues_grouped['Neighborhoods']



for ind in np.arange(venues_grouped.shape[0]):

    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(venues_grouped.iloc[ind, :], num_top_venues)



neighborhoods_venues_sorted.head()
X = venues_grouped.drop('Neighborhoods', 1)

distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1, 10) 

  

for k in K: 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X) 

    kmeanModel.fit(X)     

      

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                      'euclidean'),axis=1)) / X.shape[0]) 

    inertias.append(kmeanModel.inertia_) 

  

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                 'euclidean'),axis=1)) / X.shape[0] 

    mapping2[k] = kmeanModel.inertia_ 
plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show() 
plt.plot(K, inertias, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Inertia') 

plt.title('The Elbow Method using Inertia') 

plt.show() 
# set number of clusters

kclusters = 4



venues_grouped_clustering = venues_grouped.drop('Neighborhoods', 1)



# run k-means clustering

kmeans_v = KMeans(n_clusters=kclusters, random_state=0).fit(venues_grouped_clustering)



# check cluster labels generated for each row in the dataframe

kmeans_v.labels_[0:10] 
# add clustering labels

neighborhoods_venues_sorted.insert(0, 'Venue Cluster Labels', kmeans_v.labels_)

neighborhoods_venues_sorted.head()

#merge the data frames

nolav_merged = hoodts.set_index('Neighborhood')

nolav_merged = neighborhoods_venues_sorted.join(nolav_merged, on='Neighborhoods')

nolav_merged.tail()
import matplotlib.cm as cm

import matplotlib.colors as colors

# create map

mapv_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(nolav_merged['Latitude'], nolav_merged['Longitude'], nolav_merged['Neighborhoods'], nolav_merged['Venue Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(mapv_clusters)

       

mapv_clusters
nolav_merged['Venue Cluster Labels'].value_counts().to_frame()
civil_onehot = pd.get_dummies(civildf[['Type']], prefix="", prefix_sep="")

civil_onehot['Neighborhoods'] = civildf['Neighborhood'] 

fixed_columns = [civil_onehot.columns[-1]] + list(civil_onehot.columns[:-1])

civil_onehot = civil_onehot[fixed_columns]

civil_onehot.head()
civil_onehot.shape
#group by neighborhood and get incident type frequency

hoods_grouped = civil_onehot.groupby('Neighborhoods').mean().reset_index()

hoods_grouped.head()
def return_most_common_incident(row, num_top_incident):

    row_categories = row.iloc[1:]

    row_categories_sorted = row_categories.sort_values(ascending=False)

    

    return row_categories_sorted.index.values[0:num_top_incident]



num_top_incident = 10



indicators = ['st', 'nd', 'rd']



# create columns according to number of top venues

columns = ['Neighborhoods']

for ind in np.arange(num_top_incident):

    try:

        columns.append('{}{} Most Common Incident'.format(ind+1, indicators[ind]))

    except:

        columns.append('{}th Most Common Incident'.format(ind+1))



# create a new dataframe

neighborhoods_incidents_sorted = pd.DataFrame(columns=columns)

neighborhoods_incidents_sorted['Neighborhoods'] = hoods_grouped['Neighborhoods']



for ind in np.arange(hoods_grouped.shape[0]):

    neighborhoods_incidents_sorted.iloc[ind, 1:] = return_most_common_incident(hoods_grouped.iloc[ind, :], num_top_incident)



neighborhoods_incidents_sorted.head()
X = hoods_grouped.drop('Neighborhoods', 1)

distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1, 10) 

  

for k in K: 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X) 

    kmeanModel.fit(X)     

      

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                      'euclidean'),axis=1)) / X.shape[0]) 

    inertias.append(kmeanModel.inertia_) 

  

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                 'euclidean'),axis=1)) / X.shape[0] 

    mapping2[k] = kmeanModel.inertia_ 
plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show()
plt.plot(K, inertias, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Inertia') 

plt.title('The Elbow Method using Inertia') 

plt.show() 
# set number of clusters

kclusters = 5



nolacrime_clustering = hoods_grouped.drop('Neighborhoods', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(nolacrime_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10]
#merge the frames

merger = hoodts.set_index('Neighborhood')

neighborhoods_incidents_sorted.insert(0, 'Incident Cluster Labels', kmeans.labels_)

merger = neighborhoods_incidents_sorted.join(merger, on='Neighborhoods')

merger.head()
# create map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(merger['Latitude'], merger['Longitude'], merger['Neighborhoods'], merger['Incident Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

       

map_clusters
merger['Incident Cluster Labels'].value_counts().to_frame()
#data prep for k-means combined

#remove neighborhoods without or with low venue data from hoods_grouped and join with venues_grouped

incidents_venues_grouped = hoods_grouped[hoods_grouped['Neighborhoods'].isin(venues_grouped['Neighborhoods'])].join(venues_grouped.set_index('Neighborhoods'), on='Neighborhoods')

incidents_venues_grouped.head()
X = incidents_venues_grouped.drop('Neighborhoods', 1)

distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1, 20) 

  

for k in K: 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X) 

    kmeanModel.fit(X)     

      

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                      'euclidean'),axis=1)) / X.shape[0]) 

    inertias.append(kmeanModel.inertia_) 

  

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                 'euclidean'),axis=1)) / X.shape[0] 

    mapping2[k] = kmeanModel.inertia_ 
plt.plot(K, distortions, 'bx-')

plt.xticks(np.linspace(1, 20, 20))

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show()
plt.plot(K, inertias, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('inertia') 

plt.xticks(np.linspace(1, 20, 20))

plt.title('The Elbow Method using Inertia') 

plt.show()
#values of 8 and 11 for k look good. 

# set number of clusters

kclusters = 8



together_clustering = incidents_venues_grouped.drop('Neighborhoods', 1)



# run k-means clustering

kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(together_clustering)



# check cluster labels generated for each row in the dataframe

kmeans.labels_[0:10]
#merge dataframes to get most common incidents, venues, and previous cluster labels 

together_merged = merger[merger['Neighborhoods'].isin(nolav_merged['Neighborhoods'])].join(nolav_merged.drop(columns=['Longitude', 'Latitude', 'Incident Count']).set_index('Neighborhoods'), on='Neighborhoods')

together_merged.head()
#add cluster labels

incidents_venues_grouped.insert(0, 'Combined Cluster Labels', kmeans.labels_)

incidents_venues_grouped.head()
#merge frames

together_merged = together_merged.join(incidents_venues_grouped.iloc[:, 0:2].set_index('Neighborhoods'), on='Neighborhoods')
together_merged.head()
# create map

map_combined_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(together_merged['Latitude'], together_merged['Longitude'], together_merged['Neighborhoods'], together_merged['Combined Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_combined_clusters)

       

map_combined_clusters
together_merged['Combined Cluster Labels'].value_counts().to_frame()
#combine with choropleth

# create map

map_combined_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)



# set color scheme for the clusters

x = np.arange(kclusters)

ys = [i + x + (i*x)**2 for i in range(kclusters)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# generate choropleth map using the neighborhood-safety related incidents in nola by neighborhood

map_combined_clusters.choropleth(

    geo_data=nola_geo,

    data=hoodts,

    columns=['Neighborhood', 'Incident Count'],

    key_on='feature.properties.gnocdc_lab',

    fill_color='YlOrRd', 

    fill_opacity=0.7, 

    line_opacity=0.2,

    legend_name='Neighborhood Safety Related Incidents 2018',

    reset=True

)



# add markers to the map

markers_colors = []

for lat, lon, poi, cluster in zip(together_merged['Latitude'], together_merged['Longitude'], together_merged['Neighborhoods'], together_merged['Combined Cluster Labels']):

    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)

    folium.CircleMarker(

        [lat, lon],

        radius=5,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_combined_clusters)

       

map_combined_clusters
#nolav_merged - venues

#merger - incidents

#together_merged - together
together_merged[together_merged['Combined Cluster Labels']==0].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==1].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==2].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==3].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==4].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==5].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==6].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])
together_merged[together_merged['Combined Cluster Labels']==7].drop(columns=['Incident Cluster Labels', 'Venue Cluster Labels', 'Combined Cluster Labels'])