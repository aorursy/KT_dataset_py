!conda install -c anaconda beautifulsoup4 --yes
!conda install -c conda-forge folium=0.11.0 --yes
!conda install -c conda-forge geopy --yes
!conda install -c anaconda lxml --yes
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import numpy as np
import seaborn as sns
import requests

# for web scraping the data form webpages
from bs4 import BeautifulSoup

# convert an address into latitude and longitude values
import geopy
from geopy.geocoders import Nominatim 

# Matplotlib and associated plotting modules
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# map rendering library
import folium
from folium import plugins
from folium import Map
from folium.plugins import HeatMap

# Library for K-Means Clustering
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 

# Scraping the Wikipedia page for London areas.
url = "https://en.wikipedia.org/wiki/List_of_areas_of_London"
webpg = requests.get(url).text

parpg = BeautifulSoup(webpg, "lxml")
# print(parpg.prettify())
# Extracting required table from the webpage
nghd_table = parpg.find('table', {'class':'wikitable sortable'})


# Converting the extracted-table into into Pandas Dataframe
A = []
B = []
C = []
D = []

for row in nghd_table.findAll('tr'):
    cells = row.findAll('td')
    if len(cells)==6:
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        D.append(cells[3].find(text=True))
        
        
i = j = k = l = 0
for a in A:
    tempA = a.replace('\n','')
    A[i] = tempA
    i = i+1

for b in B:
    tempB = b.replace('\n','')
    B[j] = tempB
    j = j+1
    
for c in C:
    tempC = c.replace('\n','')
    C[k] = tempC
    k = k+1
    
for d in D:
    tempD = d.replace('\n','')
    D[l] = tempD
    l = l+1
    
    
df=pd.DataFrame(A,columns=['Neighborhood'])
df['London borough']=B
df['Post town']=C
df['Postcode district']=D
df.head(20)
# Removing the Paranthesis text written in the 'Neighborhood' column using RegEx
import re

p = 0

for text in df['Neighborhood']:
    df['Neighborhood'][p] = re.sub(r" ?\([^)]+\)", "", text)
    p = p+1
# Adding columns for Latitude and Longitude of the Neighborhoods
df['Neighborhood Latitude'] = None
df['Neighborhood Longitude'] = None
# Finding the coordinates of Neighborhood using 'NOMINATIM' and populating the above created colums of Neighborhood Lat. & Long.

o = 0
for address in df['Neighborhood']:
    address = address+', UK'
    geolocator = Nominatim(user_agent="London_explorer")
    location = geolocator.geocode(address)
    if (location != None):
        df['Neighborhood Latitude'][o] = location.latitude
        df['Neighborhood Longitude'][o] = location.longitude
    o = o+1


# One entry was not provided by 'NOMINATIM', so populating it manually
df['Neighborhood Latitude'][5] = 51.5855
df['Neighborhood Longitude'][5] = 0.0988

df.head(20)
# Saving the Dataframe for future use to save time. 
# df.to_csv('df.csv')
# Defining the 'FOURSQUARE API' credentials
CLIENT_ID = 'R5DTKSBGDOVZBKBZVLLS25VE4YJMSSJEN1OO544ZTUTEWAEP' 
CLIENT_SECRET = 'XLJGXBZXBQIWQYB3EY3FTGAYHFTYABYCMCKCCBBYDW1ZKQK3' 
VERSION = '20200709' 
radius = 500
LIMIT = 200

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
# Defining the Function for getting the nearby venues of a given coordinates/Neighborhood from "FOURSQUARE API"

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    v = 1
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(v, name)
            
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
        v = v+1

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
# Obtaining venues for London-Neighborhood
london_nghd_venues = getNearbyVenues(names=df['Neighborhood'],
                                   latitudes=df['Neighborhood Latitude'],
                                   longitudes=df['Neighborhood Longitude']
                                  )
# Saving the dataframe as csv file for future use without repeating the resources.
# london_nghd_venues.to_csv('london_nghd_venues.csv')
london_nghd_venues = pd.read_csv('../input/london-neighborhood-with-nearby-venues/london_nghd_venues.csv')
london_nghd_venues.drop(['Unnamed: 0'], axis = 1, inplace = True)
print(london_nghd_venues.shape)
london_nghd_venues.head()
# Creating a column for 'Venue Type'
london_nghd_venues['Venue Type'] = np.nan
print(london_nghd_venues.shape)
london_nghd_venues.head()
print('There are {} unique venue-categories.'.format(len(london_nghd_venues['Venue Category'].unique())))
# Reading a dataframe where Venue-Categories are further divided into VENUE-TYPE
q = pd.read_csv('../input/london-venue-type/q.csv')
q = q[q['Venue Category'].isin(list(london_nghd_venues['Venue Category'].unique()))]
q.reset_index(inplace = True)
q.drop('index', 1, inplace = True)

print(q.shape)
q.head()
# Populating Venue Types for various venue categories in London Neighborhoods
i = 0
for vc1 in london_nghd_venues['Venue Category']:
    j = 0
    for vc2 in q['Venue Category']:
        if (vc1 == vc2):
            london_nghd_venues['Venue Type'][i] = q['Venue Type'][j]
            j = j+1
            break
        else:
            j = j+1
    
    i = i+1
london_nghd_venues.head()
london_nghd_venues.isnull().sum()
london_nghd_venues.dropna(inplace = True)
london_nghd_venues.reset_index(inplace=True)
london_nghd_venues.isnull().sum()
print('There are {} unique venue-types.'.format(len(london_nghd_venues['Venue Type'].unique())))
# New DataFrame for calculating and storing the weights of each Venue-Type

imp = london_nghd_venues['Venue Type'].value_counts()
imp = pd.DataFrame(imp)
imp.reset_index(inplace=True)
imp.columns = ['Venue Type', 'Count']
imp['Weight'] =''
print(imp.shape)
imp.head(15)
# Calculating the weight for each venue type
t = 0
for vtype in imp['Venue Type']:
    imp['Weight'][t] = 1/((imp['Count'][t]/(imp['Count'].sum()))*1000)
    t = t+1
imp['Weight'] = imp['Weight'].astype(float)
imp = imp.sort_values('Venue Type')
imp.reset_index(inplace=True)
imp.drop('index', 1, inplace=True)
imp.head(15)
# one hot encoding
london_nghd_onehot = pd.get_dummies(london_nghd_venues[['Venue Type']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
temp = london_nghd_venues['Neighborhood'].tolist()
# london_nghd_onehot.drop(['Neighborhood'], axis = 1, inplace=True)
london_nghd_onehot.insert(loc=0, column='Neighborhood', value=temp)

print(london_nghd_onehot.shape)
london_nghd_onehot.head()
london_nghd_grouped = london_nghd_onehot.groupby('Neighborhood').sum().reset_index()
print(london_nghd_grouped.shape)
london_nghd_grouped.head()
london_nghd_clstrng = london_nghd_grouped.copy()
print(london_nghd_clstrng.shape)
z = 0
for ventype in imp['Venue Type']:
    london_nghd_clstrng[ventype] = london_nghd_clstrng[ventype] * imp['Weight'][z]
    z = z+1
print(london_nghd_clstrng.shape)
london_nghd_clstrng.head()
# SCALING FEATURES
tt1 = london_nghd_clstrng.drop('Neighborhood', 1)

from sklearn.preprocessing import MinMaxScaler

# create scaler
scaler = MinMaxScaler()

# scaling
tt2 = scaler.fit_transform(tt1)

# putting back the column names
tt2 = pd.DataFrame(tt2, columns = tt1.columns)

temp = london_nghd_clstrng['Neighborhood'].tolist()
tt2.insert(loc=0, column='Neighborhood', value=temp)


london_nghd_clstrng = tt2
print(london_nghd_clstrng.shape)
london_nghd_clstrng.head()
london_nghd_clstrng = london_nghd_clstrng.drop('Neighborhood', 1)
print(london_nghd_clstrng.shape)
london_nghd_clstrng.head()
distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,50) 
  
for k in K: 
    #Building and fitting the model 
    kmeanModel = KMeans(n_clusters=k).fit(london_nghd_clstrng) 
    kmeanModel.fit(london_nghd_clstrng)     
      
    distortions.append(sum(np.min(cdist(london_nghd_clstrng, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / london_nghd_clstrng.shape[0]) 
    inertias.append(kmeanModel.inertia_) 
  
    mapping1[k] = sum(np.min(cdist(london_nghd_clstrng, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / london_nghd_clstrng.shape[0] 
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
# setting number of clusters from above elbow method
kclusters = 10

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(london_nghd_clstrng)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]
# add clustering labels
# london_nghd_grouped.drop(['Cluster Labels'], axis=1, inplace=True)
london_nghd_grouped.insert(0, 'Cluster Labels', kmeans.labels_)
london_nghd_grouped.shape
london_nghd_grouped['Latitude'] = None
london_nghd_grouped['Longitude'] = None
london_nghd_grouped.isnull().sum()
p = 0

for neighborhood in london_nghd_grouped['Neighborhood']:
    q = 0
    for n_ref in london_nghd_venues['Neighborhood']:
        if(neighborhood == n_ref):
            london_nghd_grouped['Latitude'][p] = london_nghd_venues['Neighborhood Latitude'][q]
            london_nghd_grouped['Longitude'][p] = london_nghd_venues['Neighborhood Longitude'][q]
            q = q + 1
        else:
            q = q + 1
            
    p = p + 1
london_nghd_grouped.isnull().sum()
print(london_nghd_grouped.shape)
london_nghd_grouped.head()
aa = london_nghd_grouped.copy()
aa.drop('Neighborhood', 1, inplace = True)
aa.rename(columns = {'Cluster Labels':'Venue Type'}, inplace = True)
aa = aa.groupby('Venue Type').sum().reset_index()
aa.set_index('Venue Type',inplace=True)
aa = aa.transpose()
aa = aa.astype(int)
aa.rename(columns = {0:'C1', 1:'C2', 2:'C3', 3:'C4', 4:'C5', 
                    5:'C6', 6:'C7', 7:'C8', 8:'C9', 9:'C10'}, inplace = True) 

aa['SUM'] = aa.sum(axis=1)
aa.loc['TOTAL'] = aa.sum()


print(aa.shape)
aa.head(15)
# using SEABORN for color coded visualization of distribution: 
cm1 = sns.light_palette("red", as_cmap=True)

# Percentage share of each Venue-Type among different clusters (ROW-WISE):
rr = aa.copy()
rr = rr.astype(float)

for i in range(14):
    for j in range(10):
        a = (rr.iat[i, j]/rr.iat[i, 10])*100
        rr.iat[i, j] = a
rr.drop('SUM', 1, inplace = True)
rr.drop('TOTAL', 0, inplace = True)


# Percentage share of different Venue-Types within each Cluster (COLUMN-WISE):
cc = aa.copy()
cc = cc.astype(float)

for j in range(10):
    for i in range(14):
        a = (cc.iat[i, j]/cc.iat[14, j])*100
        cc.iat[i, j] = a
cc.drop('SUM', 1, inplace = True)
cc.drop('TOTAL', 0, inplace = True)


printmd('**Percentage Distribution of *Venue-Types* across Clusters (Row-wise):**')
display(rr.style.background_gradient(cmap='GnBu',  low=0, high=0, axis=1, subset=None))
print('\n')
printmd('**===========================================================================**')
print('\n')
printmd('**Percentage Distribution of Venue-Types within each *Cluster* (Column-wise):**')
display(cc.style.background_gradient(cmap='GnBu',  low=0, high=0, axis=0, subset=None))
for n in range(10):
    ln_nghd_cluster = london_nghd_grouped.loc[london_nghd_grouped['Cluster Labels'] == n,
                                london_nghd_grouped.columns[[1] + list(range(2, london_nghd_grouped.shape[1]))]]
    print('No. of Neighborhoods in Cluster ' + str((n+1)) + ': ', ln_nghd_cluster.shape[0])
# Defining Latitude and Longitude of London, UK using Nominatim
geolocator = Nominatim(user_agent="london_nghd_explorer")
location = geolocator.geocode("London, UK")
latitude = location.latitude
longitude = location.longitude

# Creating map
map_clusters = folium.Map(location=[latitude, longitude], tiles='Stamen Toner', min_zoom=9, max_zoom=12, zoom_start=11)

# Settin color-scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Adding markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(london_nghd_grouped['Latitude'], 
                                  london_nghd_grouped['Longitude'], 
                                  london_nghd_grouped['Neighborhood'], 
                                  london_nghd_grouped['Cluster Labels']):
    label = folium.Popup(str(poi) + ' (Cluster Label ' + str(cluster) + ')', parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0).add_to(map_clusters)


map_clusters
borough_profile = pd.read_csv('../input/london-borough-profiles/london-borough-profiles.csv', encoding = 'ISO-8859-1')
display(borough_profile.head())
display(borough_profile.tail(6))
# Removing last five rows, as they are just cumulative of above rows
borough_profile.drop(borough_profile.index[33:38], inplace = True)


# Removing characters and replacing '.' with NaN for proper evaluation with RegEx
borough_profile.replace('.', np.nan, inplace=True)

import re

for i in range(33):
    borough_profile['Inland_Area_(Hectares)'][i] = re.sub("[^\d\.]", "", borough_profile['Inland_Area_(Hectares)'][i])
    borough_profile['Overseas_nationals_entering_the_UK_(NINo),_(2015/16)'][i] = re.sub("[^\d\.]", "", borough_profile['Overseas_nationals_entering_the_UK_(NINo),_(2015/16)'][i])
    borough_profile['Modelled_Household_median_income_estimates_2012/13'][i] = re.sub("[^\d\.]", "", borough_profile['Modelled_Household_median_income_estimates_2012/13'][i])

# Using Dictionary to convert Data-types of specific columns:
convert_dict = {'Area_name':str, 'Inner/_Outer_London':str, 'GLA_Household_Estimate_2017':float,
                'Inland_Area_(Hectares)':float, 'Population_density_(per_hectare)_2017':float,
                'Net_internal_migration_(2015)':float, 'Net_international_migration_(2015)':float,
                'Net_natural_change_(2015)':float, '%_of_resident_population_born_abroad_(2015)':float,
                'Largest_migrant_population_by_country_of_birth_(2011)':str, '%_of_largest_migrant_population_(2011)':float,
                'Second_largest_migrant_population_by_country_of_birth_(2011)':str, 
                '%_of_second_largest_migrant_population_(2011)':float, 
                'Third_largest_migrant_population_by_country_of_birth_(2011)':str, 
                '%_of_third_largest_migrant_population_(2011)':float, '%_of_population_from_BAME_groups_(2016)':float,
                '%_people_aged_3+_whose_main_language_is_not_English_(2011_Census)':float,
                'Overseas_nationals_entering_the_UK_(NINo),_(2015/16)':float,
                'Largest_migrant_population_arrived_during_2015/16':str,
                'Second_largest_migrant_population_arrived_during_2015/16':str,
                'Third_largest_migrant_population_arrived_during_2015/16':str, 'Male_employment_rate_(2015)':float,
                'Female_employment_rate_(2015)':float, 'Unemployment_rate_(2015)':float,
                'Youth_Unemployment_(claimant)_rate_18-24_(Dec-15)':float, 
                'Proportion_of_16-18_year_olds_who_are_NEET_(%)_(2014)':float,
                'Proportion_of_the_working-age_population_who_claim_out-of-work_benefits_(%)_(May-2016)':float,
                '%_working-age_with_a_disability_(2015)':float,
                'Proportion_of_working_age_people_with_no_qualifications_(%)_2015':float,
                'Proportion_of_working_age_with_degree_or_equivalent_and_above_(%)_2015':float,
                'Gross_Annual_Pay,_(2016)':float, 'Gross_Annual_Pay_-_Male_(2016)':float,
                'Gross_Annual_Pay_-_Female_(2016)':float, 'Modelled_Household_median_income_estimates_2012/13':float,
                '%_adults_that_volunteered_in_past_12_months_(2010/11_to_2012/13)':float,
                'Number_of_jobs_by_workplace_(2014)':float,'Crime_rates_per_thousand_population_2014/15':float,
                'Fires_per_thousand_population_(2014)':float, 'Ambulance_incidents_per_hundred_population_(2014)':float,
                'Median_House_Price,_2015':float, 'Average_Band_D_Council_Tax_charge_(£),_2015/16':float,
                'New_Homes_(net)_2015/16_(provisional)':float, 'Homes_Owned_outright,_(2014)_%':float,
                'Being_bought_with_mortgage_or_loan,_(2014)_%':float, 'Rented_from_Local_Authority_or_Housing_Association,_(2014)_%':float,
                'Rented_from_Private_landlord,_(2014)_%':float, '%_of_area_that_is_Greenspace,_2005':float,
                'Total_carbon_emissions_(2014)':float, 'Household_Waste_Recycling_Rate,_2014/15':float,
                '%_of_adults_who_cycle_at_least_once_per_month,_2014/15':float, 'Average_Public_Transport_Accessibility_score,_2014':float,
                'Achievement_of_5_or_more_A*-_C_grades_at_GCSE_or_equivalent_including_English_and_Maths,_2013/14':float,
                'Rates_of_Children_Looked_After_(2016)':float, '%_of_pupils_whose_first_language_is_not_English_(2015)':float,
                'Male_life_expectancy,_(2012-14)':float, 'Female_life_expectancy,_(2012-14)':float,
                'Teenage_conception_rate_(2014)':float, 'Childhood_Obesity_Prevalance_(%)_2015/16':float,
                'Mortality_rate_from_causes_considered_preventable_2012/14':int, 'Political_control_in_council':str,
                'Proportion_of_seats_won_by_Conservatives_in_2014_election':float, 'Proportion_of_seats_won_by_Labour_in_2014_election':float,
                'Proportion_of_seats_won_by_Lib_Dems_in_2014_election':float, 'Turnout_at_2014_local_elections':float
               } 
  
borough_profile = borough_profile.astype(convert_dict)
borough_profile['Latitude'] = ''
borough_profile['Longitude'] = ''

geolocator = Nominatim(user_agent="London_borough_cordinates")
city ="London"
country ="Uk"

i = 0
for area in borough_profile['Area_name']:
    loc = geolocator.geocode(area+','+ country)
    borough_profile['Latitude'][i] = loc.latitude
    borough_profile['Longitude'][i] = loc.longitude
    i = i+1

    
# Converting the populated Latitude and Longitude to float type for uninterrupted operation:
borough_profile['Latitude'] = borough_profile['Latitude'].astype(float)
borough_profile['Longitude'] = borough_profile['Longitude'].astype(float)
# Displaying all the columns with their indices
for index, title in enumerate(list(borough_profile.columns)):
    print(index, '\t', title)
# Selecting required and dropping non-required columns
req_col = pd.DataFrame(borough_profile.columns)
req_col.columns = ['Col_name']
req_col.drop(req_col.index[[0,1,2,13,15,16,17,18,19,20,21,22,25,26,27,33,35,42,61,62,65,66,79,80,81,82,84,85]],
           inplace = True)
req_col.reset_index(drop = True, inplace = True)
req_col.head()
def HMap(col):
    
    from folium.plugins import HeatMap
    
    # Removing NaN rows, for uninteruppted Folium Map:-
    heat_df = borough_profile[['Area_name', col, 'Latitude', 'Longitude']].copy()
    heat_df.dropna(axis = 0, inplace = True)
    heat_df.reset_index(inplace = True)
    heat_df.drop('index', 1, inplace = True)
    
    
    val_max = heat_df[col].max()
#     val_max = val_max.item()
        
    hm_wide = HeatMap( list(zip(heat_df['Latitude'], heat_df['Longitude'],
                            heat_df[col])),
                      min_opacity=0.5,
                      max_val=val_max,
                      radius=14,
                      blur=10,
                      max_zoom=5,
                     )

    return hm_wide
# VISUALIZATION of Clusters and Heat-Map using Folium

# Defining the Nominatim instance for finding cordinates of London, UK
geolocator = Nominatim(user_agent="London_nghd_explorer")
location = geolocator.geocode("London, UK")
latitude = location.latitude
longitude = location.longitude


# Creating the Base-map using the above location cordinates
london_map = folium.Map(location=[latitude, longitude], min_zoom=9, max_zoom=13, zoom_start=10)

# Setting color-scheme for the clusters (as per their numbers)
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Defining Feature Group for clusters
fg1 = folium.FeatureGroup(name = 'Clusters', show=False)

# Adding markers to the map
for lat, lon, poi, cluster in zip(london_nghd_grouped['Latitude'], 
                                  london_nghd_grouped['Longitude'], 
                                  london_nghd_grouped['Neighborhood'], 
                                  london_nghd_grouped['Cluster Labels']):
    label = folium.Popup(str(poi) + ' (Cluster No. ' + str(int(cluster+1)) + ')', parse_html=True)
    fg1.add_child(folium.CircleMarker(
        [lat, lon],
        radius=2,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0
    )
                 )
    
# Adding the fisrt Feature Group to the Base Map    
london_map.add_child(fg1)

######################################################################################################################################################################
    
# HEAT-MAP for London Boroughs(as per various parameters)
for col in req_col['Col_name']:
    
    # Second Feature Group for each parameters(features of London Boroughs)
    fg2 = folium.FeatureGroup(name = col, show=False)
    
    heat_df = borough_profile[['Area_name', col, 'Latitude', 'Longitude']].copy()
    heat_df.dropna(axis = 0, inplace = True)
    heat_df.reset_index(inplace = True)
    heat_df.drop('index', 1, inplace = True)
    
    fg2.add_child(HMap(col))
    
    for lat, lng, label, temp in zip(heat_df['Latitude'], heat_df['Longitude'],
                                     heat_df['Area_name'], heat_df[col]):
        fg2.add_child(folium.CircleMarker(
            [lat, lng],
            radius=0.8,
            popup=label + ' (' + str(temp) + ')',
            fill=True,
            color='blue',
            fill_color='blue',
            fill_opacity=0.01
        )
                    )
    
    # Adding the Second Feature Group to base Map
    london_map.add_child(fg2)

# Adding the Layer Control to the base map, for controlling the layers of Feature Groups added.    
london_map.add_child(folium.LayerControl())

# Displaying the Map for easy comparisons and Analysis
display(london_map)
