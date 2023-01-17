import pandas as pd # primary data structure library
Traffic_Dataset = pd.read_csv('../input/traffic-violations-2019-moco-md/2_Traffic_Violation_2019.csv')
Traffic_Dataset.head()
print('Dataset Dimensions = Shape{}, Size({})'.format(Traffic_Dataset.shape, Traffic_Dataset.size))
#!pip install folium --upgrade pip
#!pip install matplotlib --upgrade pip
#!pip install geocoder --upgrade pip
#!pip install geopy --upgrade pip
#!pip install sklearn --upgrade pip
#!pip install seaborn --upgrade pip
import pandas as pd # primary data structure library
import numpy as np # useful for many scientific computing in Python
pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 100)

# Matplotlib and associated plotting modules
#inline backend used to generate the plots within the browser
import matplotlib.pyplot as plt
%matplotlib inline 

import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
mpl.style.use('ggplot')

import geocoder
from geopy.geocoders import Nominatim

#import k-means for clustering 
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

import folium # used to create interactive maps
from folium import plugins
from folium.plugins import HeatMap

import requests # library to handle requests

import seaborn as sns

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
# Create separate dataframe for visualization 
tvdf = Traffic_Dataset
#tvdf.columns = list(map(str, tvdf.columns))
tvdf.head(1)
tvdf = tvdf[tvdf.Latitude != 0] # Remove rows with Lat & Lng values of zero
tvdf =tvdf.reset_index(drop=True) # Reset index
#print(tvdf.shape)
#Display columns in dataframe, examine and identify unnecessary columns
print (tvdf.columns)
# Rename Columns
tvdf = tvdf.rename(columns={"Longitude": "X", "Latitude": "Y", "Time Of Stop": "Time_Of_Stop", "Personal Injury": "Personal_Injury", "Property Damage":"Property_Damage",
                            "Commercial License":"Commercial_License", "Work Zone":"Work_Zone","Search Conducted":"Search_Conducted",
                            "Search Disposition":"Search_Disposition","Search Outcome":"Search_Outcome","Search Reason":"Search_Reason",
                            "Search Reason For Stop":"Search_Reason_For_Stop","Search Type":"Search_Type","Search Arrest Reason":"Search_Arrest_Reason",
                            "Violation Type":"Violation_Type", "Contributed To Accident":"Contributed_To_Accident", "Driver City":"Driver_City",
                            "Driver State":"Driver_State", "DL State":"DL_State", "Arrest Type":"Arrest_Type"}) 

# Rename values in Gender, VehicleType & Make columns for clarity
tvdf.replace(('M', 'F', 'U'), ('Male', 'Female', 'Unknown'), inplace=True)

#VehicleType Column
tvdf["VehicleType"].replace({"01 - Motorcycle":"Motorcycle", "02 - Automobile":"Automobile","03 - Station Wagon":"Station Wagon", "04 - Limousine":"Limousine",
                            "05 - Light Duty Truck":"Light Duty Truck","06 - Heavy Duty Truck":"Heavy Duty Truck","07 - Truck/Road Tractor":"Truck/Road Tractor","08 - Recreational Vehicle":"Recreational Vehicle",
                            "09 - Farm Vehicle":"Farm Vehicle","10 - Transit Bus":"Transit Bus","11 - Cross Country Bus":"Cross Country Bus","12 - School Bus":"School Bus",
                            "13 - Ambulance":"Ambulance(Emerg)","13 - Ambulance(Emerg)":"Ambulance(Emerg)","14 - Ambulance":"Ambulance(Non-Emerg)","14 - Ambulance(Non-Emerg)":"Ambulance(Non-Emerg)",
                            "15 - Fire Vehicle":"Fire(Emerg)","15 - Fire(Emerg)":"Fire(Emerg)","16 - Fire(Non-Emerg)":"Fire(Non-Emerg)","17 - Police(Emerg)":"Police(Emerg)",
                            "18 - Police Vehicle":"Police(Non-Emerg)", "18 - Police(Non-Emerg)":"Police(Non-Emerg)", "19 - Moped":"Moped", "20 - Commercial Rig":"Commercial Rig","21 - Tandem Trailer":"Tandem Trailer",
                            "22 - Mobile Home":"Mobile Home","23 - Travel/Home Trailer":"Travel/Home Trailer", "24 - Camper":"Camper", "25 - Utility Trailer":"Utility Trailer", "26 - Boat Trailer":"Boat Trailer", 
                            "27 - Farm Equipment":"Farm Equipment", "28 - Other":"Other", "29 - Unknown":"Unknown",}, inplace=True)

#Make Column
tvdf["Make"].replace({"Toyt": "TOYOTA","1=TOYT":"TOYOTA","4STOYOTA":"TOYOTA", "NISS": "NISSAN","'NISSAN":"NISSAN", "DODG": "DODGE", "HOND": "HONDA", 
                      "CADI": "CADILLAC","MERZ":"MERCEDES", "ACUR":"ACURA", "VOLK":"VOLKSWAGEN", "VOLKS":"VOLKSWAGEN","VOLV":"VOLVO",
                      "HYUNDI":"HYUNDAI","HYUN":"HYUNDAI","SUBA":"SUBARU","SUBU":"SUBARU", "MITS":"MITSUBISHI","2005": "BMW", "325": "BMW",
                      "CHEV":"CHEVROLET","CHEVY":"CHEVROLET","4SCHEVY":"CHEVROLET","INFI":"INFINITI","LNDR":"LAND ROVER",
                      "LEXU":"LEXUS",",EX":"LEXUS",",LEXUS":"LEXUS",";EXU":"LEXUS",";EXUS":"LEXUS","LICN":"LINCOLN", "'LINC":"LINCOLN",
                      "DODEG":"DODGE", "'DODGE":"DODGE","CHRY":"CHRYSLER","CAD":"CADILLAC","JAG":"JAGUAR","ATURN":"SATURN",
                      "BUIC":"BUICK","BUICJ":"BUICK",";NISSAN":"NISSAN","0LDS":"OLDSMOBILE",}, inplace=True)
# Replace missing values with NaN
tvdf.replace('', np.NaN, inplace=True)
tvdf.replace('----', np.NaN, inplace=True) 
tvdf.replace('---', np.NaN, inplace=True)
tvdf.replace('....', np.NaN, inplace=True)

#Check dataframe for null values
print (tvdf.shape)
print (tvdf.isnull().sum())
pd.set_option('mode.chained_assignment', None)
tvdf2 = Traffic_Dataset[['Accident','Belts', 'Personal Injury', 'Property Damage',
          'Fatal', 'Commercial License', 'HAZMAT','Commercial Vehicle','Violation Type',
          'Alcohol','Work Zone', 'Gender','SubAgency']]

# Replace missing values with NaN
tvdf2.replace('', np.NaN, inplace=True)
tvdf2.replace('----', np.NaN, inplace=True) 
tvdf2.replace('---', np.NaN, inplace=True)
tvdf2.replace('....', np.NaN, inplace=True)

# Convert boolean to integer
tvdf2.replace(('True', 'False'), (1, 0), inplace=True),
tvdf2.replace(('Yes', 'No'), (1, 0), inplace=True)

tvdf2.replace(('M', 'F', 'U'), ('Male', 'Female', 'Unknown'), inplace=True)

# Add Total column to dataframe
#tvdf2.loc['Total'] = tvdf2.sum()
tvdf2['Total'] = tvdf2.sum(axis=1)

tvdf2.head()
# Retrieve coordinates for Montgomery County, Maryland
address = 'Montgomery County, MD'

geolocator = Nominatim(user_agent="MoCo_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geographical coordinates of Montgomery County, MD are {}, {}.'.format(latitude, longitude))
# Montgomery County, Maryland latitude and longitude values
#latitude = 39.1406267
#longitude = -77.2075612

# Create map of Montgomery County, Maryland
Incidents = folium.Map(location = [latitude, longitude], zoom_start = 10)#, tiles='CARTODBPOSITRON')

title_html = '''
             <h3 align="center" style="font-size:15px"><b>Traffic Violations (FY 2019)</b></h3>
             '''
Incidents.get_root().html.add_child(folium.Element(title_html))

# instantiate a mark cluster object for the incidents in the dataframe
violations = plugins.MarkerCluster().add_to(Incidents)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(tvdf.Y, tvdf.X, tvdf.Description):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(violations)

# display map
Incidents
accidents_df = tvdf[['Accident','X', 'Y', 'Description', 'Fatal', 'Gender', 'SubAgency', 'Location', 'Geolocation']]
accidents_df = accidents_df[accidents_df['Accident'] != 'No']
accidents_df =accidents_df.reset_index(drop=True)

Accidents_heat = folium.Map(location = [latitude, longitude], zoom_start = 10.5, tiles='CARTODBPOSITRON')

title_html = '''
             <h3 align="center" style="font-size:15px"><b>Accidents (FY 2019)</b></h3>
             '''
Accidents_heat.get_root().html.add_child(folium.Element(title_html))


# Mark each incident as a point
for index, row in accidents_df.iterrows():
    folium.Circle([row['Y'], row['X']],
                       radius=15,
                       popup=row['Description'],
                        #fill_color="#3db7e4",
                        fill_color="coral"  
                       ).add_to(Accidents_heat)

# Convert to (n, 2) nd-array format for heatmap
#stationArr = accidents_df[['Y', 'X']].as_matrix()
violationsArr = accidents_df[['Y', 'X']].values

# plot heatmap
Accidents_heat.add_child(plugins.HeatMap(violationsArr, radius=15))
folium.LayerControl().add_to(Accidents_heat)
# Display map
Accidents_heat
# Create new dataframe with the above-mentioned categories
district_df = tvdf2[['Accident','SubAgency','Fatal','Alcohol', 'Gender', 'Personal Injury', 'Property Damage', 'Commercial License']]#,'Commercial Vehicle' ]] 
district_df[district_df['Accident'] != 0]
district_df.set_index('SubAgency', inplace=True)
district_df = district_df.groupby('SubAgency', axis=0).sum()

#Plot Barchart
ax = district_df.plot(kind='bar',width=.9, align='edge', figsize=(18,10),color=('purple', 'red', 'gold', 'brown', 'chocolate', 'steelblue'),
                      fontsize=13);
ax.set_alpha(0.8)
ax.set_title("Incidents by Police District (FY 2019)",
fontsize=15, color= 'navy', fontstyle='normal')

ax = plt.axes()
ax.set_facecolor("white")
ax.grid(which='major', linestyle='-', linewidth='0.3', color='pink')
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth='0.3', color='gold')

plt.xticks(rotation=20)

ax.set_ylabel("Incidents", fontsize=15, color='navy');
ax.set_xlabel('District', fontsize=15, color='navy')

# set individual bar lables using above list
for i in ax.patches:
    #get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()- .002, i.get_height()+9, \
            str(round((i.get_height()), 2)), fontsize=10, color='navy',fontstyle='normal',
                rotation=0)
# Define Foursquare Credentials and Version
CLIENT_ID = 'XXXXXXXXXXXXXXXXXXX' 
CLIENT_SECRET = 'XXXXXXXXXXXXXXXX' 
VERSION = '20180605' # Foursquare API version

#print('My credentails:')
#print('CLIENT_ID: ' + CLIENT_ID)
#print('CLIENT_SECRET:' + CLIENT_SECRET)
print ('Done')
#Retrieve latitude and longitude values for each district's CBD
bethesda_address = 'Bethesda Metro'
geolocator_bethesda = Nominatim(user_agent="Bethesda_explorer")
Bethesda = geolocator.geocode(bethesda_address)
latitude_bethesda = Bethesda.latitude
longitude_bethesda = Bethesda.longitude

germantown_address = 'Germantown, MD'
geolocator_germantown = Nominatim(user_agent="Germantown_explorer")
Germantown = geolocator_germantown.geocode(germantown_address)
germantown_latitude = Germantown.latitude
germantown_longitude = Germantown.longitude

gaithersburg_address = 'Gaithersburg / Montgomery Village, MD'
geolocator_gaithersburg = Nominatim(user_agent="Gaithersburg_explorer")
Gaithersburg = geolocator_gaithersburg.geocode(gaithersburg_address)
gaithersburg_latitude = Gaithersburg.latitude
gaithersburg_longitude = Gaithersburg.longitude

rockville_address = 'Rockville, MD'
geolocator_rockville = Nominatim(user_agent="Rockville_explorer")
Rockville = geolocator_rockville.geocode(rockville_address)
rockville_latitude = Rockville.latitude
rockville_longitude = Rockville.longitude

wheaton_address = 'Wheaton, MD'
geolocator_wheaton = Nominatim(user_agent="Wheaton_explorer")
Wheaton = geolocator_wheaton.geocode(wheaton_address)
wheaton_latitude = Wheaton.latitude
wheaton_longitude = Wheaton.longitude

silver_spring_address = 'Silver Spring, MD'
geolocator_silver_spring = Nominatim(user_agent="SilverSpring_explorer")
SilverSpring = geolocator_silver_spring.geocode(silver_spring_address)
silver_spring_latitude = SilverSpring.latitude
silver_spring_longitude = SilverSpring.longitude

print('The geographical coordinates of Bethesda are {}, {}.'.format(latitude_bethesda, longitude_bethesda))
print('The geographical coordinates of Germantown are {}, {}.'.format(germantown_latitude, germantown_longitude))
print('The geographical coordinates of Gaithersburg are {}, {}.'.format(gaithersburg_latitude, gaithersburg_longitude))
print('The geographical coordinates of Rockville are {}, {}.'.format(rockville_latitude, rockville_longitude))
print('The geographical coordinates of Wheaton are {}, {}.'.format(wheaton_latitude, wheaton_longitude))
print('The geographical coordinates of Silver Spring are {}, {}.'.format(silver_spring_latitude, silver_spring_longitude))
#Store coordinates in corresponding district
rockville_latitude = 39.0840054 # neighborhood latitude value
rockville_longitude = -77.1527573 # neighborhood longitude value
district_1 = 'Rockville, MD' # neighborhood name

bethesda_latitude = 38.9846988 
bethesda_longitude = -77.0945393 
district_2 = 'Bethesda, MD' 

silver_spring_latitude = 38.9959461 
silver_spring_longitude = -77.0276231 
district_3 = 'Silver Spring, MD' 

wheaton_latitude = 39.0398314 
wheaton_longitude = -77.0552555 
district_4 = 'Wheaton, MD'

germantown_latitude = 39.1731621 
germantown_longitude = -77.2716502 
district_5 = 'Germantown, MD' 

gaithersburg_montgomery_village_latitude = 39.1434406 
gaithersburg_montgomery_village_longitude = -77.2013705 
district_6 = 'Gaithersburg / Montgomery Village, MD' 
#Retrieve the top 100 venues in each district

LIMIT = 100 # Limit of number of venues returned by Foursquare API

radius_bethesda = 2000 # Define radius
radius_germantown = 2000
radius_rockville = 2000
radius_silver = 1000
radius_wheaton = 2000
radius_gaithersburg = 2000

bethesda_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    bethesda_latitude, 
    bethesda_longitude, 
    radius_bethesda, 
    LIMIT)

germantown_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    germantown_latitude, 
    germantown_longitude, 
    radius_germantown, 
    LIMIT)

rockville_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    rockville_latitude, 
    rockville_longitude, 
    radius_rockville, 
    LIMIT)

silverspring_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    silver_spring_latitude, 
    silver_spring_longitude, 
    radius_silver, 
    LIMIT)

wheaton_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    wheaton_latitude, 
    wheaton_longitude, 
    radius_wheaton, 
    LIMIT)

gaithersburg_url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    gaithersburg_latitude, 
    gaithersburg_longitude, 
    radius_gaithersburg, 
    LIMIT)


print (bethesda_url) 
print (germantown_url)
print (rockville_url) 
print (silverspring_url)
print (wheaton_url) 
print (gaithersburg_url)
#Send the GET request and examine the results
results = requests.get(bethesda_url).json()
results2 = requests.get(germantown_url).json()
results3 = requests.get(rockville_url).json()
results4 = requests.get(silverspring_url).json()
results5 = requests.get(wheaton_url).json()
results6 = requests.get(gaithersburg_url).json()

#print(results, results2, results3, results4, results5, results6)
print ("Requests completed")
#Create function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
#Bethesda
#Clean the json and structure it into a pandas dataframe.
venues = results['response']['groups'][0]['items']  
bethesda_venues = pd.json_normalize(venues) # flatten JSON
# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
bethesda_venues =bethesda_venues.loc[:, filtered_columns]
# filter the category for each row
bethesda_venues['venue.categories'] = bethesda_venues.apply(get_category_type, axis=1)
# clean columns
bethesda_venues.columns = [col.split(".")[-1] for col in bethesda_venues.columns]
# Add District Column
bethesda_venues['District'] = '2nd District, Bethesda'
#bethesda_venues.set_index('District', inplace=True)
#print(bethesda_venues.shape)
#bethesda_venues.head()

#Germantown
venues2 = results2['response']['groups'][0]['items']  
germantown_venues = pd.json_normalize(venues2) # flatten JSON
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
germantown_venues =germantown_venues.loc[:, filtered_columns]
germantown_venues['venue.categories'] = germantown_venues.apply(get_category_type, axis=1)
germantown_venues.columns = [col.split(".")[-1] for col in germantown_venues.columns]
germantown_venues['District'] = '5th District, Germantown'
#germantown_venues.set_index('District', inplace=True)
#print(germantown_venues.shape)
#germantown_venues.head()

#Rockville
venues3 = results3['response']['groups'][0]['items'] 
rockville_venues = pd.json_normalize(venues3) # flatten JSON
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
rockville_venues = rockville_venues.loc[:, filtered_columns]
rockville_venues['venue.categories'] = rockville_venues.apply(get_category_type, axis=1)
rockville_venues.columns = [col.split(".")[-1] for col in rockville_venues.columns]
rockville_venues['District'] = '1st District, Rockville'

#Silver Spring
venues4 = results4['response']['groups'][0]['items']
silverspring_venues = pd.json_normalize(venues4) # flatten JSON
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
silverspring_venues = silverspring_venues.loc[:, filtered_columns]
silverspring_venues['venue.categories'] = silverspring_venues.apply(get_category_type, axis=1)
silverspring_venues.columns = [col.split(".")[-1] for col in silverspring_venues.columns]
silverspring_venues['District'] = '3rd District, Silver Spring'

#Wheaton
venues5 = results5['response']['groups'][0]['items']
wheaton_venues = pd.json_normalize(venues5) # flatten JSON
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
wheaton_venues = wheaton_venues.loc[:, filtered_columns]
wheaton_venues['venue.categories'] = wheaton_venues.apply(get_category_type, axis=1)
wheaton_venues.columns = [col.split(".")[-1] for col in wheaton_venues.columns]
wheaton_venues['District'] = '4th District, Wheaton'

#Gaithersburg
venues6 = results6['response']['groups'][0]['items']
gaithersburg_venues = pd.json_normalize(venues6) # flatten JSON
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
gaithersburg_venues = gaithersburg_venues.loc[:, filtered_columns]
gaithersburg_venues['venue.categories'] = gaithersburg_venues.apply(get_category_type, axis=1)
gaithersburg_venues.columns = [col.split(".")[-1] for col in gaithersburg_venues.columns]
gaithersburg_venues['District'] = '6th District, Gaithersburg / Montgomery Village'

print("1st District, Rockville dimensions =", rockville_venues.shape)
print("2nd District, Bethesda dimensions =", bethesda_venues.shape)
print("3rd District, Silver Spring dimensions =", silverspring_venues.shape)
print("4th District, Wheaton dimensions =", wheaton_venues.shape)
print("5th District, Germantown dimensions =", germantown_venues.shape)
print("6th District, Gaithersburg dimensions =", gaithersburg_venues.shape)
# Display number of unique venue categories in each district
print('There are {} unique categories in the 1st District, Rockville.'.format(len(rockville_venues['categories'].unique())))
#print (rockville_venues.groupby('categories')['name'].count().head())
#print('------------------------------')
print('There are {} unique categories in the 2nd District, Bethesda.'.format(len(bethesda_venues['categories'].unique())))
#print (bethesda_venues.groupby('categories')['name'].count().head())
#print('------------------------------')
print('There are {} unique categories in the 3rd District, Silver Spring.'.format(len(silverspring_venues['categories'].unique())))
#print (silverspring_venues.groupby('categories')['name'].count().head())
#print('------------------------------')
print('There are {} unique categories in the 4th District, Wheaton.'.format(len(wheaton_venues['categories'].unique())))
#print (wheaton_venues.groupby('categories')['name'].count().head())
#print('------------------------------')
print('There are {} unique categories in the 5th District, Germantown.'.format(len(germantown_venues['categories'].unique())))
#print (germantown_venues.groupby('categories')['name'].count().head())
#print('------------------------------')
print('There are {} unique categories in the 6th District, Gaithersburg.'.format(len(gaithersburg_venues['categories'].unique())))
#print (gaithersburg_venues.groupby('categories')['name'].count().head())

#Check for the number unique categories in all of the six districts combined
frames = [rockville_venues, bethesda_venues, silverspring_venues, wheaton_venues, germantown_venues, gaithersburg_venues]

Combined_venues = pd.concat(frames).reset_index(drop=True)

Combined_venues = Combined_venues[['District', 'name', 'categories', 'lat', 'lng']]
print('There are collectively {} unique categories in Districts 1-6.'.format(len(Combined_venues['categories'].unique())))
# one hot encoding
Combined_onehot = pd.get_dummies(Combined_venues[['categories']], prefix="", prefix_sep="")

# add District column back to dataframe
Combined_onehot['YDistrict'] = Combined_venues['District']

Combined_onehot.columns[-1]

# move neighborhood column to the first column
fixed_columns = [Combined_onehot.columns[-1]] + list(Combined_onehot.columns[:-1])
Combined_onehot = Combined_onehot[fixed_columns]
Combined_onehot.rename(columns={'YDistrict': 'District'}, inplace=True)

print (Combined_onehot.shape)
Combined_onehot.head()
#Create new dataframe
Combined_grouped = Combined_onehot.groupby('District').mean().reset_index()
Combined_grouped
num_top_venues = 10

for hood in Combined_grouped['District']:
    #print("----"+District+"----")
    temp = Combined_grouped[Combined_grouped['District'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
# Sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

# Create a new dataframe and display the top 10 venues for each neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# Create columns according to number of top venues
columns = ['District']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# Create a new dataframe
Combined_venues_sorted = pd.DataFrame(columns=columns)
Combined_venues_sorted['District'] = Combined_grouped['District']

for ind in np.arange(Combined_grouped.shape[0]):
    Combined_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Combined_grouped.iloc[ind, :], num_top_venues)

Combined_venues_sorted.head(10)
# set number of clusters
kclusters = 6

Combined_grouped_clustering = Combined_grouped.drop('District', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Combined_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# add clustering labels
Combined_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Combined_merged = Combined_venues

# merge Combined_grouped with toronto_data to add latitude/longitude for each neighborhood
Combined_merged = Combined_merged.join(Combined_venues_sorted.set_index('District'), on='District')

Combined_merged.head() # check the last columns!
Combined_venues_sorted.head(10)
# Create map
latitude = 39.0840 
longitude = -77.152

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11,tiles='cartodbpositron')

# Set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# Add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Combined_merged['lat'], Combined_merged['lng'], Combined_merged['District'], Combined_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + 'Cluster' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster -1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
      
map_clusters
# Create map of Montgomery County, Maryland
Venues = folium.Map(location = [latitude, longitude], zoom_start = 11)

# instantiate a mark cluster object for the incidents in the dataframe
businesses = plugins.MarkerCluster().add_to(Venues)

# loop through the dataframe and add each data point to the mark cluster
for lat, lng, label, in zip(Combined_venues.lat, Combined_venues.lng, Combined_venues.categories):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(businesses)

# display map
Venues
venues_heat = folium.Map(location = [latitude, longitude], zoom_start = 11, tiles='cartodbpositron')

# Mark each incident as a point
for index, row in Combined_venues.iterrows():
    folium.Circle([row['lat'], row['lng']],
                        radius=15,
                        popup=row['categories'],
                        #fill_color="#3db7e4",
                        fill_color="red"  
                       ).add_to(venues_heat)

# Convert to (n, 2) nd-array format for heatmap
violationsArr = Combined_venues[['lat', 'lng']].values

# plot heatmap
venues_heat.add_child(plugins.HeatMap(violationsArr, radius=18))
folium.LayerControl().add_to(venues_heat)
# Display map
venues_heat
Combined_merged.loc[Combined_merged['Cluster Labels'] == 0, Combined_merged.columns[[0] + list(range(5, Combined_merged.shape[1]))]].head(1)
Combined_merged.loc[Combined_merged['Cluster Labels'] == 1, Combined_merged.columns[[0] + list(range(5, Combined_merged.shape[1]))]].head(1)
Combined_merged.loc[Combined_merged['Cluster Labels'] == 2, Combined_merged.columns[[0] + list(range(5, Combined_merged.shape[1]))]].head(1)
Combined_merged.loc[Combined_merged['Cluster Labels'] == 3, Combined_merged.columns[[0] + list(range(5, Combined_merged.shape[1]))]].head(1)
Combined_merged.loc[Combined_merged['Cluster Labels'] == 4, Combined_merged.columns[[0] + list(range(5, Combined_merged.shape[1]))]].head(1)
Combined_merged.loc[Combined_merged['Cluster Labels'] == 5, Combined_merged.columns[[0] + list(range(5, Combined_merged.shape[1]))]].head(1)