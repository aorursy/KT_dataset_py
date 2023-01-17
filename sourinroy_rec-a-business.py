# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors
import json
import requests
import ipywidgets as widgets
from pandas.io.json import json_normalize
from geopy.distance import great_circle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_delhi = pd.read_csv('/kaggle/input/delhi-neighborhood-data/delhi_dataSet.csv',index_col = 'Unnamed: 0')
df_delhi.head()
df_delhi.shape
df_delhi.isnull().sum()
df_present = df_delhi.where(df_delhi['latitude'].isnull() == False)
df_present.dropna(subset = ['Borough','Neighborhood'],inplace=True)
df_present.reset_index(inplace=True)
df_present.drop(['index'], axis=1,inplace=True)
df_present.head()
df_missing = df_delhi.where(df_delhi['latitude'].isnull() == True)
df_missing.dropna(subset = ['Borough','Neighborhood'],inplace=True)
df_missing.reset_index(inplace=True)
df_missing.drop(['index'], axis=1,inplace=True)
df_missing.head()
df_missing['Neighborhood'].replace('Sundar Nagar[1]' ,'Sundar Nagar',inplace=True)
df_missing.head()
# defining series lat and lng for assigning the columns
lat = pd.Series([],dtype=float)
lng = pd.Series([],dtype=float)

#Assigning the data

#Bugum Pur
lat[0], lng[0] = 28.727248, 77.064975 

#Rohini Sub City
lat[1],lng[1] = 28.741073, 77.082574

#Ghantewala is a famous sweet shop, that got closed, so I am dropping this data
df_missing.drop([2],inplace=True)

#Gulabi Bagh
lat[3],lng[3] = 28.672190, 77.191620

#Sadar Bazaar 
lat[4],lng[4] = 28.659395, 77.212782

#Tees Hazari
lat[5],lng[5] = 28.665682, 77.216413

#New Usmanpur
lat[6],lng[6] = 28.677737, 77.256637

#Sadatpur
lat[7],lng[7] = 28.726746, 77.261097

#Rajender Nagar
lat[8],lng[8] = 28.641024, 77.185038

#Sadar Bazaar
lat[9],lng[9] = 28.657305, 77.212750

#Laxmibai Nagar
lat[10],lng[10] = 28.575276, 77.209630

#Silampur
lat[11],lng[11] = 28.664181, 77.270916

#Jamroodpur Village
lat[12],lng[12] = 28.557592, 77.237061

#Kotla Mubarakpur
lat[13],lng[13] = 28.575783, 77.227396

#Pulpehaladpur
lat[14],lng[14] = 28.499831, 77.290347

#Sundar Nagar
lat[15],lng[15] = 28.601985, 77.243725

#Dabri
lat[16],lng[16] = 28.611823, 77.087268

#Dwarka Sub City
lat[17],lng[17] = 28.582154, 77.049576

#Sagar Pur
lat[18],lng[18] = 28.605670, 77.099189

#Partap Nagar
df_missing.drop([19],inplace=True)

#Tihar Village
lat[20],lng[20] = 28.634353, 77.107331

#Uttam Nagar
lat[21],lng[21] = 28.619573, 77.054916
df_missing['latitude_mod'] = lat
df_missing['longitude_mod'] = lng
df_missing.reset_index(inplace=True)
df_missing.drop(columns=['latitude', 'longitude','index'],inplace=True)
df_missing.rename(columns={'latitude_mod' : 'latitude' , 'longitude_mod' : 'longitude' }, inplace=True)
df_missing.head()
frames = [df_missing,df_present]
df = pd.concat(frames)
df.head()
df2 = df[df['latitude'] < 30]
df1 = df[df['latitude']>30]
df1.reset_index(inplace = True)
df1.drop(['index'],axis=1,inplace=True)
cor_lat = pd.Series([],dtype = float)
cor_lng = pd.Series([],dtype = float)

#Assigning the data

#Dhaka
cor_lat[0], cor_lng[0] = 28.709278, 77.206249

#Model Town
cor_lat[1],cor_lng[1] = 28.720445, 77.192263

#Pul Bangash
cor_lat[2], cor_lng[2] = 28.666822, 77.206229

#Shastri Nagar , ND
cor_lat[3],cor_lng[3] = 28.675370, 77.181173

#Gandhi Nagar
cor_lat[4], cor_lng[4] = 28.666734, 77.274094 

#Shastri Nagar , ED
cor_lat[5],cor_lng[5] = 28.647588, 77.273977

#Govindpuri
cor_lat[6], cor_lng[6] = 28.535112, 77.263185 

#Kalkaji
cor_lat[7],cor_lng[7] = 28.540973, 77.259523

#Ashok Nagar
cor_lat[8], cor_lng[8] = 28.637112, 77.102459

#Nehru Nagar
cor_lat[9],cor_lng[9] = 28.663061, 77.169519
df1['latitude_mod'] = cor_lat
df1['longitude_mod'] = cor_lng
df1.drop(['latitude','longitude'],axis=1,inplace=True)
df1.rename(columns={'latitude_mod' : 'latitude' , 'longitude_mod' : 'longitude' }, inplace=True)
df1.head()
frames = [df1,df2]
df = pd.concat(frames)
df.reset_index(inplace=True)
df.drop(['index'],axis=1,inplace=True)
df.head()
df.shape
#the dataframe has 183 rows with 4 columns
df.columns
#the column names are in order
df.isnull().sum()
#there are no missing values
CLIENT_ID = 'HSKSPYIMS3JWAL4IJ3IWQT1MOLT04ITR1UTIPF3OYZ220JXF' # Foursquare ID
CLIENT_SECRET = 'YS1P0XFBMDIERG42KRRFGQHG4AYWYC4LBNIKTWLFXBLEKDKN' # Foursquare Secret
VERSION = '20200515' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
df_venues = pd.DataFrame(columns = ['Venue Category'])
for j in range(0,183):
    
    try:
        lat,lng = df.iloc[j][2] , df.iloc[j][3]
        TIME = 'any'
        radius = 1000
        LIMIT = 50
        url =  'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&time={}&day={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT,
            TIME,
            TIME)

        results = requests.get(url).json()
        venues = results['response']['groups'][0]['items']
        nearby_venues = pd.json_normalize(venues)
        filtered_columns = ['venue.name', 'venue.categories']
        nearby_venues =nearby_venues.loc[:, filtered_columns]
        nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)
        nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
        
        l = nearby_venues.shape[0]    
        for i in range(0,l):
            df_venues = df_venues.append({'Venue Category' : nearby_venues.iloc[i][1]} , 
                                           ignore_index=True)
    
    except:
        
        continue


df_venues['frequency'] = 1
df_venues = df_venues.groupby('Venue Category').count()
df_venues.head()
df_venues.drop(['ATM','Airport','Airport Terminal','Bus Station','Campground','College Cafeteria','Cricket Ground',
                'Farmers Market','Historic Site','History Museum','IT Services','Lake','Light Rail Station',
                'Molecular Gastronomy Restaurant','Museum','Park','Pool','Road',
                'Stadium','Temple','Tourist Information Center','Trail','Train Station','University'], axis = 0,inplace = True)
df_venues.drop(['Metro Station','Other Great Outdoors','Nightlife Spot','Farm','Track','Astrologer','Zoo'], axis= 0,inplace= True)
df_venues.sort_values(['frequency'],axis=0,ascending = False,inplace=True)
df_venues.reset_index(inplace=True)
df_venues.head(10)
#Merging Cafe and Coffee Shop into one and re-sorting the data

df_venues.replace({'Café' : 'Coffee Shop' , 202 : (202+132)} , inplace = True)
df_venues.drop([4],inplace=True)
df_venues.sort_values(['frequency'],axis=0,ascending = False,inplace=True)
df_venues.reset_index(inplace=True)
df_venues.drop(['index'],axis=1,inplace = True)
df_venues
df_venues.shape
# The dataframe has 182 rows with 2 columns
df_venues.isnull().sum().sum()
# The dataframe contains no missing values
bor = 'North West Delhi'
drop_down = widgets.Dropdown(options=df['Borough'].unique(),
                                description='Borough',
                                disabled=False)

def dropdown_handler(change):
    global bor
    bor = change.new

drop_down.observe(dropdown_handler, names='value')
display(drop_down)
opt = []
for i in range(0,173):
    if (df.iloc[i][0] == bor):
        opt.append(df.iloc[i][1])

ngbor = ''
drop_down = widgets.Dropdown(options=opt,
                                description='Neighborhood',
                                disabled=False)

def dropdown_handler(change):
    global ngbor
    ngbor = change.new

drop_down.observe(dropdown_handler, names='value')
display(drop_down)
for i in range(0,df.shape[0]):
    if (df.iloc[i][0] == bor and df.iloc[i][1] == ngbor):
        ven_lat = df.iloc[i][2]
        ven_lng = df.iloc[i][3]
try:
    cnt = 0
    radius = 1000
    LIMIT = 50
    url =  'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
        CLIENT_ID, 
        CLIENT_SECRET, 
        VERSION, 
        ven_lat, 
        ven_lng, 
        radius, 
        LIMIT)

    results = requests.get(url).json()
    venues = results['response']['groups'][0]['items']
    nearby_venues = pd.json_normalize(venues)
    filtered_columns = ['venue.name', 'venue.categories']
    nearby_venues =nearby_venues.loc[:, filtered_columns]
    nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)
    nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

except:
    
    cnt = 1
if(cnt == 0):
    
    ven_business = nearby_venues.categories.unique()
    rcs = []
    counter = 0

    l = df_venues.shape[0]
    vl = len(ven_business)
    for i in range(0,l):
        for j in range(0,vl):
            if(df_venues.iloc[i][0] == ven_business[j]):
                counter = 1 
                break

        if(counter == 0):
            rcs.append(df_venues.iloc[i][0])
        else:
            counter = 0

else:
    
    cnt = 0
    rcs = df_venues['Venue Category']
print("For the user selected neighborhood at - " + bor + ", " + ngbor + " Following are the 10 best reccomendations for new business ideas that could be set up - ")
print()

for i in range(0,10):
    print(rcs[i])
a = 'Popular business ideas'
b = 'Facny business ideas'

choice = a
drop_down = widgets.Dropdown(options= [a,b],
                                description='Choose - ',
                                disabled=False)

def dropdown_handler(change):
    global choice
    choice = change.new

drop_down.observe(dropdown_handler, names='value')
display(drop_down)
if(choice == a):
    chc = df_venues["Venue Category"][:25]
else:
    chc = df_venues['Venue Category'][:-25:-1]
b_choice = ''
drop_down = widgets.Dropdown(options= chc,
                                description='Choose - ',
                                disabled=False)

def dropdown_handler(change):
    global b_choice
    b_choice = change.new

drop_down.observe(dropdown_handler, names='value')
display(drop_down)
bor = 'North West Delhi'
drop_down = widgets.Dropdown(options=df['Borough'].unique(),
                                description='Borough',
                                disabled=False)

def dropdown_handler(change):
    global bor
    bor = change.new

drop_down.observe(dropdown_handler, names='value')
display(drop_down)
opt = []
for i in range(0,df.shape[0]):
    if (df.iloc[i][0] == bor):
        opt.append(df.iloc[i][1])

ngbor = ''
drop_down = widgets.Dropdown(options=opt,
                                description='Neighborhood',
                                disabled=False)

def dropdown_handler(change):
    global ngbor
    ngbor = change.new

drop_down.observe(dropdown_handler, names='value')
display(drop_down)
df_dis = pd.DataFrame(columns = ['Neighborhood','Distance in km' , 'latitude' , 'longitude'])

for i in range(0,df.shape[0]):
    if (df.iloc[i][0] == bor and df.iloc[i][1] == ngbor):
        ven_lat = df.iloc[i][2]
        ven_lng = df.iloc[i][3]
dis = []
lat = []
lng = []
neighbor = pd.Series(opt,dtype = object)
df_dis['Neighborhood'] = neighbor

for i in range(0,df.shape[0]):
    if(df.iloc[i][0] == bor):
        it_neighborhood = (df.iloc[i][2] , df.iloc[i][3])
        lat.append(df.iloc[i][2])
        lng.append(df.iloc[i][3])
        selected_neighborhood = (ven_lat, ven_lng) 
        dist = great_circle(it_neighborhood,selected_neighborhood).km
        dist = format(dist, '.3f')
        dis.append(dist)

geo_dist = pd.Series(dis,dtype = float)
df_dis['Distance in km'] = geo_dist

df_dis['latitude'] = pd.Series(lat,dtype=float)
df_dis['longitude'] = pd.Series(lng,dtype=float)
        
df_dis.sort_values(['Distance in km'],axis = 0,ascending = True , inplace = True)
df_dis.reset_index(inplace = True)
df_dis.drop(['index'],axis=1,inplace=True)
for j in range(0,df_dis.shape[0]):
    
    try:
        counter = 0
        radius = 1000
        LIMIT = 50
        ven_lat = df_dis.iloc[j][2]
        ven_lng = df_dis.iloc[j][3]
        url =  'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            ven_lat, 
            ven_lng, 
            radius, 
            LIMIT)

        results = requests.get(url).json()
        venues = results['response']['groups'][0]['items']
        nearby_venues = pd.json_normalize(venues)
        filtered_columns = ['venue.name', 'venue.categories']
        nearby_venues =nearby_venues.loc[:, filtered_columns]
        nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)
        nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
        
        ven_buss = pd.Series( (nearby_venues['categories'].unique()) ,dtype=object )
        v_length = ven_buss.shape[0]
        
        for k in range(0,v_length):
            if(b_choice == ven_buss[k]):
                counter = 1
                break
        
        if(counter == 0):
            
            print("The optimal place to start your business, " + b_choice + " is in - " + df_dis.iloc[j][0] + ", " + bor)
            break
        
        
    except:
        
        print("The optimal place to start your business, " + b_choice + " is in - " + df_dis.iloc[j][0] + ", " + bor)
    