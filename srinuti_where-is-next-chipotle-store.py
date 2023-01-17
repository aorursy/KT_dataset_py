

from IPython.display import Image

from IPython.core.display import HTML 



url = 'https://mma.prnewswire.com/media/1003669/Chipotle_Plant_Powered.jpg?p=publish'

Image(url= url, width=1000, height=300, unconfined=True)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
loc_df = pd.read_csv('/kaggle/input/chipotle-locations/chipotle_stores.csv')



#The dataset contains population data county wise. 

df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')

df_sample.replace('Doña Ana County', 'Dona Ana County', inplace= True) # the name has forigen character, difficult to search



#Import dataset containing the Median Household income county wise

url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

county_data = f'{url}/us_county_data.csv'

df = pd.read_csv(county_data, na_values=[' '])

!pip install zipcodes
import numpy as np

import pandas as pd

import geopandas as gpd

from shapely.geometry import Point, Polygon



import zipcodes as zp

#import networkx as nx



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import sys



import plotly.express as px



if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")

    

from geopy.distance import geodesic as gdist



from plotly.offline import iplot, init_notebook_mode, plot, download_plotlyjs

init_notebook_mode()



import plotly.graph_objects as go

import plotly.figure_factory as ff



from plotly import tools

from plotly.subplots import make_subplots



loc_df.head()
df_sample.head()
df.head()
# Zipcode column addition extract info from address column. 



loc_df['zip_code'] = loc_df['address'].apply(lambda x: x[-8:-3])



# adding lat and long data from zipcodes into loc_df. 



loc_df['zip_lat'] = loc_df['zip_code'].apply(lambda x: zp.matching(x)[0]['lat'])

loc_df['zip_long'] = loc_df['zip_code'].apply(lambda x: zp.matching(x)[0]['long'])



# adding county details from the zipcodes into loc_df. 

loc_df['county_name'] = loc_df['zip_code'].apply(lambda x:zp.matching(x)[0]['county'])



# convert datatype from object to float. 

loc_df['zip_lat'] =loc_df['zip_lat'].astype(float)

loc_df['zip_long'] =loc_df['zip_long'].astype(float)



# extract the population values from the df_sample frame.

pop_list = []

fips_list = []

for j in range(0,2629):

    county = loc_df['county_name'][j]

    state = loc_df['state'][j]

    if state == 'Washington DC':

        state = county

    pop_list.append(list(df_sample[(df_sample['CTYNAME'] == county) & (df_sample['STNAME'] == state)]['TOT_POP'].values)[0])

    fips_list.append(list(df_sample[(df_sample['CTYNAME'] == county) & (df_sample['STNAME'] == state)]['FIPS'].values)[0])



#update the dataframe with new column. 

loc_df['total_population'] = pd.Series(pop_list)

loc_df['FIPS'] = pd.Series(fips_list)



#grab median house hold income 2011 data from the df dataframe into loc_df dataframe. 

loc_df['Median_income_2011'] = loc_df['FIPS'].apply(lambda x: list(df[df['FIPS_Code'] == x]['Median_Household_Income_2011'].values)[0])

df_county = loc_df.groupby(['county_name','state', 

                            'total_population', 'FIPS',

                           'Median_income_2011']).size().reset_index(name = 'counts')



df_county.sort_values(['counts'], axis=0, ascending=False, inplace=True, ignore_index=True)

df_county['density'] = round(df_county['total_population']/(df_county['counts']*10000),0)



# Market Opportunity

df_opp = df_sample[df_sample['TOT_POP'] >= 125000]



opp_set = set(list(df_opp['CTYNAME'].values))

current_set = set(list(df_county['county_name'].values))



# markets that need attention. 

new_market = opp_set - current_set

#print(len(new_market))



# new market oppurunity 

filter_market = list(new_market)



state_lst = []

county_lst = []

pop_lst= []

fips_lst = []

for n in filter_market:

    fips_lst.append(df_opp[df_opp['CTYNAME']== n].values[0][0])

    state_lst.append(df_opp[df_opp['CTYNAME']== n].values[0][1])

    county_lst.append(df_opp[df_opp['CTYNAME']== n].values[0][2])

    pop_lst.append(df_opp[df_opp['CTYNAME']== n].values[0][3])



new_market_data = {'FIPS': fips_lst, 'state': state_lst,'county_name': county_lst,'total_population': pop_lst}

#row_index = ['state', 'county_name', 'total_population']

df_newmarket = pd.DataFrame(new_market_data, columns= ['FIPS','state', 'county_name', 'total_population'])



#grab the median income data from df 

df_newmarket['Median_income_2011'] = df_newmarket['FIPS'].apply(lambda x: list(df[df['FIPS_Code'] == x]['Median_Household_Income_2011'].values)[0])

df_newmarket.sort_values(['total_population'], axis=0, ascending=False, inplace=True, ignore_index=True)





# adding points data into loc_df using latitude and longitude columns as a tuple. 

long = loc_df['longitude'].to_list()

lat = loc_df['latitude'].to_list()



zip_lng = loc_df['zip_long'].to_list()

zip_lat = loc_df['zip_lat'].to_list()



points_loc = []

points_zip = []



for i, t in enumerate(zip(lat, long)):

    points_loc.append(t)



for n, k in enumerate(zip(zip_lat, zip_lng)):

    points_zip.append(k)

    



loc_df['points_loc'] = pd.Series(points_loc)

loc_df['points_zip'] = pd.Series(points_zip)



df_county.sort_values(['density'], axis=0, ascending=False, inplace=True, ignore_index=True)

df_county_plot = df_county.copy()

#df_county_plot.head()





trace =go.Scatter(

    x=df_county_plot['counts'].values,

    y=df_county_plot['density'].values,

    mode='markers',

     marker=dict(

         color=df_county_plot['counts'].values,

         size=(df_county_plot['counts'].values),

         showscale=True

         )

)



data = [trace]



layout = { 'title': 'Correlation btw store counts and population density (10k population/per store)', 

          'xaxis': {'title': 'counts', 'zeroline': False }, 

          'yaxis': {'title': 'population density', 'zeroline': False }   



}



iplot({'data': data, 'layout': layout})
df_county_plot_focus = df_county_plot[df_county_plot['density'] >= 25]

#print(df_county_plot_focus['counts'].value_counts().sum())



trace =go.Scatter(

    x=df_county_plot_focus['counts'].values,

    y=df_county_plot_focus['density'].values,

    mode='markers',

     marker=dict(

         color=df_county_plot_focus['counts'].values,

         size=(df_county_plot_focus['counts'].values)*10,

         showscale=True

         )

)



data = [trace]



layout = { 'title': 'Stores that cater more than 0.25 million population', 

          'xaxis': {'title': 'counts', 'zeroline': False }, 

          'yaxis': {'title': 'population density', 'zeroline': False }   

    

}



iplot({'data': data, 'layout': layout})
# Store distribution statewise

counts = list(loc_df['state'].value_counts().values)

states = list(loc_df['state'].value_counts().index)



data = {'states': states, 'counts': counts}

df_state = pd.DataFrame(data=data)

#df_state.head()



fig = px.treemap(df_state.sort_values(by = 'counts', ascending= False).reset_index(drop = True),

                         path = ['states'], values= 'counts', height = 700,

                         title = 'Most number of stores state wise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
fig = go.Figure(data=go.Scattergeo(

        lon = loc_df['longitude'],

        lat = loc_df['latitude'],

        text = loc_df['location'],

        mode = 'markers',

        marker_color = 'green',

        

        ))



fig.update_layout(

        title = 'Chiplote stores throughout USA',

        geo_scope='usa',

    )

fig.show()
# Store distribution statewise

counts = list(loc_df['state'].value_counts().values)

states = list(loc_df['state'].value_counts().index)



data = {'states': states, 'counts': counts}

df_state = pd.DataFrame(data=data)

df_state = df_state.tail(10)



fig = px.treemap(df_state.sort_values(by = 'counts', ascending= False).reset_index(drop = True),

                         path = ['states'], values= 'counts', height = 700,

                         title = 'least number of stores state wise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
# Store distribution location wise

counts = list(loc_df['location'].value_counts().values)

location = list(loc_df['location'].value_counts().index)



data_loc = {'location': location, 'counts': counts}

df_location = pd.DataFrame(data=data_loc)

df_loc_plot = df_location.head(30)



fig = px.treemap(df_loc_plot.sort_values(by = 'counts', ascending= False).reset_index(drop = True),

                         path = ['location'], values= 'counts', height = 700,

                         title = 'Most number of stores location wise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
#county wise data

df_county.sort_values(['counts'], axis=0, ascending=False, inplace=True, ignore_index=True)

df_county_plot = df_county.head(10)





fig = px.treemap(df_county_plot.sort_values(by = 'counts', ascending= False).reset_index(drop = True),

                         path = ['county_name'], values= 'counts', height = 700,

                         title = 'Most stores county wise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
url = 'https://www.incimages.com/uploaded_files/image/1920x1080/getty_510222832_346191.jpg'

Image(url= url, width=1000, height=300, unconfined=True)
#county wise data

df_top10 = df_newmarket.head(10)

fig = px.treemap(df_top10.sort_values(by = 'total_population', ascending= False).reset_index(drop = True),

                         path = ['state', 'Median_income_2011'], values= 'total_population', height = 700,

                         title = 'Top 10 opportunities statewise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()
df_top10 = df_newmarket.head(10)

fig = px.treemap(df_top10.sort_values(by = 'total_population', ascending= False).reset_index(drop = True),

                         path = ['county_name', 'Median_income_2011'], values= 'total_population', height = 700,

                         title = 'Top 10 opportunities countywise',

                         color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label + text+ value'



fig.show()

#function to filter the dataframe points closest to the given coordinate. 

def short_data(data, zipcode):

    '''

    Input:- 1. Dataframe containing 'longitude' and 'latitude' arrays. Names should match. 

    2. Gps coordinates of given point lat and long

    

    Output:- Shortlisted dataframe containing closest points within 25mile radius.

    '''

    delta_lat = 0.3628 # this is for 25mile radius

    delta_long = 0.4360 # this is for 25mile radius

    

    lat, long = (float(zp.matching(zipcode)[0]['lat']), float(zp.matching(zipcode)[0]['long']))

    

    #find the limits of search. 

    lat_up = lat + delta_lat

    lat_dwn = lat - delta_lat

    long_up = long + delta_long

    long_dwn = long - delta_long

    

    short_data = data[(lat_up >= data['latitude']) & (lat_dwn <= data['latitude']) & 

      (long_up >= data['longitude']) & (long_dwn <= data['longitude'])]

    

    # if the filtered data does not contain any point closest to 25 mile radius. 

    if short_data.shape[0] <=0:

        print('No points exists close to the given coordinates')

        

    else:

        pass

    

    return short_data  
test = short_data(loc_df,'94105')



fig = go.Figure(data=go.Scattergeo(

        lon = test['longitude'],

        lat = test['latitude'],

        text = test['location'],

        mode = 'markers',

        marker_color = 'green',

        

        ))



fig.update_layout(

        title = 'Chiplote stores within 25miles to zipcode',

        geo_scope='usa',

    )

fig.show()



print('Number of stores near zipcode', test.shape[0])



'''

credits:-

fellow kagglers @Surya sai teja desu, @C4rl05/V, @MAD

'''