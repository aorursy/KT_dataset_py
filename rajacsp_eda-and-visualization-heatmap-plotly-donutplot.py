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
FILEPATH = '/kaggle/input/consumer-complaints-financial-products/Consumer_Complaints.csv'
df = pd.read_csv(FILEPATH)
df.info()
df.describe()
df.head()
df.isnull().sum()
df.isnull().sum().sum()
import missingno as miss
miss.matrix(df)
miss.heatmap(df)
miss.dendrogram(df)
miss.bar(df)
df.columns
len(df.columns)
# How many products



df['Product'].unique()
len(df['Product'].unique())
import matplotlib.pyplot as plt



def show_donut_plot(col):

    

    rating_data = df.groupby(col)[['Complaint ID']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Complaint ID']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot by ' +str(col), loc='center')

    

    plt.show()
show_donut_plot('Product')
show_donut_plot('Submitted via')
show_donut_plot('State')
import squarify



def show_treemap(col):

    df_type_series = df.groupby(col)['Complaint ID'].count().sort_values(ascending = False).head(20)



    type_sizes = []

    type_labels = []

    for i, v in df_type_series.items():

        type_sizes.append(v)

        

        type_labels.append(str(i) + ' ('+str(v)+')')





    fig, ax = plt.subplots(1, figsize = (12,12))

    squarify.plot(sizes=type_sizes, 

                  label=type_labels[:10],  # show labels for only first 10 items

                  alpha=.2 )

    plt.title('TreeMap by '+ str(col))

    plt.axis('off')

    plt.show()
show_treemap('Product')
show_treemap('State')
show_treemap('Sub-issue')
show_treemap('Tags')
def show_pie_plot(col):

    

    rating_data = df.groupby(col)[['Complaint ID']].count()



    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Complaint ID']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)

    

    plt.axis('equal')



    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    

    plt.legend(cols)

    

    plt.title('Pie Plot by ' +str(col), loc='center')

    

    plt.show()
show_pie_plot('Timely response?')
show_pie_plot('Consumer disputed?')
# Show death counts by country

import seaborn as sns



print(df['Timely response?'].value_counts())



ax = sns.barplot(

    x = df['Timely response?'].value_counts().keys(), 

    y = df['Timely response?'].value_counts().values

)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

plt.show()
def show_donut_plot_by_state(col, state):

    

    state_df = df[df['State'] == state]

    

    rating_data = state_df.groupby(col)[['Complaint ID']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Complaint ID']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot by ' +str(col) + ' in '+ str(state), loc='center')

    

    plt.show() 
show_donut_plot_by_state('Product', 'CA')
show_donut_plot_by_state('Product', 'VA')
show_donut_plot_by_state('Consumer disputed?', 'VA')
show_donut_plot_by_state('Consumer disputed?', 'CA')
df['date_received'] = pd.to_datetime(df['Date received'])

df['month_received'] = df['date_received'].dt.month

df['year_received'] = df['date_received'].dt.year
show_donut_plot('month_received')
show_donut_plot('year_received')
# Show top 10 companies received complaints 



df_dispute = df[df['Consumer disputed?'] == 'Yes']



ax = sns.barplot(

    x = df_dispute['Company'].value_counts().head(10).keys(), 

    y = df_dispute['Company'].value_counts().head(10).values

)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)



plt.title('Consumer Disputed with companies')

plt.show()
def show_donut_plot_by_comany(company, col):

    

    cur_df = df_dispute[df_dispute['Company'] == company]

    

    rating_data = cur_df.groupby(col)[['Complaint ID']].count().head(10)

    plt.figure(figsize = (12, 8))

    plt.pie(rating_data[['Complaint ID']], autopct = '%1.0f%%', startangle = 140, pctdistance = 1.1, shadow = True)



    # create a center circle for more aesthetics to make it better

    gap = plt.Circle((0, 0), 0.5, fc = 'white')

    fig = plt.gcf()

    fig.gca().add_artist(gap)

    

    plt.axis('equal')

    

    cols = []

    for index, row in rating_data.iterrows():

        cols.append(index)

    plt.legend(cols)

    

    plt.title('Donut Plot - ' + str(company) + ' vs  Consumer Dispute', loc='center')

    

    plt.show() 
show_donut_plot_by_comany('Bank of America', 'State')
show_donut_plot_by_comany('Equifax', 'State')
show_donut_plot_by_comany('Citibank', 'State')
import json



# collected from: https://gist.githubusercontent.com/meiqimichelle/7727723/raw/0109432d22f28fd1a669a3fd113e41c4193dbb5d/USstates_avg_latLong

# https://www.factmonster.com/us/postal-information/state-abbreviations-and-state-postal-codes



us_states_json = """{ "states" : [

  {

    "code" : "AK",

    "state":"Alaska",

    "latitude":61.3850,

    "longitude":-152.2683

  },

  {

    "code" : "AL",  

    "state":"Alabama",

    "latitude":32.7990,

    "longitude":-86.8073

  },

  {

    "code" : "AR",

    "state":"Arkansas",

    "latitude":34.9513,

    "longitude":-92.3809

  },

  {

    "code" : "AZ",

    "state":"Arizona",

    "latitude":33.7712,

    "longitude":-111.3877

  },

  {

    "code" : "CA",

    "state":"California",

    "latitude":36.1700,

    "longitude":-119.7462

  },

  {

    "code" : "CO",

    "state":"Colorado",

    "latitude":39.0646,

    "longitude":-105.3272

  },

  {

    "code" : "CT",

    "state":"Connecticut",

    "latitude":41.5834,

    "longitude":-72.7622

  },

  {

    "code" : "DE",

    "state":"Delaware",

    "latitude":39.3498,

    "longitude":-75.5148

  },

  {

    "code" : "FL",

    "state":"Florida",

    "latitude":27.8333,

    "longitude":-81.7170

  },

  {

    "code" : "GA",

    "state":"Georgia",

    "latitude":32.9866,

    "longitude":-83.6487

  },

  {

    "code" : "HI",

    "state":"Hawaii",

    "latitude":21.1098,

    "longitude":-157.5311

  },

  {

    "code" : "IA",

    "state":"Iowa",

    "latitude":42.0046,

    "longitude":-93.2140

  },

  {

    "code" : "ID",

    "state":"Idaho",

    "latitude":44.2394,

    "longitude":-114.5103

  },

  {

    "code" : "IL",

    "state":"Illinois",

    "latitude":40.3363,

    "longitude":-89.0022

  },

  {

    "code" : "IN",

    "state":"Indiana",

    "latitude":39.8647,

    "longitude":-86.2604

  },

  {

    "code" : "KS",

    "state":"Kansas",

    "latitude":38.5111,

    "longitude":-96.8005

  },

  {

    "code" : "KY",

    "state":"Kentucky",

    "latitude":37.6690,

    "longitude":-84.6514

  },

  {

    "code" : "LA",

    "state":"Louisiana",

    "latitude":31.1801,

    "longitude":-91.8749

  },

  {

    "code" : "MA",

    "state":"Massachusetts",

    "latitude":42.2373,

    "longitude":-71.5314

  },

  {

    "code" : "MD",

    "state":"Maryland",

    "latitude":39.0724,

    "longitude":-76.7902

  },

  {

    "code" : "ME",

    "state":"Maine",

    "latitude":44.6074,

    "longitude":-69.3977

  },

  {

    "code" : "MI",

    "state":"Michigan",

    "latitude":43.3504,

    "longitude":-84.5603

  },

  {

    "code" : "MN",

    "state":"Minnesota",

    "latitude":45.7326,

    "longitude":-93.9196

  },

  {

    "code" : "MO",

    "state":"Missouri",

    "latitude":38.4623,

    "longitude":-92.3020

  },

  {

    "code" : "MS",

    "state":"Mississippi",

    "latitude":32.7673,

    "longitude":-89.6812

  },

  {

    "code" : "MO",

    "state":"Montana",

    "latitude":46.9048,

    "longitude":-110.3261

  },

  {

    "code" : "NC",

    "state":"North Carolina",

    "latitude":35.6411,

    "longitude":-79.8431

  },

  {

    "code" : "ND",

    "state":"North Dakota",

    "latitude":47.5362,

    "longitude":-99.7930

  },

  {

    "code" : "NE",

    "state":"Nebraska",

    "latitude":41.1289,

    "longitude":-98.2883

  },

  {

    "code" : "NH",

    "state":"New Hampshire",

    "latitude":43.4108,

    "longitude":-71.5653

  },

  {

    "code" : "NJ",

    "state":"New Jersey",

    "latitude":40.3140,

    "longitude":-74.5089

  },

  {

    "code" : "NM",

    "state":"New Mexico",

    "latitude":34.8375,

    "longitude":-106.2371

  },

  {

    "code" : "NV",

    "state":"Nevada",

    "latitude":38.4199,

    "longitude":-117.1219

  },

  {

    "code" : "NY",

    "state":"New York",

    "latitude":42.1497,

    "longitude":-74.9384

  },

  {

    "code" : "OH",

    "state":"Ohio",

    "latitude":40.3736,

    "longitude":-82.7755

  },

  {

    "code" : "OK",

    "state":"Oklahoma",

    "latitude":35.5376,

    "longitude":-96.9247

  },

  {

    "code" : "OR",

    "state":"Oregon",

    "latitude":44.5672,

    "longitude":-122.1269

  },

  {

    "code" : "PA",

    "state":"Pennsylvania",

    "latitude":40.5773,

    "longitude":-77.2640

  },

  {

    "code" : "RI",

    "state":"Rhode Island",

    "latitude":41.6772,

    "longitude":-71.5101

  },

  {

    "code" : "SC",

    "state":"South Carolina",

    "latitude":33.8191,

    "longitude":-80.9066

  },

  {

    "code" : "SD",

    "state":"South Dakota",

    "latitude":44.2853,

    "longitude":-99.4632

  },

  {

    "code" : "TN",

    "state":"Tennessee",

    "latitude":35.7449,

    "longitude":-86.7489

  },

  {

    "code" : "TX",

    "state":"Texas",

    "latitude":31.1060,

    "longitude":-97.6475

  },

  {

    "code" : "UT",

    "state":"Utah",

    "latitude":40.1135,

    "longitude":-111.8535

  },

  {

    "code" : "VA",

    "state":"Virginia",

    "latitude":37.7680,

    "longitude":-78.2057

  },

  {

    "code" : "VT",

    "state":"Vermont",

    "latitude":44.0407,

    "longitude":-72.7093

  },

  {

    "code" : "WA",

    "state":"Washington",

    "latitude":47.3917,

    "longitude":-121.5708

  },

  {

    "code" : "WI",

    "state":"Wisconsin",

    "latitude":44.2563,

    "longitude":-89.6385

  },

  {

    "code" : "WV",

    "state":"West Virginia",

    "latitude":38.4680,

    "longitude":-80.9696

  },

  {

    "code" : "WY",

    "state":"Wyoming",

    "latitude":42.7475,

    "longitude":-107.2085

  }

]}"""



us_states_json = json.loads(us_states_json)



# print(us_states_json)



state_list = us_states_json['states']



def get_lat(code):

    

    for state_dict in state_list:

        

        if(state_dict['code'] == code):

            return state_dict['latitude']

    

    return None
def get_long(code):

    

    for state_dict in state_list:

        

        if(state_dict['code'] == code):

            return state_dict['longitude']

    

    return None
df['latitude'] = df['State'].apply(get_lat)

df['longitude'] = df['State'].apply(get_long)
# drop NA on lat, long cols to avoid errors in the map making process

df.dropna(subset = ["latitude", "longitude"], inplace=True)
import folium 

from folium import plugins



usa_map = folium.Map([39.358, -98.118], zoom_start=4)

map_title = 'US States Heatmap - Consumer Complaints'

title_html = '''

             <h3 align="center"><b>{}</b></h3>

             '''.format(map_title) 

usa_map.get_root().html.add_child(folium.Element(title_html))

usa_map.add_child(plugins.HeatMap(df[['latitude', 'longitude']]))

usa_map
state_series = df.groupby('State')['Complaint ID'].count()

state_df = pd.DataFrame({'code':state_series.index, 'count':state_series.values})
import plotly.graph_objects as go



fig = go.Figure(data=go.Choropleth(

    locations = state_df['code'], # Spatial coordinates

    z = state_df['count'].astype(float), # Data to be color-coded

    locationmode = 'USA-states', # set of locations match entries in `locations`

    colorscale = 'Greens',

    colorbar_title = "Count Bar",

))



fig.update_layout(

    title_text = 'Consumer Complaints by State',

    geo_scope='usa', # limite map scope to USA

)



fig.show()



# Color scales: Cividis, Greens