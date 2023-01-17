# Importing library for posting youtube videos in frame

from IPython.display import IFrame, YouTubeVideo

YouTubeVideo('aerq4byr7ps',width=600, height=400)
"""Importing required libraries"""

# Import libraries

import numpy as np

import pandas as pd

from datetime import date

import pandas_profiling as pp





'''Customize visualization

Seaborn and matplotlib visualization.'''

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline

import folium 

from folium import plugins





'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook





'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print("Path of Coronavirus Dataset : ")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
"""Loading and reading the dataset"""

# Load coronavirus dataset

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")



# View dataset

data.head()
"""Statistics and info of dataset"""

#  info of dataset

data.info(verbose=True, null_counts = False)
# Describe dataset

data.describe()
# Other way to describe the data

data.describe(include="O")
"""Profiling of entire data using ProfileReport"""

# An alternative to describe/info

profile = pp.ProfileReport(data)

profile
# The report can also be exported into an interactive HTML file with the following code.

profile.to_file("hosuing_data_profiling.html")



# Click on Toggle details to get more details
"""Data Cleaning"""

# From data.info(), we can see that date column is of type object

# Convert Date and Last_Update column to datatime format

data['Date'] = data['Date'].apply(pd.to_datetime)

data['Last_Update'] = data['Last_Update'].apply(pd.to_datetime)

data.head()
# Let's furthur divide Last_Update date into day and hour

data['Day'] = data['Last_Update'].apply(lambda x:x.day)

data['Hour'] = data['Last_Update'].apply(lambda x:x.hour)

# View data

data.head()
"""Let's look of the feature info"""

# Again checking the dataset info

data.info(verbose=True, null_counts = False)



# Now dates columns are updated to datatime64 datatype
# List of affected countries 

countries = data['Country'].unique().tolist()

print(countries)

# Note that China and Mainland China have been reported separately.



# Total countries affected by virus

print("\nTotal countries affected by virus: ",len(countries))
# Combining China and Mainland China cases

data['Country'].replace({'Mainland China':'China'},inplace=True)

countries = data['Country'].unique().tolist()

print(countries)

print("\nTotal countries affected by virus: ",len(countries))



# Note now we have combined China and Mainland China with China.
# Doing manupulation on Date column to get date

bold("**Present Global Scenario for Latest Data**")

d = data['Date'][-1:].astype('str')

year = int(d.values[0].split('-')[0])

month = int(d.values[0].split('-')[1])

day = int(d.values[0].split('-')[2].split()[0])



data_latest = data[data['Date'] > pd.Timestamp(date(year,month,day))]

data_latest.head()
# Some more insights

bold('**Present Gobal condition: Confirmed, Death and Recovered**')

print('Globally Confirmed Cases: ',data_latest['Confirmed'].sum())

print('Global Deaths Cases: ',data_latest['Deaths'].sum())

print('Globally Recovered Cases: ',data_latest['Recovered'].sum())
# Let's look the various Provinces/States affected

data_latest.groupby(['Country','Province/State']).sum()
# Country wise confirmed, death and recovered cases.

bold("** Country wise confirmed, death and recovered cases, 4th february 2020**")

temp_data = data_latest.groupby('Country')['Confirmed','Deaths','Recovered'].sum().reset_index()



cm = sns.light_palette("green", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [

  ('font-size', '12px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



## Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '12px'),

  ('color', 'black')

   ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(temp_data.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_caption('*China Have most confirmed, deaths & recovered cases.')

  .set_table_styles(styles))
# Creating a dataframe with total no of confirmed cases for every country

number_of_countries = len(data_latest['Country'].value_counts())



cases = pd.DataFrame(data_latest.groupby('Country')['Confirmed'].sum())

cases['Country'] = cases.index

cases.index=np.arange(1,number_of_countries+1)



global_cases = cases[['Country','Confirmed']]

#global_cases.sort_values(by=['Confirmed'],ascending=False)

global_cases
# Provinces where deaths have taken place

bold("**Provinces where deaths have taken place**")

data_latest.groupby('Country')['Deaths'].sum().sort_values(ascending=False)[:5]
# Importing the world_coordinates dataset

world_coordinates = pd.read_csv('../input/world-coordinates/world_coordinates.csv')



# Merging the coordinates dataframe with original dataframe

world_data = pd.merge(world_coordinates,global_cases,on='Country')

world_data.head()
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data['latitude'], world_data['longitude'], world_data['Confirmed'], world_data['Country']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='red',

                        fill_opacity=0.7 ).add_to(world_map)

world_map
china_data_latest = data_latest[data_latest['Country']=='China'][["Province/State","Confirmed","Deaths","Recovered"]]



bold("**Present Scenario of China Condition, February 2020**")



cm = sns.light_palette("green", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [

  ('font-size', '11px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



## Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '11px'),

  ('color', 'black')

   ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(china_data_latest.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_table_styles(styles))
# Top 10 Infected State in China

bars = alt.Chart(china_data_latest.head(10)).mark_bar(color='orange',opacity=0.7).encode(

    x='Confirmed:Q',

    y=alt.Y('Province/State:O', sort='-x')

).properties(

    title={

    "text":['Top 10 Infected State in China'],

    "subtitle":['*Hubei have most confirmed cases'],

    "fontSize":15,

    "fontWeight": 'bold',

    "font":'Courier New',

    }

)



text = bars.mark_text(

    align='left',

    baseline='middle',

    dx=3  # Nudges text to right so it doesn't appear on top of the bar

).encode(

    text='Confirmed:Q'    

)



(bars + text).properties( height=300, width=600)
# Top 10 States With Recovered Cases in China

temp_data = china_data_latest[china_data_latest['Recovered']> 0]

bars = alt.Chart(temp_data.head(10)).mark_bar(color='green',opacity=0.7).encode(

    x='Recovered:Q',

    y=alt.Y('Province/State:O', sort='-x')

).properties(

    title={

    "text":['Top 10 States With Recovered Cases in China'],

    "subtitle":['*Hubei, Guangdong, Zhejiang have most recovered cases'],

    "fontSize":15,

    "fontWeight": 'bold',

    "font":'Courier New',

    }

)



text = bars.mark_text(

    align='left',

    baseline='middle',

    dx=3  # Nudges text to right so it doesn't appear on top of the bar

).encode(

    text='Recovered:Q'    

)



(bars + text).properties( height=300, width=600)
# States With Deaths Case in China

temp = china_data_latest[china_data_latest['Deaths'] > 0]

bars = alt.Chart(temp.head(10)).mark_bar(color='red',opacity=0.7).encode(

    x='Deaths:Q',

    y=alt.Y('Province/State:O', sort='-x')

).properties(

    title={

    "text":['States With Deaths Case in China'],

    "subtitle":['*Hubei, Henan,Heilongjiang have most deaths cases'],

    "fontSize":15,

    "fontWeight": 'bold',

    "font":'Courier New',

    }

)



text = bars.mark_text(

    align='left',

    baseline='middle',

    dx=3  # Nudges text to right so it doesn't appear on top of the bar

).encode(

    text='Deaths:Q'    

)



(bars + text).properties( height=300, width=600)
# Confirmed vs Recovered vs Death figures of Provinces of China other than Hubei

# bold("**Confirmed vs Recovered vs Death figures of Provinces of China other than Hubei**")

f, ax = plt.subplots(figsize=(15, 10))





sns.barplot(x="Confirmed", y="Province/State", data=china_data_latest[1:],

            label="Confirmed", color="orange",alpha=0.7)





sns.barplot(x="Recovered", y="Province/State", data=china_data_latest[1:],

            label="Recovered", color="g",alpha=0.7)





sns.barplot(x="Deaths", y="Province/State", data=china_data_latest[1:],

            label="Deaths", color="r",alpha=0.7)



# Add a legend and informative axis label

ax.set_title('Confirmed vs Recovered vs Death figures of Provinces of China other than Hubei', fontsize=20, fontweight='bold', position=(0.53, 1.05))

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 24), ylabel="",

       xlabel="Stats")

sns.despine(left=True, bottom=True)

india_data_latest = data_latest[data_latest['Country']=='India'][["Province/State","Confirmed","Deaths","Recovered"]]



bold("**Present Scenario of India Condition, February 2020**")



cm = sns.light_palette("green", as_cmap=True)



# Set CSS properties for th elements in dataframe

th_props = [

  ('font-size', '11px'),

  ('text-align', 'center'),

  ('font-weight', 'bold'),

  ('color', '#6d6d6d'),

  ('background-color', '#f7f7f9')

  ]



## Set CSS properties for td elements in dataframe

td_props = [

  ('font-size', '11px'),

  ('color', 'black')

   ]



# Set table styles

styles = [

  dict(selector="th", props=th_props),

  dict(selector="td", props=td_props)

  ]



(india_data_latest.style

  .background_gradient(cmap=cm, subset=["Confirmed","Deaths","Recovered"])

  .highlight_max(subset=["Confirmed","Deaths","Recovered"])

  .set_table_styles(styles))