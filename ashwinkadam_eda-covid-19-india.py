# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import folium

import plotly_express as exp

import plotly.graph_objects as go
data_india = pd.read_csv('../input/covid19-in-india/covid_19_india.csv',parse_dates=['Date'],dayfirst=True)

#data_india.head()
data_india.info()
data_india=data_india.drop(axis=1,columns=['Sno','Time'])
data_india=data_india.rename(columns={'Confirmed':'Active'})
#data_india.head()
data_india=data_india.replace(to_replace='-', value=int(0))
data_co = pd.read_csv('../input/coronavirus-cases-in-india/Indian Coordinates.csv')

data_co.drop(axis=1,columns='Unnamed: 3',inplace=True)

data_co = data_co.rename(columns={'Name of State / UT': 'State/UnionTerritory'})
data_india["State/UnionTerritory"].replace({'Andaman and Nicobar Islands':'Andaman And Nicobar '

                                            , 'Andhra Pradesh':'Andhra Pradesh'

                                            , 'Bihar':'Bihar '

                                            ,'Chandigarh':'Chandigarh'

                                            , 'Chattisgarh':'Chhattisgarh '

                                            , 'Chhattisgarh':'Chhattisgarh '

                                            , 'Delhi': 'Delhi'

                                            , 'Goa':'Goa '

                                            ,'Gujarat':'Gujarat'

                                            , 'Haryana':'Haryana'

                                            , 'Himachal Pradesh':'Himachal Pradesh '

                                            , 'Jammu and Kashmir':'Jammu and Kashmir'

                                            ,'Karnataka':'Karnataka'

                                            , 'Kerala':'Kerala'

                                            , 'Ladakh': 'Ladakh'

                                            , 'Madhya Pradesh':'Madhya Pradesh '

                                            , 'Maharashtra':'Maharashtra'

                                            ,'Manipur':'Manipur '

                                            , 'Mizoram':'Mizoram '

                                            , 'Odisha':'Odisha'

                                            , 'Pondicherry':'Pondicherry'

                                            , 'Puducherry':'Pondicherry'

                                            ,'Punjab':'Punjab'

                                            , 'Rajasthan':'Rajasthan'

                                            , 'Tamil Nadu':'Tamil Nadu'

                                            , 'Telengana':'Telengana'

                                            , 'Uttar Pradesh':'Uttar Pradesh'

                                            ,'Uttarakhand':'Uttarakhand'

                                            , 'West Bengal':'West Bengal '}, inplace=True)
data_india = pd.merge(data_india,data_co, how='left', on='State/UnionTerritory')
#data_india
data_india.isnull().sum()
#Rearranging columns

data_india=data_india[['Date','State/UnionTerritory','Latitude','Longitude','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths','Active']]
date_sorted=data_india.groupby('Date')['State/UnionTerritory','ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths','Active'].sum().reset_index()
#date_sorted
latest_date_data = data_india[data_india['Date']==max(data_india['Date'])]
#latest_date_data
indian_map = folium.Map(location=[20.5937,78.9629], zoom_start=4.5)



for lat, lon, cur,ded,act,state in zip(latest_date_data['Latitude'], latest_date_data['Longitude'], latest_date_data['Cured'],

                                 latest_date_data['Deaths'], latest_date_data['Active'],latest_date_data['State/UnionTerritory'] ):

    folium.Marker(

        [lat, lon],

        popup = ('State/UnionTerritory: ' + str(state) + '<br>' 

                 'Cured: '  + str(cur) + '<br>'

                 'Deaths: ' + str(ded) + '<br>'

                 'Active: ' + str(act) 

                ),

        tooltip='Click here'

        ).add_to(indian_map)

    

indian_map
fig = exp.pie(latest_date_data, values='Active', names='State/UnionTerritory',title='Active Case Distribution')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=date_sorted.Date,

                y=date_sorted['Cured'],

                name="Cured",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=date_sorted.Date,

                y=date_sorted['Deaths'],

                name="Deaths",

                line_color='dimgray',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=date_sorted.Date,

                y=date_sorted['Active'],

                name="Active",

                opacity=0.8))



fig.update_layout(title_text="Cured, Death and Active Cases Over Time in India")

fig.show()
data_Maharashtra=data_india[data_india['State/UnionTerritory']=='Maharashtra'][['Date','Cured','Deaths','Active']]
#data_Maharashtra
maha_fig = go.Figure()

maha_fig.add_trace(go.Scatter(

                x=data_Maharashtra.Date,

                y=data_Maharashtra['Cured'],

                name="Cured",

                line_color='deepskyblue',

                opacity=0.8))



maha_fig.add_trace(go.Scatter(

                x=data_Maharashtra.Date,

                y=data_Maharashtra['Deaths'],

                name="Deaths",

                line_color='dimgray',

                opacity=0.8))



maha_fig.add_trace(go.Scatter(

                x=data_Maharashtra.Date,

                y=data_Maharashtra['Active'],

                name="Active",

                opacity=0.8))



maha_fig.update_layout(title_text="Cured, Death and Active  Cases Over Time in Maharashtra")

maha_fig.show()
data_age = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')

data_age = data_age[:9]

data_age
fig = exp.bar(data_age, x='AgeGroup', y='TotalCases',

             hover_data=['Percentage'], color='TotalCases',title='AgeGroup Vs TotalCases')

fig.show()
data_individual = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')

#data_individual
data_individual['gender'].value_counts()