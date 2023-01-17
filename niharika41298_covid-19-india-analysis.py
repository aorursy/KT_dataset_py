import plotly.graph_objects as go

import plotly.offline as py

autosize =False





# Use `hole` to create a donut-like pie chart

values=[431000, 1520000, 294000]

labels=['Confirmed',"Recovered","Deaths"]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,

                  marker=dict(colors=['#00008b','#fffdd0'], line=dict(color='#FFFFFF', width=2.5)))

fig.update_layout(

    title='COVID-19 ACTIVE CASES VS CURED WORLDWIDE')

py.iplot(fig)
# Use `hole` to create a donut-like pie chart

values=[74281, 24386, 2415]

labels=['Confirmed',"Recovered","Deaths"]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,

                  marker=dict(colors=['#DAA520','#800000'], line=dict(color='#FFFFFF', width=2.5)))

fig.update_layout(

    title='COVID-19 ACTIVE CASES VS CURED INDIA')

austosize=False

py.iplot(fig)
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

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import geopandas

import seaborn as sns
data=pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")



data.tail()
import json

import folium



statecases=data.groupby('Name of State / UT')['Total Confirmed cases','Death','Cured/Discharged/Migrated'].max().reset_index()



with open('/kaggle/input/indian-state-geojson-data/india_state_geo.json') as file:

    geojsonData = json.load(file)



for i in geojsonData['features']:

    if(i['properties']['NAME_1']=='Orissa'):

        i['properties']['NAME_1']='Odisha'

    elif(i['properties']['NAME_1']=='Uttaranchal'):

        i['properties']['NAME_1']='Uttarakhand'

        

for i in geojsonData['features']:

    i['id'] = i['properties']['NAME_1']

    



map_choropleth = folium.Map(location = [20.5937,78.9629], zoom_start = 4)



folium.Choropleth(geo_data=geojsonData,

                 data=statecases,

                 name='CHOROPLETH',

                 key_on='feature.id',

                 columns = ['Name of State / UT','Total Confirmed cases'],

                 fill_color='YlOrRd',

                 fill_opacity=0.7,

                 line_opacity=0.8,

                 legend_name='Confirmed Cases',

                 highlight=True).add_to(map_choropleth)



folium.LayerControl().add_to(map_choropleth)

display(map_choropleth)
zones=pd.read_csv('/kaggle/input/covid19indiazones/India-District-Zones.csv')
import plotly.express as px

fig = px.treemap(zones, path=['State','District'],

                  color='Zone', hover_data=['Zone'], color_discrete_map={'Red Zone':'red', 'Green Zone':'green', 'Orange Zone':'orange'})

autosize=False

py.iplot(fig)
data.shape
# Create a plot

plt.figure(figsize=(8,12))



# Add title

plt.title("Total cases by state")



grouped_data=data.groupby("Name of State / UT").sum()



sns.barplot(x=grouped_data['Total Confirmed cases'], y=grouped_data.index)



data['Total Confirmed cases'].sum()
grouped_by_date_data=data.groupby("Date").sum()



plt.figure(figsize=(17,16))



plt.xticks(rotation=90)

sns.lineplot(data=grouped_by_date_data["Total Confirmed cases"],label="Total Confirmed cases")
symptoms={'symptom':['Fever',

        'Dry cough',

        'Fatigue',

        'Sputum production',

        'Shortness of breath',

        'Muscle pain',

        'Sore throat',

        'Headache',

        'Chills',

        'Nausea or vomiting',

        'Nasal congestion',

        'Diarrhoea',

        'Haemoptysis',

        'Conjunctival congestion'],

        'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8],

          'parent':['high','high','high','high','medium','medium','medium','medium','medium','medium','low','low','low',"low"]}



symptoms=pd.DataFrame(symptoms)
fig =px.sunburst(

    symptoms,

    path=['symptom','parent'],

    values='percentage',

    color='percentage')

autosize=False

py.iplot(fig)
plt.figure(figsize=(20,20))



# Add title

plt.title("Active cases")



#Total Active cases

data["Active cases"]=data["Total Confirmed cases"]-data["Cured/Discharged/Migrated"]-data["Death"]



grouped_active_data=data.groupby("Name of State / UT").sum()

grouped_active_data=(grouped_active_data.sort_values(by="Active cases"))

plt.xticks(rotation=90)

sns.barplot(x=grouped_active_data['Active cases'], y=grouped_active_data.index,)
plt.figure(figsize=(12,6))

plt.plot(grouped_by_date_data["Total Confirmed cases"].diff().fillna(0),linewidth=3,label="Confirmed Cases")

plt.plot(grouped_by_date_data["Cured/Discharged/Migrated"].diff().fillna(0),linewidth=3,label="Recovered Cases")

plt.plot(grouped_by_date_data["Death"].diff().fillna(0),linewidth=3,label="Death Cases")

plt.ylabel("Increase in Number of Cases")

plt.xlabel("Date")

plt.title("Daily increase in different types of cases in India")

plt.xticks(rotation=90)

plt.legend()
def color_negative_red(val):

    """

    Takes a scalar and returns a string with

    the css property `'color: red'` for negative

    strings, black otherwise.

    """

    color = 'red' if val > 30 else 'green'

    return 'color: %s' % color
total_cases=grouped_data["Total Confirmed cases"].sum()

rec_cases=grouped_data["Cured/Discharged/Migrated"].sum()

death=grouped_data["Death"].sum()

y_axis=[total_cases,rec_cases,death]

x_axis=["Total Confirmed cases","Recovered","Deaths"]
plt.figure(figsize=(8,5))

sns.barplot(x=x_axis, y=y_axis)

plt.title('Status of affected people.')

plt.xlabel('Status', fontsize=15)

plt.ylabel('Count of people affected', fontsize=15)

plt.show()

testing=pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")

testing.tail()
sttest=testing.groupby("State").sum()
plt.figure(figsize=(20,20))

plt.barh(sttest.index,sttest['TotalSamples'],label="Total Samples",color='black')

plt.barh(sttest.index, sttest['Positive'],label="Positive Cases",color='coral')

plt.xlabel('Cases',size=30)

plt.ylabel("States",size=30)

plt.legend(frameon=True, fontsize=12)

plt.title('Recoveries and Total Number of Cases Statewise',fontsize = 20)

plt.show()
age=pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")
age
plt.figure(figsize=(10,10))

#plt.title("Current age group scenario in india",fontsize=50)

labels=age['AgeGroup']

len(labels)

sizes=['3.18','3.9','24.86','21.10','16.18','11.13','12.86','4.05','1.45','1.3']

plt.pie(sizes,labels=labels,autopct='%1.1f%%')

plt.show() 
hospbeds=pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")

hospbeds.head()
hospbeds= hospbeds.fillna(0)

hospbeds.head()
centers=['NumPrimaryHealthCenters_HMIS','NumCommunityHealthCenters_HMIS','NumSubDistrictHospitals_HMIS','NumDistrictHospitals_HMIS','TotalPublicHealthFacilities_HMIS','NumPublicBeds_HMIS','NumRuralHospitals_NHP18','NumRuralBeds_NHP18','NumUrbanHospitals_NHP18','NumUrbanHospitals_NHP18']
hospbeds['NumPrimaryHealthCenters_HMIS'] = hospbeds['NumPrimaryHealthCenters_HMIS'].str.replace(',', '')

hospbeds['NumPrimaryHealthCenters_HMIS']=hospbeds['NumPrimaryHealthCenters_HMIS'].astype(str).astype(int)
hospbeds.dtypes
plt.figure(figsize=(20,60))

for i,col in enumerate(centers):

    plt.subplot(8,2,i+1)

    sns.barplot(data=hospbeds,y='State/UT',x=col)

    plt.xlabel('Number of Cases')

    plt.ylabel('')

    plt.title(col)

plt.tight_layout()

plt.show()
icmrtestlabs= pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingLabs.csv")

icmrtestlabs


import plotly.express as px

fig = px.treemap(icmrtestlabs, path=['state','city'],

                  color='city', hover_data=['lab','address'],color_continuous_scale='Purples')

autosize=False



py.iplot(fig)