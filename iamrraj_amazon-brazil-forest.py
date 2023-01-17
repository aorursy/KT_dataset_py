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
data = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv" ,encoding='latin1')
data.head()
data.shape
data.info()
data.describe().T
#checking if there are any nulls we are dealing with (missing data)

data.isna().sum()
data.state.unique()
data.month.unique()
import seaborn as sns

import plotly.express as px

import geopandas as gpd





import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



import matplotlib.pyplot as plt
latitude={'Acre':-9.02,'Alagoas':-9.57,'Amapa':02.05,'Amazonas':-5.00,'Bahia':-12.00,'Ceara':-5.00,

          

          'Distrito Federal':-15.45,'Espirito Santo':-20.00,'Goias':-15.55,'Maranhao':-5.00,'Mato Grosso':-14.00

          

          ,'Minas Gerais':-18.50,'Pará':-3.20,'Paraiba':-7.00,'Pernambuco':-8.00,'Piau':-7.00,'Rio':-22.90,

          

          'Rondonia':-11.00,'Roraima':-2.00,'Santa Catarina':-27.25,'Sao Paulo':-23.32,'Sergipe':-10.30,

         

         'Tocantins':-10.00

         }





longitude={

    'Acre':-70.8120,'Alagoas':-36.7820,'Amapa':-50.50,'Amazonas':-65.00,'Bahia':-42.00,'Ceara':-40.00,

    

    'Distrito Federal':-47.45,'Espirito Santo':-40.45,'Goias':-50.10,'Maranhao':-46.00,'Mato Grosso':-55.00,

    

    'Minas Gerais':-46.00,'Pará':-52.00,'Paraiba':-36.00,'Pernambuco':-37.00,'Piau':-73.00, 'Rio':-43.17,

    

    'Rondonia':-63.00,'Roraima':-61.30,'Santa Catarina':-48.30,'Sao Paulo':-46.37,'Sergipe':-37.30,

    

    'Tocantins':-48.00

}



data['latitude']=data['state'].map(latitude)

data['longitude']=data['state'].map(longitude)

data
data.number.sum()
'''Seaborn and Matplotlib Visualization'''

import matplotlib                  # 2D Plotting Library

import matplotlib.pyplot as plt

import seaborn as sns              # Python Data Visualization Library based on matplotlib

import geopandas as gpd            # Python Geospatial Data Library

plt.style.use('fivethirtyeight')

%matplotlib inline



'''Plotly Visualizations'''

import plotly as plotly                # Interactive Graphing Library for Python

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)



'''Spatial Visualizations'''

import folium

import folium.plugins



'''NLP - WordCloud'''

import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



'''Machine Learning'''

import sklearn

from sklearn import preprocessing

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
plt.figure(figsize=(16,7))

sns.distplot(data['number'])
plt.figure(figsize=(16,7))

sns.scatterplot(x='year',y='number',data=data)
plt.figure(figsize=(16,7))

sns.scatterplot(x='month',y='number',data=data)
plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude)

plt.ioff()
fire_file_gpd=gpd.GeoDataFrame(data,geometry=gpd.points_from_xy(data['longitude'],data['latitude']))

fire_file_gpd.crs={'init':'epsg:4326'}



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin([ 'South America'])]

americas=americas.loc[americas['name']=='Brazil']



ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

fire_file_gpd.plot(ax=ax, markersize=50,color='red')

import folium

from folium.plugins import HeatMap

m=folium.Map([-14.23,-51.92],zoom_start=3)

HeatMap(data[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
import folium

from folium.plugins import MarkerCluster

from folium import plugins

print("State With More Burn")

Long = 51.9253

Lat = 14.2350

mapdf1 = folium.Map([Lat, Long], zoom_start=13)

mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon, label in zip(data.latitude,data.longitude,data.state):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1
#we are already given the year column, however for good practice we can also extract it from the date one

data['Year']=pd.DatetimeIndex(data['date']).year

#cheking unique years in new created column 

data.Year.unique()
#we are not going to be using old year column and date column as they serve no significant purpose anymore 

#amazon_df.drop(columns=['date', 'year'], axis=1, inplace=True)

#changing order of columns for preffered format

#amazon_df=amazon_df[['state','number','month','Year']]

#changing names of columns for preffered format

#amazon_df.rename(columns={'state': 'State', 'number': 'Fire_Number', 'month': 'Month'}, inplace=True)

#checking changes made

#amazon_df.head()

plt.figure(figsize = (8,8))

sns.heatmap(data.corr(),annot = True,linewidths = 0.5,cmap='cubehelix_r');

plt.savefig('Correlation Heatmap.png')

year_fires=data[data.year==1998] # to see the monthly fires trend for year 1998

year_fires
# Function for displaying the map

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')



# Create a base map

m_4 = folium.Map(location=[-14.23,-51.92], tiles='cartodbpositron', zoom_start=4)





def color_producer(val):

    if val =='january':

        return 'darkred'

    

    elif val=='feburary':

        return 'blue'

    

    elif val=='march':

        return 'darkgreen'

    

    elif val=='april':

        return 'green'

    

    elif val=='may':

        return 'yellow'

    

    elif val=='june':

        return 'orange'

    

    elif val=='july':

        return 'red'

    

    elif val=='september':

        return 'darkpurple'

    

    elif val=='october':

        return 'black'

    

    elif val=='november':

        return 'lightred'

    elif val=='december':

        return 'lightgreen'

    

    
# Add a bubble map to the base map

for i,row in year_fires.iterrows():

    Circle(

        location=[row['latitude'], row['longitude']],

        radius=20,

        color=color_producer(row['month'])).add_to(m_4)



# Display the map

embed_map(m_4, 'm_4.html')
months_portugese=list(pd.unique(data['month']))

months_english=['january','feburary','march','april','may','june','july','august','september','october','november','december']

dict_month=dict(zip(months_portugese,months_english))

dict_month
print(data[(data["state"] == "Rio") & (data["year"] == 1998)])

print(data[(data["state"] == "Amazonas") & (data["year"] == 1998)])
data['state'].value_counts()
data["month"].value_counts()
data[(data["year"] == 2017) & (data["state"] == 'Acre')]
amazonia_legal = ['Amazonas', 'Pará', 'Roraima', 'Amapa', 'Rondonia', 'Acre', 'Tocantins', 'Mato Grosso', 'Maranhao']



estados = data.groupby('state')['number'].sum().sort_values(ascending=False)



ax = plt.gca()

colors = ['C0' if i not in amazonia_legal else 'r' for i in estados.index]

estados.plot(kind='bar',ax=ax,color=colors, figsize=(10, 5))

h,l = ax.get_legend_handles_labels()

ax.set_title("States with the largest fires")

ax.set_ylabel('Number of fires')

ax.set_xlabel('States')

ax.legend(["Amazonia Legal", "Other states"], labelspacing=2)
amazonia_legal = ['Amazonas', 'Pará', 'Roraima', 'Amapa', 'Rondonia', 'Acre', 'Tocantins', 'Mato Grosso', 'Maranhao']



estados = data.groupby('month')['number'].sum().sort_values(ascending=False)



ax = plt.gca()

colors = ['C0' if i not in amazonia_legal else 'r' for i in estados.index]

estados.plot(kind='bar',ax=ax,color=colors, figsize=(10, 5))

h,l = ax.get_legend_handles_labels()

ax.set_title("States with the largest fires")

ax.set_ylabel('Number of fires')

ax.set_xlabel('States')

ax.legend(["Amazonia Legal", "Other states"], labelspacing=2)
meses_incendio = data.groupby('month')['number'].sum()

meses_incendio_amazonia = data[data.state.isin(amazonia_legal)].groupby('month')['number'].sum()



ax = plt.gca()

meses_incendio.plot(kind='bar',x='month',y='number', ax=ax, stacked=True, figsize=(10, 5))

meses_incendio_amazonia.plot(kind='bar',x='month',y='number', ax=ax, stacked=True, color='r', figsize=(10, 5))

ax.set_title("Total fires and fires in Amazonia")

ax.set_ylabel('Number of fires')

ax.set_xlabel('Month')

ax.legend(["Total fires", "Fires in Amazon"])
#Total Number  on each state in descending order

data.groupby('state').sum().sort_values(by='number',ascending=False)[['number']].plot(kind='bar',figsize=(16,8),title='Total Number By State')
#Total Number on each state in descending order

data.groupby('month').sum().sort_values(by='number',ascending=False)[['number']].plot(kind='bar',figsize=(16,8),title='Total Number By State')
def annual_analysis_for_state(state_name):



    states=data.groupby('state') # gropuing dataframe state wise



    state_name_group=states.get_group(str(state_name))# statename



    state_name_year=state_name_group.groupby('year')# Year by Groups



    years=list(data.year.unique())# list of years from 1998 to 2019



    total_annual_fires=[]# list to calculate numnber of forest fires from 1998 to 2019 





    for year in years:

        total_annual_fires.append(state_name_year.get_group(year).number.sum())

    years_df=pd.DataFrame(data={'Years':years,

                                'Total_Fires':total_annual_fires})



    plt.figure(figsize=(20,10))





    fig = px.bar(years_df, x='Years', y='Total_Fires',color='Total_Fires')



    fig.update_layout(

        title="TRENDS OF FOREST FIRES IN "+str(state_name.upper()),

        xaxis_title="YEARS",

        yaxis_title="TOTAL NUMBER OF FIRES",

        font=dict(

            family="Courier New",

            size=18,

            color="black"

        )

    )

    fig.show()

annual_analysis_for_state('Rio')#put the name of state here
months_portugese=list(pd.unique(data['month']))

months_english=['january','feburary','march','april','may','june','july','august','september','october','november','december']

dict_month=dict(zip(months_portugese,months_english))

dict_month
data.month=data['month'].map(dict_month)

data
def monthly_fires_for_states(state_name,year_name):

    states=data.groupby('state')

    state_name_group=states.get_group(str(state_name))

    state_name_year=state_name_group.groupby('year')

    year_X=state_name_year.get_group(year_name)

    month_X=year_X.groupby('month')

    months=['january','feburary','march','april','may','june','july','august','september','october','november','december']



    monthly_fires=[]

    for month in months:

        monthly_fires.append(month_X.get_group(month).number.sum())





    annual_df=pd.DataFrame(data={

        'Months':months,

        'Monthly_fires':monthly_fires

    })

    plt.figure(figsize=(20,8))



    fig = px.bar(annual_df, x='Months', y='Monthly_fires',color='Monthly_fires')



    title="MONTHLY TRENDS OF FOREST FIRES IN "+str(state_name.upper()+" FOR YEAR "+str(year_name))

    fig.update_layout(

        title=title,

        xaxis_title="MONTHS",

        yaxis_title="TOTAL NUMBER OF FIRES",

        font=dict(

            family="Courier New",

            size=18,

            color="black"

        )

    )

    fig.show()

# monthly analysis for STATE OF RIO in year 2010

monthly_fires_for_states('Rio',2010)# put the name of state and year