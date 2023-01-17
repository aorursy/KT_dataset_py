from beautifultext import BeautifulText as bt # utility script

g1=bt(font_family='Comic Sans MS',color='Dark Black',font_size=19)

g1.printbeautiful('ABSTRACT')
g=bt(font_family='Comic Sans MS',color='#008080')

g.printbeautiful('''Fires are a serious problem in Brazil. As stated under the Dataset description, "Understanding the frequency of forest fires in a time series can help to take action to prevent them". Being able to pin-point where and when that frequency is most observed should give

some clarity on what is the scope we are looking at.With this data, it is possible to assess the evolution of fires over

the years as well as the regions where they were concentrated. The legal Amazon comprises the states of

Acre, Amapá, Pará, Amazonas, Rondonia, Roraima, and part of Mato Grosso, Tocantins, and Maranhão.''')
g1=bt(font_family='Comic Sans MS',color='Dark Black',font_size=19)

g1.printbeautiful('CONTENT')
g=bt(font_family='Comic Sans MS',color='#008080')

g.printbeautiful('''This dataset report of the number of forest fires in Brazil divided by states. 

The series comprises the period of approximately 10 years (1998 to 2017). The data were obtained from the official website of the Brazilian government.''' )
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import seaborn as sns

import plotly.express as px

import geopandas as gpd





import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#using pandas library and 'read_csv' function to read amazon csv file

#as file already formated for us from Kaggle



fire_file=pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv'

                     , encoding='latin1')

fire_file.head(5)


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


fire_file['latitude']=fire_file['state'].map(latitude)

fire_file['longitude']=fire_file['state'].map(longitude)

fire_file

g=bt(font_family='Comic Sans MS',color='Red',font_size=19)

g.printbeautiful('States of Brazil ')
fire_file_gpd=gpd.GeoDataFrame(fire_file,geometry=gpd.points_from_xy(fire_file['longitude'],fire_file['latitude']))

fire_file_gpd.crs={'init':'epsg:4326'}



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin([ 'South America'])]

americas=americas.loc[americas['name']=='Brazil']



ax = americas.plot(figsize=(10,10), color='white', linestyle=':', edgecolor='gray')

fire_file_gpd.plot(ax=ax, markersize=50,color='red')

year_fires=fire_file[fire_file.year==1998] # to see the monthly fires trend for year 1998

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
months_portugese=list(pd.unique(fire_file['month']))

months_english=['january','feburary','march','april','may','june','july','august','september','october','november','december']

dict_month=dict(zip(months_portugese,months_english))

dict_month
g=bt(font_family='Comic Sans MS',color='#008080')

g.printbeautiful('AFTER MAPPING')
fire_file.month=fire_file['month'].map(dict_month)

fire_file
brazil_states_markdown=bt(font_family='Times New Roman',color='blue')

brazil_states_markdown.printbeautiful('STATES OF BRAZIL')
fire_file.isnull().sum()
brazil_eda_markdown=bt(font_family='Time New Roman',font_size=20,color='Purple')

brazil_eda_markdown.printbeautiful('STATES OF BRAZIL')
fire_file.state.unique()
brazil_eda_markdown=bt(font_family='Time New Roman',font_size=20,color='Purple')

brazil_eda_markdown.printbeautiful('DATA VISUALIZATION for BRAZL FOREST FIRES')
comic = bt(font_family='Time New Roman', color='green',font_size=20)

comic.printbeautiful('TOP 5 STATES RECORDING HIGHEST FOREST FIRES FROM 1998 TO 2017')
total_state_fires=fire_file.groupby('state')# gropuing dataframe state wise



states_names=list(fire_file.state.unique())# name of each state present in Dataset



top_5_states_numbers=[]# to hold the numbers for TOP 5 places that caught most fires from 1998 to 2017

top_5_states_names=[]# to hold the names for TOP 5 places that caught most fires from 1998 to 2017





for state in states_names:

    top_5_states_numbers.append(total_state_fires.get_group(state).number.sum())

    # sum of all fires that took place in each state from 1998 to 2017

    top_5_states_names.append(state)

    

    

df_total_fires=pd.DataFrame(data={'States':top_5_states_names,

                                 'Total_Fires':top_5_states_numbers},columns=['States','Total_Fires'])



df_total_fires=df_total_fires.sort_values(['Total_Fires'],ascending=False).iloc[:5]



sns.set_style('darkgrid')

plt.figure(figsize=(15,7))

sns.barplot(df_total_fires.States,df_total_fires.Total_Fires,palette='winter')

plt.xlabel('STATES',fontsize=20)

plt.ylabel('HIGHEST NUMBER OF FIRES',fontsize=15)
mato_grosso_markdown=bt(font_family='Time New Roman',color='#008080',font_size=20)

mato_grosso_markdown.printbeautiful('ANNUAL ANALYSIS FOR FOREST FIRES IN EACH STATE ')
def annual_analysis_for_state(state_name):



    states=fire_file.groupby('state') # gropuing dataframe state wise



    state_name_group=states.get_group(str(state_name))# statename



    state_name_year=state_name_group.groupby('year')# Year by Groups



    years=list(fire_file.year.unique())# list of years from 1998 to 2019



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
year_2009_mato=bt(font_family='Times New Roman',color='green',font_size=19)

year_2009_mato.printbeautiful('MONTHLY ANALYSIS FOR FOREST FIRES IN EACH STATE')
def monthly_fires_for_states(state_name,year_name):

    states=fire_file.groupby('state')

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
g=bt(font_family='Times New Roman',font_size=19,color='blue')

g.printbeautiful('''Hope you have liked my work. If you found this kernel interesting please upvote and if you any suugesstions comment box is for you.

                 Have a GOOD DAY''')