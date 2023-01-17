#modules

import numpy as np

import pandas as pd

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.patches import PathPatch

import seaborn as sns

import geopandas as gpd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go
amazon_df=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

amazon_df.head(5)
month_map={'Janeiro': 'January', 'Fevereiro': 'February', 'Março': 'March', 'Abril': 'April', 'Maio': 'May',

          'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October',

          'Novembro': 'November', 'Dezembro': 'December'}

amazon_df['month']=amazon_df['month'].map(month_map)

#checking the month column after the changes were made

amazon_df.month.unique()
amazon_df['Year']=pd.DatetimeIndex(amazon_df['date']).year

amazon_df.Year.unique()
amazon_df.drop(columns=['date', 'year'], axis=1, inplace=True)

amazon_df=amazon_df[['state','number','month','Year']]

amazon_df.rename(columns={'state': 'State', 'number': 'Fire_Number', 'month': 'Month'}, inplace=True)

amazon_df.head()
#Number of fires per year in Brazil from 1998 to 2017

plt.figure(figsize=(12,4))

sns.barplot(x='Year',y='Fire_Number',data=amazon_df)

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.title('Fires by Year', fontsize = 18)

plt.ylabel('Number of fires', fontsize=14)

plt.xlabel('Year', fontsize=14)
plt.figure(figsize=(12,4))

sns.barplot(x=amazon_df.State,y=amazon_df.Fire_Number,data=amazon_df)

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.title('States wise Fires' , fontsize=15)

plt.ylabel('Number of fires', fontsize=14)

plt.xlabel('States', fontsize=14)
import plotly

import plotly.offline as py

plotly.offline.init_notebook_mode()



pie1 = amazon_df.Fire_Number

labels = amazon_df.Year

# figure

fig = {

  "data": [

    {

      "values": pie1,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Forest Fires",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Forest Fires of Brazil Rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Forest Fires",

                "x": 2,

                "y": 2

            },

        ]

    }

}

py.iplot(fig)
#creating a list of years

years=list(amazon_df.Year.unique())

#creating an empty list, which will be populated later with amount of fires reported

sub_fires_per_year=[]

#using for loop to extract sum of fires reported for each year and append list above

for i in years:

    y=amazon_df.loc[amazon_df['Year']==i].Fire_Number.sum().round(0)

    sub_fires_per_year.append(y)

#creating a dictionary with results     

fire_year_dic={'Year':years,'Total_Fires':sub_fires_per_year}

#creating a new sub dataframe for later plot 

time_plot_1_df=pd.DataFrame(fire_year_dic)
time_plot_1=go.Figure(go.Scatter(x=time_plot_1_df.Year, y=time_plot_1_df.Total_Fires,

                                 mode='lines+markers', line={'color': 'red'}))

time_plot_1.update_layout(title='Brazil Fires per 1998-2017 Years',

                   xaxis_title='Year',

                   yaxis_title='Fires')



time_plot_1.show()
data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv',encoding="ISO-8859-1")

data.head()
fig = go.Figure()

for i in data['state'].unique():

    datas = data[data['state']==i][['date','state','number']].groupby(['date','state']).mean().reset_index()

    fig.add_trace(go.Scatter(x=datas['date'], y=datas['number'], name=i,

                        line_shape='linear'))

fig.show()

states=list(amazon_df.State.unique())

#creating empty list for each state that will be later appended

acre_list=[]

alagoas_list=[] 

amapa_list=[] 

amazonas_list=[] 

bahia_list=[] 

ceara_list=[]

distrito_list=[] 

espirito_list=[] 

goias_list=[] 

maranhao_list=[] 

mato_list=[] 

minas_list=[]

para_list=[] 

paraiba_list=[] 

perna_list=[]

piau_list=[]

rio_list=[]

rondonia_list=[]

roraima_list=[]

santa_list=[]

sao_list=[]

sergipe_list=[]

tocantins_list=[]



for x in states:

    st=x

    for i in years:

        ye=i

        if st=='Acre':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            acre_list.append(y)

        elif st=='Alagoas':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            alagoas_list.append(y)

        elif st=='Amazonas':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            amazonas_list.append(y)

        elif st=='Amapa':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            amapa_list.append(y)

        elif st=='Bahia':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            bahia_list.append(y)

        elif st=='Ceara':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            ceara_list.append(y)

        elif st=='Distrito Federal':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            distrito_list.append(y)

        elif st=='Espirito Santo':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            espirito_list.append(y)

        elif st=='Goias':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            goias_list.append(y)

        elif st=='Maranhao':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            maranhao_list.append(y)

        elif st=='Mato Grosso':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            mato_list.append(y)

        elif st=='Minas Gerais':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            minas_list.append(y)

        elif st=='Pará':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            para_list.append(y)

        elif st=='Paraiba':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            paraiba_list.append(y)

        elif st=='Pernambuco':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            perna_list.append(y)

        elif st=='Piau':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            piau_list.append(y)

        elif st=='Rio':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            rio_list.append(y)

        elif st=='Rondonia':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            rondonia_list.append(y)

        elif st=='Roraima':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            roraima_list.append(y)

        elif st=='Santa Catarina':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            santa_list.append(y)

        elif st=='Sao Paulo':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            sao_list.append(y)

        elif st=='Sergipe':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            sergipe_list.append(y)

        elif st=='Tocantins':

            y=amazon_df.loc[(amazon_df['State']== st) & (amazon_df['Year']== ye)].Fire_Number.sum().round(0)

            tocantins_list.append(y)



time_plot_2_df=pd.DataFrame(list(zip(years, acre_list, alagoas_list, amapa_list, amazonas_list,

                                     bahia_list, ceara_list, distrito_list, espirito_list,

                                     goias_list, maranhao_list, mato_list, minas_list, para_list,

                                     paraiba_list, perna_list, piau_list, rio_list, rondonia_list,

                                     roraima_list, santa_list, sao_list, sergipe_list, tocantins_list)),

                            columns =['Year', 'Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara',

                                      'Distrito Federal', 'Espirito Santo', 'Goias', 'Maranhao',

                                      'Mato Grosso', 'Minas Gerais', 'Pará', 'Paraiba', 'Pernambuco',

                                      'Piau', 'Rio', 'Rondonia', 'Roraima', 'Santa Catarina',

                                      'Sao Paulo', 'Sergipe', 'Tocantins'])



#creating subdataframe for visualizing this states geographically

geo_plot_df=pd.DataFrame(time_plot_2_df.sum().nlargest(11))

#formatting new dataframe

geo_plot_df.rename(columns={0:'Count'}, inplace=True)

geo_plot_df.reset_index(inplace=True)

geo_plot_df.rename(columns={'index':'State'}, inplace=True)

geo_plot_df.drop(geo_plot_df.index[5], inplace=True)



lat=[-16.350000, -22.15847, -23.533773, -22.908333, -11.409874, -21.5089, -16.328547,

     -19.841644, -21.175, -3.416843]

long=[-56.666668, -43.29321, -46.625290, -43.196388, -41.280857, -43.3228, -48.953403,

     -43.986511, -43.01778, -65.856064]

#adding new coordinates as columns to subdataframe above

geo_plot_df['Lat']=lat

geo_plot_df['Long']=long



#using scatter geo with above created subdataframe

fig = px.scatter_geo(data_frame=geo_plot_df, scope='south america',lat='Lat',lon='Long',

                     size='Count', color='State', projection='hammer')

fig.update_layout(

        title_text = '1998-2017 Top-10 States in Brazil with reported fires')

fig.show()



fire_file_gpd=gpd.GeoDataFrame(geo_plot_df,geometry=gpd.points_from_xy(geo_plot_df['Long'],geo_plot_df['Lat']))

fire_file_gpd.crs={'init':'epsg:4326'}



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

americas = world.loc[world['continent'].isin([ 'South America'])]

americas=americas.loc[americas['name']=='Brazil']



ax = americas.plot(figsize=(10,10), color='Yellow', linestyle=':', edgecolor='black')

fire_file_gpd.plot(ax=ax, markersize=50,color='red')
month_array_summer=['June','July','August']

month_array_fall=['September','October','November']

#leaving data only for hottest months

box_plot_df_summer=amazon_df.loc[amazon_df['Month'].isin(month_array_summer)]

box_plot_df_fall=amazon_df.loc[amazon_df['Month'].isin(month_array_fall)]

#visualizing reports

box_plot=go.Figure()



box_plot.add_trace(go.Box(y=box_plot_df_summer.Fire_Number, x=box_plot_df_summer.Month,

                          name='Summer', marker_color='firebrick',

                          boxpoints='all', jitter=0.5, whiskerwidth=0.2,

                          marker_size=2,line_width=2))

box_plot.add_trace(go.Box(y=box_plot_df_fall.Fire_Number, x=box_plot_df_fall.Month,

                         name='Fall', marker_color='blue',

                         boxpoints='all', jitter=0.5, whiskerwidth=0.2,

                          marker_size=2,line_width=2))



box_plot.update_layout(

        title_text = 'Distribution of Fire Reports from 1998-2017 in the hottest months')

box_plot.show()