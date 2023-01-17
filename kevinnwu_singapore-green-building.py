import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

%matplotlib inline
df = pd.read_csv('../input/annual-energy-consumption-from-singapore-buildings/listing-of-building-energy-performance-data-for-commercial-buildings.csv')
df.head()
df.info()
df.isnull().sum()
sns.heatmap(df.isnull())
df = df.dropna(subset = ['buildingaddress','2017energyuseintensity','2018energyusintensity']).reset_index(drop=True)
df = df.drop(['buildingname','buildingsize','voluntarydisclosure'], axis=1).reset_index(drop=True)
df.isnull().sum()
df[['greenmarkrating','greenmarkyearaward']]
def splitpostalcode(address):

    code = address.split(', ')[1].split(' ')[1]

    return code
df['postalcode'] = df['buildingaddress'].apply(lambda x : splitpostalcode(x))
df['postalcode']
df['grossfloorarea'] = df['grossfloorarea'].str.replace(',','').astype(int)
df['buildingtype'].unique()
df['buildingtype'] = df['buildingtype'].apply(lambda x : 'University' if x == 'Univerisity' else x)
df = df.rename(columns={'2018energyusintensity':'2018energyuseintensity'})
df
color_2_types = ['DarkGray','LawnGreen']

color_rating = dict(Certified='DarkGreen', Legislated='Red', Gold='Gold', GoldPlus='DarkGoldenRod', NotCertified='Black', Platinum='#e5e4e2')
def newLegend(fig, newNames):

    for item in newNames:

        for i, elem in enumerate(fig.data[0].labels):

            if elem == item:

                fig.data[0].labels[i] = newNames[item]

    return(fig)
fig = px.pie(df, 

             names='greenmarkstatus',

             color_discrete_sequence=color_2_types, 

             title='Composition Green Building in Singapore', 

            )



fig.update_traces(textposition='inside', 

                  textinfo='percent+label+value',

                  textfont_size=16, 

                  showlegend=False,

                  )



fig = newLegend(fig = fig, 

                newNames = dict(No='Normal Building',

                                Yes='GreenBuilding',

                               )

               )

fig.show()
data1 = df[df['greenmarkstatus'] == 'Yes'].groupby('buildingtype').count()

data2 = df[df['greenmarkstatus'] == 'No'].groupby('buildingtype').count()



fig = go.Figure()

fig.add_trace(go.Bar(x=data1.index,

                     y=data1['greenmarkstatus'],

                     name='Green Building',

                     marker_color='LawnGreen',

                    )

             )



fig.add_trace(go.Bar(x=data2.index,

                     y=data2['greenmarkstatus'],

                     name='Normal Building',

                     marker_color='DarkGray',

                    )

             )



fig.update_layout(barmode='group',

                  xaxis=dict(categoryorder='total descending'),

                  yaxis=dict(tickvals=[0,150,300]),

                  title='Building Type Wise Green Building',

                 )



fig.show()
data = df[df['greenmarkstatus'] == 'Yes'].groupby(['greenmarkrating','buildingtype']).count().reset_index()



fig = px.treemap(data, 

                 path=['greenmarkrating', 'buildingtype'], 

                 values='greenmarkstatus',

                 color='greenmarkrating',

                 color_discrete_map=color_rating,

                 title='Rating Wise Green Building'

                )



fig.update_layout(uniformtext_minsize=14, 

                  uniformtext_mode='hide',

                 )



fig.show()
data = df.groupby('greenmarkyearaward').count().reset_index()



fig = px.scatter(df, 

                 x=data['greenmarkyearaward'], 

                 y=data['greenmarkstatus'], 

                 text=data['greenmarkstatus'], 

                 size_max=16,

                 trendline=True,

                 range_y=[0,65],

                 title='Year Wise Number of Green Building',

                )



fig.add_shape(

            type="rect",

            x0=2016.5,

            y0=0,

            x1=2018.5,

            y1=65,

            line=dict(width=0,

                     ),

            fillcolor="DarkOrange",

            opacity=0.3,

        )





fig.update_traces(textposition='top center',

                  textfont_size=16,

                  marker=dict(symbol=[20], 

                              color='LawnGreen',

                              size=8,

                             ),                 

                 )



fig.update_layout(yaxis=dict(tickvals=[0,30,60],

                             title=''),

                  xaxis=dict(tickvals=[2006,2008,2010,2012,2014,2016,2017,2018],

                             title=''),

                 )



fig.show()
data = df[df['greenmarkstatus']=='Yes'].sort_values('greenmarkyearaward', ascending=False)



fig = px.parallel_categories(data,

                             dimensions=['greenmarkyearaward', 'greenmarkrating','buildingtype'],

                             labels=dict(greenmarkyearaward='Year', 

                                              greenmarkrating='Rating', 

                                              buildingtype='Type'),

                             title='Green Building at Singapore',

                            )

fig.show()
colors = ['DarkGreen', 'Gold', 'DarkGoldenRod', 'Red', 'Black', '#e5e4e2']



x_data=['2017','2018']

y_data = df.fillna('NotCertified').groupby('greenmarkrating').mean()[['2017energyuseintensity','2018energyuseintensity']].reset_index()



fig = go.Figure()



for i in range(0, 6):

    fig.add_trace(go.Scatter(x=x_data, 

                             y=y_data.iloc[i][['2017energyuseintensity','2018energyuseintensity']], 

                             mode='lines+markers',

                             name=y_data.iloc[i]['greenmarkrating'],

                             line=dict(color=colors[i], width=4),

                             marker=dict(size=16),

                            )

                 )

    

    

fig.update_layout(xaxis=dict(range=['2017','2018'], 

                             nticks=2, 

                             linecolor='Black', 

                             tickfont=dict(size=16)

                            ),

                  yaxis=dict(range=[230,320], 

                             tickvals=[230,280,320],

                             linecolor='Black', 

                             tickcolor='Grey', 

                             mirror='allticks', 

                             title='Energy Intensity',

                             tickfont=dict(size=16)

                            ),

                  showlegend=True,

                  width=800,

                  height=600,

                  title='Energy Use Intensity from 2017-2018'

                  )



fig.show()
data = df

data['percent'] = (data['2018energyuseintensity']-data['2017energyuseintensity'])/data['2017energyuseintensity']*100

data = data.fillna('NotCertified')



fig = px.box(data,

                   x="percent", 

                   title='Percentage Diference between 2017 and 2018', 

                   color='greenmarkrating',

                   color_discrete_map=color_rating,

                  )



fig.update_layout(xaxis=dict(title='Difference Energy Usage (%)'),

                  yaxis=dict(visible=False))

fig.show()
import folium

import math

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster
sing = pd.read_csv('../input/sg-postal-code/SG.txt',sep='\t', lineterminator='\n', error_bad_lines=False, header=None)

sing = sing.drop([3,4,5,6,7,8,11], axis=1)

sing.columns = ['Country', 'Postal_Code', 'Address', 'Lat', 'Long']

sing.drop_duplicates(subset ="Postal_Code",  keep = False, inplace = True)
def getlongtitude (postal):

    long = sing[sing['Postal_Code'] == int(postal)]['Long'].values

    if len(long) == 1:

        return long[0]

    else:

        return np.nan



def getlatitude (postal):

    lat = sing[sing['Postal_Code'] == int(postal)]['Lat'].values

    if len(lat) == 1:

        return lat[0]

    else:

        return np.nan
df = pd.concat([df, df['postalcode'].apply(lambda x: pd.Series({'Long':getlongtitude(x), 'Lat':getlatitude(x)}))], axis=1)
data = df.dropna()

# Create the map

m_3 = folium.Map(location=[1.283333, 103.833333], tiles='cartodbpositron', zoom_start=11)



# Add points to the map

mc = MarkerCluster()

for idx, row in data.iterrows():

    if not math.isnan(row['Long']) and not math.isnan(row['Lat']):

        mc.add_child(Marker([row['Lat'], row['Long']]))

m_3.add_child(mc)