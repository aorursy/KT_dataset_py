# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import pandas as pd #data processing

import numpy as np #linear algebra

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns #Data visualization

import math

from mpl_toolkits.basemap import Basemap

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read Data

gtdata=pd.read_csv("/kaggle/input/gtd/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1')
#Show top data rows

gtdata.head()
#Renaming the columns

gtdata.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',\

                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',\

                       'nwound':'Wounded','summary':'Summary',\

                          'gname':'Group','targtype1_txt':'Target_type',\

                          'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
#Read data with new column names

gtdata=gtdata[['Year','Month','Day','Country','state','Region','city',\

               'latitude','longitude','AttackType','Target','Killed','Wounded',

               'Summary','Group','Target_type','Weapon_type','Motive']]

gtdata.head()
#Show information about the data 

gtdata.info()
#Show the columns in the data



gtdata.columns
#See the no of rows and columns in the data 

gtdata.shape
#See the null values



gtdata.isnull().sum()
#See the unique values in Country column

gtdata.Country.unique()
#Show the data types of all the columns

gtdata.dtypes
#Terrorist attacks by Year

sns.set(font_scale=1.5)

plt.figure(figsize=(20,10))

sns.countplot(x="Year",data=gtdata)

plt.xlabel('')

plt.ylabel('')

plt.xticks(rotation=45)

plt.title("Terrorist attacks by Year")

plt.tight_layout()



#Terrorist attacks by Month

sns.set(font_scale=1.5)

plt.figure(figsize=(20,10))

sns.countplot(x="Month",data=gtdata)

plt.xlabel('')

plt.ylabel('')

plt.xticks(rotation=45)

plt.title("Terrorist attacks by Month")

plt.tight_layout()





#Top 20 countries with highest crime rate

top_countries=gtdata.Country.value_counts(dropna=True)

t=top_countries.head(20)



sns.set(font_scale=1.5)

plt.figure(figsize=(16,10))

sns.barplot(x=t.index,y=t.values,data=gtdata)

plt.xlabel('Countries')

plt.ylabel('')

plt.xticks(rotation=45)

plt.title("Top 20 countries with highest crime rate",fontsize=20)

plt.tight_layout()
#Top 20 cities with highest crime rate

top_cities=gtdata.city.value_counts(dropna=True)

t=top_cities.head(20)

t
#Visualize

sns.set(font_scale=1.5)

plt.figure(figsize=(16,10))

sns.barplot(x=t.index,y=t.values,data=gtdata)

plt.xlabel('City')

plt.ylabel('')

plt.xticks(rotation=45)

plt.title("Top 20 cities with highest crime rate",fontsize=20)

plt.tight_layout()
#Show unique values of attacktype column

gtdata.AttackType.unique()
#Top attacks 

cnt=gtdata.AttackType.value_counts()

c=cnt.head(10)

c





sns.set(font_scale=1.5)

plt.figure(figsize=(12,10))

sns.barplot(x=c.values,y=c.index,data=gtdata)

plt.xlabel('')

plt.ylabel('Attack Type 1')

plt.xticks(rotation=45)

plt.title("Top attacks",fontsize=20)

plt.tight_layout()



#Group with the most attacks

u=gtdata.Group.value_counts()

t1=u.head()

print( "Group with the most attacks:", u.index[1],"and the count is :", u.values[1])
sns.set(font_scale=1.5)

plt.figure(figsize=(10,8))

sns.barplot(x=t1.values,y=t1.index,data=gtdata)

plt.xlabel("Count")

plt.ylabel("Most attacks")

plt.title("Group with the most attacks:")

plt.xticks(rotation=45)

plt.tight_layout()
# Most no of attacks

print("Most Attack Types:", gtdata['AttackType'].value_counts().idxmax())



#Top 40 Worst Terror Attacks in History from 1982 to 2016





gtdata['Wounded']=gtdata['Wounded'].fillna(0).astype(int)

gtdata['Killed']=gtdata['Killed'].fillna(0).astype(int)

gtdata['Casualities']=gtdata['Wounded']+ gtdata['Killed']

gtdata1=gtdata.sort_values(by='Casualities',ascending=False)[:40]

heat= gtdata1.pivot_table(index='Country',columns='Year',values='Casualities')

heat.fillna(0,inplace=True)







import plotly.offline as py



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]

heatmap=go.Heatmap(z=heat.as_matrix(), x=heat.columns, y= heat.index, colorscale=colorscale)

data=[heatmap]

layout=go.Layout(title="Top 40 Worst Terror Attacks in History from 1982 to 2016",xaxis=dict(ticks='',nticks=20),\

                yaxis=dict(ticks=''))

fig=go.Figure(data=data,layout=layout)

py.iplot(fig, filename='heatmap',show_link=False)



sns.set(font_scale=1.5)

plt.figure(figsize=(18,10))

sns.barplot(gtdata['Country'].value_counts()[:15].index,gtdata['Country'].value_counts()[:15].values,\

           palette='Blues_d')

plt.xlabel("Countries")

plt.ylabel("Count")

plt.title("Top countries affected by Attacks")

plt.xticks(rotation=45)

plt.tight_layout()
# History of worst terror attacks





gtdata['Wounded']=gtdata['Wounded'].fillna(0).astype(int)

gtdata['Killed']=gtdata['Killed'].fillna(0).astype(int)

gtdata['Casualities']=gtdata['Wounded']+ gtdata['Killed']

gtdata1=gtdata.sort_values(by='Casualities',ascending=False)[:40]

heat= gtdata1.pivot_table(index='Country',columns='Year',values='Casualities')

heat.fillna(0,inplace=True)







#Countries with most no of terror attacks ( not inlcuding Unknown)

gtdata_bubble=gtdata[(gtdata['Group']!='Unknown')& (gtdata['Casualities']>50)]

gtdata_bubble.sort_values(by='Casualities',ascending=False).head()
gtdata_bubble=gtdata_bubble.sort_values(['Region','Country'])
gtdata_bubble=gtdata_bubble.drop(['latitude','longitude','Target','Summary','Motive'],axis=1)
gtdata_bubble=gtdata_bubble.dropna(subset=['city'])
gtdata_bubble.isnull().sum()
#Top Five country those have suffered most attacks

gtdata_bubble.Country.value_counts().head()


hover_text=[]

for i,row in gtdata_bubble.iterrows():

    hover_text.append(('City: {city}<br>'+

                      'Group: {group}<br>'+

                      'Casualities: {casualities}<br>'+

                      'Year: {year}').format(city=row['city'],group=row['Group'],casualities=row['Casualities'],

                                            year=row['Year']))

gtdata_bubble['text']=hover_text
trace0=go.Scatter(



    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Iraq'],

    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Iraq'],

    mode='markers',

    name='Iraq',

    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Iraq'],

    marker=dict(

    

        symbol='circle',

        sizemode='area',

        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Iraq'],

        line=dict(width=2),

    )

    

)

trace1=go.Scatter(



    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Afghanistan'],

    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Afghanistan'],

    mode='markers',

    name='Afghanistan ',

    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Afghanistan'],

    marker=dict(

    

        symbol='circle',

        sizemode='area',

        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Afghanistan'],

        line=dict(width=2),

    )

    

)

trace2=go.Scatter(



    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Pakistan'],

    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Pakistan'],

    mode='markers',

    name='Pakistan',

    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Pakistan'],

    marker=dict(

    

        symbol='circle',

        sizemode='area',

        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Pakistan'],

        line=dict(width=2),

    )

    

)

trace3=go.Scatter(



    x=gtdata_bubble['Year'] [gtdata_bubble['Country']=='Nigeria'],

    y=gtdata_bubble['Casualities'] [gtdata_bubble['Country']=='Nigeria'],

    mode='markers',

    name='Nigeria',

    text=gtdata_bubble['text'][gtdata_bubble['Country']=='Nigeria'],

    marker=dict(

    

        symbol='circle',

        sizemode='area',

        size= gtdata_bubble['Casualities'][gtdata_bubble['Country']=='Nigeria'],

        line=dict(width=2),

    )

    

)
data=[trace0,trace1,trace2,trace3]

layout=go.Layout(

         title = 'Top 4 countries',

         xaxis = dict(

             title = 'Year',

             

             range = [1976,2016],

             tickmode = 'auto',

             nticks = 30,

             showline = True,

             showgrid = False

             ),

         yaxis = dict(

             title = 'Casualities',

             type = 'log',

             range = [1.8,3.6],

             tickmode = 'auto',

             nticks = 40,

             showline = True,

             showgrid = False),

         paper_bgcolor='rgb(243, 243, 243)',

         plot_bgcolor='rgb(243, 243, 243)',

         )



fig=go.Figure(data=data, layout=layout)

py.iplot(fig,filename='Terrorism')
# Which groups have attacked the most

gtdata.Group.value_counts()[1:15]
gtdata_terror=gtdata[gtdata.Group.isin(['Taliban','Islamic State of Iraq and the Levant (ISIL)','Shining Path'])]



# In which country 'Taliban','Islamic State of Iraq and the Levant (ISIL)','Shining Path' has attacked the most

gtdata_terror.Country.unique()
# Map countries with terror attacks using folium



gtdata_Group = gtdata.dropna(subset=['latitude','longitude'])
gtdata_Group = gtdata_Group.drop_duplicates(subset=['Country','Group'])





terrorist_groups = gtdata.Group.value_counts()[1:8].index.tolist()





gtdata_Group = gtdata_Group.loc[gtdata_Group.Group.isin(terrorist_groups)]



gtdata_Group.Group.unique()



import folium

from folium.plugins import MarkerCluster

m1 = folium.Map(location=[20, 0], tiles="Stamenterrain", zoom_start=2)

marker_cluster = MarkerCluster(

    name='clustered icons',

    overlay=True,

    control=False,

    icon_create_function=None

)

for i in range(0,len(gtdata_Group)):

    marker=folium.Marker([gtdata_Group.iloc[i]['latitude'],gtdata_Group.iloc[i]['longitude']]) 

    popup='Group:{}<br>Country:{}'.format(gtdata_Group.iloc[i]['Group'],

                                          gtdata_Group.iloc[i]['Country'])

    folium.Popup(popup).add_to(marker)

    marker_cluster.add_child(marker)

marker_cluster.add_to(m1)

folium.TileLayer('openstreetmap').add_to(m1)

folium.TileLayer('Mapbox Bright').add_to(m1)

folium.TileLayer('cartodbdark_matter').add_to(m1)

folium.TileLayer('stamentoner').add_to(m1)

folium.LayerControl().add_to(m1)

m1.save('Terrorist_Organizations_in_Country_cluster.html')
m1