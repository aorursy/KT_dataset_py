import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
import io
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

%matplotlib inline

from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
from scipy.misc import imread
import codecs
from subprocess import check_output
import folium 
from folium import plugins
from folium.plugins import HeatMap
terror=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1');
terror.head(5)
attackTypes =  pd.concat(objs=[terror['attacktype1_txt'], 
                                    terror['attacktype2_txt'],
                                    terror['attacktype3_txt']],
                     axis=0).reset_index(drop=True)
x = attackTypes.value_counts().index
y = attackTypes.value_counts().values

trace2 = go.Bar(
        x=x ,
        y=y,
        marker=dict(
            color=y,
            colorscale = 'Viridis',
            reversescale = True
        ),
        name="Attack types",    
    )
layout = dict(
        title="Attack types",
        #width = 900, height = 500,
        xaxis=go.layout.XAxis(
          automargin=True),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
    #         domain=[0, 0.85],
        ), 
    )
fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)
weaponTypes =  pd.concat(objs=[terror['weaptype1_txt'], 
                                    terror['weaptype2_txt'],
                                    terror['weaptype3_txt'],
                                    terror['weaptype4_txt']],
                     axis=0).reset_index(drop=True)
x = weaponTypes.value_counts().index
y = weaponTypes.value_counts().values

trace2 = go.Bar(
        x=x ,
        y=y,
        marker=dict(
            color=y,
            colorscale = 'Viridis',
            reversescale = True
        ),
        name="Weapons used",    
    )
layout = dict(
        title="Weapons used",
        #width = 900, height = 500,
        xaxis=go.layout.XAxis(
          automargin=True),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
    #         domain=[0, 0.85],
        ), 
    )
fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)
targetTypes =  pd.concat(objs=[terror['targtype1_txt'], 
                                    terror['targtype3_txt'],
                                    terror['targtype3_txt']],
                     axis=0).reset_index(drop=True)
x = targetTypes.value_counts().index
y = targetTypes.value_counts().values

trace2 = go.Bar(
        x=x ,
        y=y,
        marker=dict(
            color=y,
            colorscale = 'Viridis',
            reversescale = True
        ),
        name="Who sufferred the most",    
    )
layout = dict(
        title="Who sufferred the most",
        #width = 900, height = 500,
        xaxis=go.layout.XAxis(
          automargin=True),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
    #         domain=[0, 0.85],
        ), 
    )
fig1 = go.Figure(data=[trace2], layout=layout)
iplot(fig1)
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_count = terror_filter.groupby(['country_txt'])['targtype1_txt'].count()
countries = pd.DataFrame({'country':terror_count.index,'number':terror_count.values })
data = [dict(
    type='choropleth',
    locations=countries['country'],
    locationmode='country names',
    z=countries['number'],
    text=countries['country'],
    colorscale='Viridis',
    reversescale=True,
    marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
    colorbar = {'title': 'No of incidents'},
)]
layout = dict(
    title='No of incidents across the world to disrupt Private Citizens & Property',
    geo=dict(showframe=False, showcoastlines=True, projection=dict(type='mercator'))
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
terror_filter = terror[terror['targtype1_txt'] == "Military"]
terror_filter = terror_filter.groupby(['country_txt'])['targtype1_txt'].count()
data =  dict(
        type = 'choropleth',
        locations = terror_filter.index,
        locationmode = 'country names',
        z = terror_filter.values,
        text = terror_filter.index,
        colorscale='Viridis',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar = {'title': 'No of incidents'})

layout = dict( title = 'No of incidents across the world to oppose their own Military',
         geo = dict(showframe = False,
         projection = {'type' : 'mercator'}))

choromap3 = go.Figure(data = [data],layout=layout)
iplot(choromap3)
terror_filter = terror[terror['targtype1_txt'] == "Police"]
terror_filter = terror_filter.groupby(['country_txt'])['targtype1_txt'].count()
data =  dict(
        type = 'choropleth',
        locations = terror_filter.index,
        locationmode = 'country names',
        z = terror_filter.values,
        text = terror_filter.index,
        colorscale='Viridis',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar = {'title': 'No of incidents'})

layout = dict( title = 'No of incidents across the world to oppose Police',
         geo = dict(showframe = False,
         projection = {'type' : 'mercator'}))

choromap3 = go.Figure(data = [data],layout=layout)
iplot(choromap3)
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['targtype1_txt'].count().unstack()
#terror_filter.columns.name = None      
#terror_filter = terror_filter.reset_index()  
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['targtype1_txt'] == "Military"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['targtype1_txt'].count().unstack()
#terror_filter.columns.name = None      
#terror_filter = terror_filter.reset_index()  
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['targtype1_txt'] == "Police"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['targtype1_txt'].count().unstack()
#terror_filter.columns.name = None      
#terror_filter = terror_filter.reset_index()  
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_filter = terror[terror['country_txt'] == "India"]
terror_filter = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter[terror_filter['city']!='Unknown' ]
data = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter.drop_duplicates(subset=None, keep='first', inplace=False)
data_city = pd.DataFrame({
    'city':data['city'].value_counts().index,
   'value':data['city'].value_counts().values
})
data = [
    {
        'x': data_city['city'][0:5].values,
        'y': data_city['value'][0:5].values,
        'mode': 'markers',
        'marker': {
            'sizemode': 'area',
         #   'sizeref': 'sizeref',
            'size': data_city['value'][0:5]
              }
    }
]
iplot(data)
City_State = pd.merge(data_city, terror_filter, how='left', left_on='city', right_on='city')
City_State = City_State.drop_duplicates(subset='city', keep='first', inplace=False)
count = City_State['value'].values
m = folium.Map(location=[28,81], tiles="Mapbox Bright", zoom_start=4.5)
for i in range(0,5):
   folium.Circle(
      location=[City_State.iloc[i]['latitude'], City_State.iloc[i]['longitude']],
      #location=[20, 81],
      popup=City_State.iloc[i]['city'],
      radius=int(count[i])*300,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_filter = terror[terror['country_txt'] == "Iraq"]
terror_filter = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter[terror_filter['city']!='Unknown' ]
data = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter.drop_duplicates(subset=None, keep='first', inplace=False)
data_city = pd.DataFrame({
    'city':data['city'].value_counts().index,
   'value':data['city'].value_counts().values
})
data = [
    {
        'x': data_city['city'][0:5].values,
        'y': data_city['value'][0:5].values,
        'mode': 'markers',
        'marker': {
            'sizemode': 'area',
   #         'sizeref': 'sizeref',
            'size': data_city['value'][0:5]
              }
    }
]
iplot(data)
City_State = pd.merge(data_city, terror_filter, how='left', left_on='city', right_on='city')
City_State = City_State.drop_duplicates(subset='city', keep='first', inplace=False)
count = City_State['value'].values
m = folium.Map(location=[33,44], tiles="Mapbox Bright", zoom_start=4.5)
for i in range(0,5):
   folium.Circle(
      location=[City_State.iloc[i]['latitude'], City_State.iloc[i]['longitude']],
      #location=[20, 81],
      popup=City_State.iloc[i]['city'],
      radius=int(count[i])*20,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_filter = terror[terror['country_txt'] == "Afghanistan"]
terror_filter = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter[terror_filter['city']!='Unknown' ]
data = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter.drop_duplicates(subset=None, keep='first', inplace=False)
data_city = pd.DataFrame({
    'city':data['city'].value_counts().index,
   'value':data['city'].value_counts().values
})

data = [
    {
        'x': data_city['city'][0:5].values,
        'y': data_city['value'][0:5].values,
        'mode': 'markers',
        'marker': {
            'sizemode': 'area',
   #         'sizeref': 'sizeref',
            'size': data_city['value'][0:5]
              }
    }
]
iplot(data)
City_State = pd.merge(data_city, terror_filter, how='left', left_on='city', right_on='city')
City_State = City_State.drop_duplicates(subset='city', keep='first', inplace=False)
count = City_State['value'].values
m = folium.Map(location=[33,70], tiles="Mapbox Bright", zoom_start=4.5)
for i in range(0,5):
   folium.Circle(
      location=[City_State.iloc[i]['latitude'], City_State.iloc[i]['longitude']],
      #location=[20, 81],
      popup=City_State.iloc[i]['city'],
      radius=int(count[i])*100,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m
terror_filter = terror[terror['targtype1_txt'] == "Private Citizens & Property"]
terror_filter = terror[terror['country_txt'] == "Pakistan"]
terror_filter = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter[terror_filter['city']!='Unknown' ]
data = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter.drop_duplicates(subset=None, keep='first', inplace=False)
data_city = pd.DataFrame({
    'city':data['city'].value_counts().index,
   'value':data['city'].value_counts().values
})

data = [
    {
        'x': data_city['city'][0:5].values,
        'y': data_city['value'][0:5].values,
        'mode': 'markers',
        'marker': {
            'sizemode': 'area',
    #        'sizeref': 'sizeref',
            'size': data_city['value'][0:5]
              }
    }
]
iplot(data)

City_State = pd.merge(data_city, terror_filter, how='left', left_on='city', right_on='city')
City_State = City_State.drop_duplicates(subset='city', keep='first', inplace=False)
count = City_State['value'].values
m = folium.Map(location=[28,70], tiles="Mapbox Bright", zoom_start=4.5)
for i in range(0,5):
   folium.Circle(
      location=[City_State.iloc[i]['latitude'], City_State.iloc[i]['longitude']],
      #location=[20, 81],
      popup=City_State.iloc[i]['city'],
      radius=int(count[i])*100,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m
orgs = terror['gname'].value_counts().head(25).index

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(14, 10))
cmap = plt.get_cmap('coolwarm')

map = Basemap(projection='cyl')
map.drawmapboundary()
map.fillcontinents(color='lightgray', zorder=1)
org=['Taliban','Shining Path (SL)']
#plt.scatter(5,15,s=50000,cmap=cmap,color = 'lightblue',marker='o', 
#                   alpha=0.5, zorder=10)
plt.text(-60,70,'Terrorist Groups',color='r',fontsize=15)
plt.text(-60,65,'---------------------',color='r',fontsize=15)
j=60
for i in range(25) :
   if i > 0 :
     plt.text(-60,j,orgs[i],color='darkblue',fontsize=13)
   j = j - 6
plt.title('Top Terrorist Groups across the world')
plt.show()
#terror['claimmode_txt'].value_counts()
f, ax = plt.subplots(figsize=(9, 6)) 
sns.barplot( y = terror['gname'].value_counts().head(6).index,
            x = terror['gname'].value_counts().head(6).values,
                palette="GnBu_d")
ax.set_ylabel('')
ax.set_title('Active Terrorist Organizations' );
terror_filter = terror[terror['gname'] == "Taliban"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 5)) 
g = sns.heatmap(terror_filter[0:3],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['gname'] == "Shining Path (SL)"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 5)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['gname'] == "Islamic State of Iraq and the Levant (ISIL)"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15,8 )) 
g = sns.heatmap(terror_filter[0:20],annot=True,fmt="2.0f",cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['gname'] == "Farabundo Marti National Liberation Front (FMLN)"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values([1991], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 5)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['gname'] == "Al-Shabaab"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 5)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['gname'] == "Boko Haram"]
terror_filter = terror_filter.groupby(['country_txt','iyear'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 5)) 
g = sns.heatmap(terror_filter[0:10],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
#terror['claimmode_txt'].value_counts()
f, ax = plt.subplots(figsize=(9, 6)) 
sns.barplot( y = terror['claimmode_txt'].value_counts().index,
            x = terror['claimmode_txt'].value_counts().values,
                palette="GnBu_d")
ax.set_ylabel('')
ax.set_title('Different modes used to assume responsibility of the attacks by Terrorists' );
terror_filter = terror[['gname','claimmode_txt']]
terror_filter = terror_filter.groupby(['gname','claimmode_txt'])['gname'].count().unstack()
terror_filter = terror_filter.sort_values(['Personal claim','Posted to website, blog, etc.'], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(8, 10)) 
g = sns.heatmap(terror_filter[0:20],cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
killed_terror = terror[['city','nkill']]
terror_filter = terror[['city','latitude','longitude']]
terror_filter = terror_filter[terror_filter['city']!='Unknown' ]
data = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter.drop_duplicates(subset=None, keep='first', inplace=False)
data_city = pd.DataFrame({
    'city':killed_terror.dropna().groupby(['city'])['nkill'].sum().index,
   'value':killed_terror.dropna().groupby(['city'])['nkill'].sum().values
})

data_city = data_city.sort_values(['value'], ascending=False)

data = [
    {
        'x': data_city['city'][0:20].values,
        'y': data_city['value'][0:20].values,
        'mode': 'markers',
        'marker': {
            'sizemode': 'area',
    #        'sizeref': 'sizeref',
            'size': data_city['value'][0:20]
              }
    }
]
iplot(data)

City_State = pd.merge(data_city, terror_filter, how='left', left_on='city', right_on='city')
City_State = City_State.drop_duplicates(subset='city', keep='first', inplace=False)
City_State = City_State.dropna()
count = City_State['value'].values
m = folium.Map(location=[28,2], tiles='stamentoner', zoom_start=2)
for i in range(0,100):
   folium.Circle(
      location=[City_State.iloc[i]['latitude'], City_State.iloc[i]['longitude']],
      #location=[20, 81],
      popup=City_State.iloc[i]['city'],
      radius=int(count[i])*100,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m
killed_terror = terror[['city','nwound']]
terror_filter = terror[['city','latitude','longitude']]
terror_filter = terror_filter[terror_filter['city']!='Unknown' ]
data = terror_filter[['city','latitude','longitude']]
terror_filter = terror_filter.drop_duplicates(subset=None, keep='first', inplace=False)
data_city = pd.DataFrame({
    'city':killed_terror.dropna().groupby(['city'])['nwound'].sum().index,
   'value':killed_terror.dropna().groupby(['city'])['nwound'].sum().values
})

data_city = data_city.sort_values(['value'], ascending=False)

data = [
    {
        'x': data_city['city'][0:20].values,
        'y': data_city['value'][0:20].values,
        'mode': 'markers',
        'marker': {
            'sizemode': 'area',
  #          'sizeref': 'sizeref',
            'size': data_city['value'][0:20]
              }
    }
]
iplot(data)

City_State = pd.merge(data_city, terror_filter, how='left', left_on='city', right_on='city')
City_State = City_State.drop_duplicates(subset='city', keep='first', inplace=False)
City_State = City_State.dropna()
count = City_State['value'].values
m = folium.Map(location=[28,2], tiles='stamentoner', zoom_start=2)
for i in range(0,50):
   folium.Circle(
      location=[City_State.iloc[i]['latitude'], City_State.iloc[i]['longitude']],
      #location=[20, 81],
      popup=City_State.iloc[i]['city'],
      radius=int(count[i])*50,
      color='crimson',
      fill=True,
      fill_color='crimson'
   ).add_to(m)
m

terror_filter1 = terror[terror['propextent_txt'] == "Catastrophic (likely >= $1 billion)"]
terror_filter1 = terror_filter1[['country_txt','propvalue','iyear']]
terror_filter1 = terror_filter1.fillna(terror_filter1.propvalue.mean())

terror_filter = terror[terror['propextent_txt'] == "Major (likely >= $1 million but < $1 billion)"]
terror_filter = terror_filter[['country_txt','propvalue','iyear']]
terror_filter = terror_filter.fillna(terror_filter.propvalue.mean())
terror_filter = terror_filter.append(terror_filter1)
terror_filter = terror_filter.groupby(['country_txt'])['propvalue'].sum()
                                      
data =  dict(
        type = 'choropleth',
        locations = terror_filter.index,
        locationmode = 'country names',
        z = terror_filter.values,
        text = terror_filter.index,
        colorscale='Viridis',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar = {'title': 'Property Damage in US $'})

layout = dict( title = 'Extent of Property Damage in US $ across the world',
         geo = dict(showframe = False,
         projection = {'type' : 'mercator'}))

choromap3 = go.Figure(data = [data],layout=layout)
iplot(choromap3)
terror_filter = terror[terror['propextent_txt'] == "Catastrophic (likely >= $1 billion)"]
terror_filter = terror_filter[['country_txt','propvalue','iyear']]
terror_filter = terror_filter.fillna(terror_filter.propvalue.mean())
#terror_filter = terror_filter.append(terror_filter1)

terror_filter = terror_filter.groupby(['country_txt','iyear'])['propvalue'].sum().unstack()
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(12,3)) 
g = sns.heatmap(terror_filter,cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['propextent_txt'] == "Major (likely >= $1 million but < $1 billion)"]
terror_filter = terror_filter[['country_txt','propvalue','iyear']]
terror_filter = terror_filter.fillna(terror_filter.propvalue.mean())
#terror_filter = terror_filter.append(terror_filter1)

terror_filter = terror_filter.groupby(['country_txt','iyear'])['propvalue'].sum().unstack()
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 30)) 
g = sns.heatmap(terror_filter,cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()
terror_filter = terror[terror['propextent_txt'] == "Major (likely >= $1 million but < $1 billion)"]
#terror_filter = terror[terror['country_txt'] == "United States"]
terror_filter = terror_filter[['city','gname','propvalue','iyear']]
terror_filter = terror_filter.fillna(terror_filter.propvalue.mean())
terror_filter = terror_filter.sort_values(['iyear'], ascending=False)
terror_filter = terror_filter.groupby(['iyear'])['propvalue'].sum()
data = [
    {
        'x': terror_filter.index,
        'y': terror_filter.values,
        'mode': 'lines',
        'marker': {
            'sizemode': 'area',
 #           'sizeref': 'sizeref',
           }
    }
]
iplot(data)
def text_process(mess):
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    return(nopunc)
    # Now just remove any stopwords
    #return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

df = terror[['motive','country_txt','iyear']]
df = df.dropna()
df = df.reset_index(drop=True)
df['motive'] = df['motive'].apply(text_process)
years = [2009,2010,2011,2012,2013,2014,2015,2016]
plt.figure(figsize=(14,15))
gs = gridspec.GridSpec(4, 2)
for i, cn in enumerate(years):
    ax = plt.subplot(gs[i])
    df_country = df[df['iyear'] == cn]
    country_motive = df_country['motive'].str.lower().str.cat(sep=' ')
    words=nltk.tokenize.word_tokenize(country_motive)
    word_dist = nltk.FreqDist(words)
    stopwords = nltk.corpus.stopwords.words('english')
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(" ".join(words_except_stop_dist))
    ax.imshow(wordcloud)
    ax.set_title('Year ' + str(cn) + ' at a glance' )
    ax.axis('off')
df_country = df[df['country_txt'] == 'United States']
#df_country = df[df['iyear'] == 2014]
country_motive = df_country['motive'].str.lower().str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(country_motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(14,6)
plt.axis('off')
plt.show()
years = [2011,2012,2013,2014,2015,2016]
df_country = df[df['country_txt'] == 'United States']
plt.figure(figsize=(14,15))
gs = gridspec.GridSpec(3, 2)
for i, cn in enumerate(years):
    ax = plt.subplot(gs[i])
    df_time = df_country[df_country['iyear'] == cn]
    country_motive = df_time['motive'].str.lower().str.cat(sep=' ')
    words=nltk.tokenize.word_tokenize(country_motive)
    word_dist = nltk.FreqDist(words)
    stopwords = nltk.corpus.stopwords.words('english')
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(" ".join(words_except_stop_dist))
    ax.imshow(wordcloud)
    ax.set_title('Year ' + str(cn) + ' at a glance' )
    ax.axis('off')
df_country = df[df['country_txt'] == 'India']
country_motive = df_country['motive'].str.lower().str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(country_motive)
word_dist = nltk.FreqDist(words)
stopwords = nltk.corpus.stopwords.words('english')
words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(" ".join(words_except_stop_dist))
plt.imshow(wordcloud)
fig=plt.gcf()
fig.set_size_inches(14,6)
plt.axis('off')
plt.show()
years = [2011,2012,2013,2014,2015,2016]
df_country = df[df['country_txt'] == 'India']
plt.figure(figsize=(14,9))
gs = gridspec.GridSpec(3, 2)
for i, cn in enumerate(years):
    ax = plt.subplot(gs[i])
    df_time = df_country[df_country['iyear'] == cn]
    country_motive = df_time['motive'].str.lower().str.cat(sep=' ')
    words=nltk.tokenize.word_tokenize(country_motive)
    word_dist = nltk.FreqDist(words)
    stopwords = nltk.corpus.stopwords.words('english')
    words_except_stop_dist = nltk.FreqDist(w for w in words if w not in stopwords) 
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(" ".join(words_except_stop_dist))
    ax.imshow(wordcloud)
    ax.set_title('Year ' + str(cn) + ' at a glance' )
    ax.axis('off')
terror_filter = terror[terror['suicide'] == 1]
terror_filter = terror_filter.groupby(['gname','iyear'])['gname'].count().unstack()
terror_filter = terror_filter[[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2012,2013,2014,2015,2016]]
terror_filter = terror_filter.sort_values([2016], ascending=False)
terror_filter = terror_filter.fillna(0)
f, ax = plt.subplots(figsize=(15, 10)) 
g = sns.heatmap(terror_filter[0:10],annot=True,fmt="2.0f",cmap='YlGnBu',linewidths=.5,vmin=0.01)
plt.show()