import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
%matplotlib inline
import seaborn as sns
#Plotly
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.set_config_file(offline=True)

dataset = pd.read_csv('../input/globalterrorismdb_0718dist.csv', engine='python')
dataset.head()
dataset.columns
plt.figure(figsize=(16,7))
sns.heatmap(dataset.isnull(),cmap='viridis',cbar=False)
dataset=dataset[['eventid','iyear', 'imonth', 'country_txt', 'region_txt', 'provstate', 'city', 'latitude', 'longitude', 'success', 'attacktype1_txt', 'targtype1_txt', 'natlty1_txt', 'gname', 'weaptype1_txt', 'weapsubtype1_txt', 'nkill', 'nwound']]

dataset.rename(columns={'imonth': 'Month',
 'iyear': 'Year',
 'eventid': 'Event ID',
 'country_txt': 'Country',
 'region_txt': 'Region',
 'provstate': 'State',
 'city': 'City',
 'attacktype1_txt': 'Attack type',
 'targtype1_txt': 'Target',
 'natlty1_txt': 'Nationality',
 'gname': 'Terrorist Group',
 'weaptype1_txt': 'Weapon type',
 'weapsubtype1_txt': 'Weapon subtype',
 'nkill': 'Killed',
 'nwound': 'Wounded'},inplace=True)
dataset.head()
plt.figure(figsize=(16,7))
sns.heatmap(dataset.isnull(),cmap='viridis',cbar=False)
dataset['Weapon subtype'].fillna('No Record',inplace = True)
dataset['Nationality'].fillna('Unknown', inplace = True)
dataset.dropna(inplace = True)
dataset['Casualties'] = dataset['Killed']+dataset['Wounded']
plt.figure(figsize=(16,7))
sns.heatmap(dataset.isnull(),cmap='viridis',cbar=False)
India = dataset[dataset['Country']=='India']
India.head()
Attack_counts = India.Year.value_counts()
sns.set_context(context='notebook',font_scale=1.8)
plt.figure(figsize=(16,9))
sns.barplot(Attack_counts.index, Attack_counts.values)
plt.xticks(rotation=90)

filter =India[India['Terrorist Group'].isin(India['Terrorist Group'].value_counts()[0:15].index)][['Casualties','Terrorist Group']].groupby('Terrorist Group').sum().reset_index().merge(India[India['Terrorist Group'].isin(India['Terrorist Group'].value_counts()[0:15].index)]['Terrorist Group'].value_counts().to_frame().reset_index().rename(columns={'index':'Terrorist Group','Terrorist Group':'Attacks'}), on='Terrorist Group').sort_values(by='Attacks',ascending=False)
filter.plot.barh(x='Terrorist Group',y=['Attacks','Casualties'],figsize=(12,10),stacked=True)
filter = India[India['State']=='Jammu and Kashmir']['Terrorist Group'].value_counts()[0:10]
sns.set_context(context='notebook',font_scale=1.5)
plt.figure(figsize=(16,6))
plt.xticks(rotation=90)
sns.barplot(filter.index, filter.values)
India['State'].value_counts()[0:10].to_frame().plot.bar(figsize=(16,6),width=0.3)
plt.ylabel('Attacks')

data = [go.Scattermapbox(
            lat= India['latitude'] ,
            lon= India['longitude'],
            customdata = India['Event ID'],
            mode='markers',
            marker=dict(
                size= 3.5,
                color = 'red',
                opacity = .7,
            ),
          )]
layout = go.Layout(autosize=False,
                   mapbox= dict(accesstoken="pk.eyJ1Ijoic2hhejEzIiwiYSI6ImNqYXA3NjhmeDR4d3Iyd2w5M2phM3E2djQifQ.yyxsAzT94VGYYEEOhxy87w",
                                bearing=0,
                                pitch=50,
                                zoom=4.5,
                                center= dict(
                                         lat=19.5,
                                         lon=80.5),
                                style= "mapbox://styles/shaz13/cjk4wlc1s02bm2smsqd7qtjhs"),
                    width=900,
                    height=600, title = "Terrorist attack locations in india")
fig = dict(data=data, layout=layout)
iplot(fig)
pd.crosstab(India[India['Terrorist Group'].isin(India['Terrorist Group'].value_counts()[1:6].index)]['Year'],India[India['Terrorist Group'].isin(India['Terrorist Group'].value_counts()[1:6].index)]['Terrorist Group']).plot(color=sns.color_palette('Paired',10),figsize=(18,6))
v1=India[India['Terrorist Group'].isin(India['Terrorist Group'].value_counts()[1:11].index)]
pd.crosstab(v1['Terrorist Group'],v1['Attack type']).plot.barh(stacked=True,figsize=(9,5),width=0.7)
plt.legend(loc=9,bbox_to_anchor=(1.05,1.05))