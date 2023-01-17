# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

from datetime import datetime

from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image

import squarify



import plotly as py

import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.set_option('display.max_columns', 500)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
terrorism_df = pd.read_csv(r'../input/gtd/globalterrorismdb_0718dist.csv', encoding = "iso-8859-1")
terrorism_df.head(2)
terrorism_df.shape
terrorism_df.columns
terrorism_df.rename(columns= {'iyear':'Year', 

                              'imonth':'Month', 

                              'iday':'Day', 

                              'resolution': 'Resolution', 

                              'country_txt':'Country', 

                              'region_txt': 'Region', 

                              'city':'City', 

                              'latitude': 'Latitude', 

                              'longitude':'Longitude', 

                              'summary':'Summary', 

                              'attacktype1_txt':'AttackType', 

                              'targtype1_txt': 'TargetType', 

                              'targsubtype1_txt': 'TargetSubtype', 

                              'corp1':'TargetedCorporation', 

                              'target1':'Target', 

                              'natlty1_txt': 'TargetNat', 

                              'gname': 'GroupName', 

                              'motive': 'Motive', 

                              'weaptype1_txt':'Weapon_type', 

                              'nkill':'Killed',

                              'nwound':'Wounded', 

                              'propextent_txt': 'PropExtent', 

                              'propvalue': 'PropCost', 

                              'nhours':'Hours', 

                              'ndays': 'Days', 

                              'kidhijcountry': 'TerroristCountry', 

                              'ransomamt':'RansomAmount', 

                              'ransompaid':'RansomPaid'}, inplace =True )
terrorism_df = terrorism_df[['Year', 

                              'Month', 

                              'Day', 

                              'Resolution', 

                              'Country', 

                              'Region', 

                              'City', 

                              'Latitude', 

                              'Longitude', 

                              'Summary', 

                              'AttackType', 

                              'TargetType', 

                              'TargetSubtype', 

                              'TargetedCorporation', 

                              'Target', 

                              'TargetNat', 

                              'GroupName', 

                              'Motive', 

                              'Weapon_type', 

                              'Killed',

                              'Wounded', 

                              'PropExtent', 

                              'PropCost', 

                              'Hours', 

                              'Days', 

                              'TerroristCountry', 

                              'RansomAmount', 

                              'RansomPaid']]
terrorism_df['Casualities'] = terrorism_df['Killed'] + terrorism_df['Wounded']
terrorism_df.shape
terrorism_df.head(2)
terrorism_df['Region'].value_counts()
plt.figure(figsize = (25,10))

ax = sns.countplot(terrorism_df['Year'], palette= 'Greens')

plt.xticks(rotation = 90)

ax.set_xlabel('Year', fontdict = {'size':15})

ax.set_ylabel('Count', fontdict = {'size':15})

ax.set_title('Terrorism per Year', fontdict = {'size':20, 'weight':'bold'})

ax.grid(True, c= 'black')
plt.figure(figsize=(10,20))

wordcloud = WordCloud(background_color= 'black', 

                      max_font_size=500, 

                      collocations= False, 

                      relative_scaling= 0.4, 

                      colormap= "Reds").generate(''.join([x for x in terrorism_df['Region'].str.replace(' ','')]))

plt.title('Frequent Attack Regions', fontdict = {'size':20, 'weight':'bold'})

plt.imshow(wordcloud, interpolation= 'bilinear')

plt.tight_layout(pad=0)

plt.axis('off')

plt.show()
ax = pd.crosstab(terrorism_df['Year'], terrorism_df['Region']).plot(color = sns.color_palette('Spectral_r',12))

fig = plt.gcf()

fig.set_size_inches(18,6)

ax.grid(True, color = 'black')

plt.show()
plt.figure(figsize=(10,20), 

#            facecolor='k'

          )

wordcloud = WordCloud(background_color= 'black',width=1600, height=800, max_font_size=500, collocations= False, relative_scaling= 0.4, 

                      colormap= "Reds").generate(' '.join([x for x in terrorism_df['AttackType'].str.replace(' ','')]))

plt.title('Frequent Attack Types', fontdict = {'size':20, 'weight':'bold'})

plt.imshow(wordcloud, interpolation= 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
plt.figure(figsize = (25,10))

ax = sns.countplot(terrorism_df['AttackType'], palette= 'viridis', order= terrorism_df['AttackType'].value_counts().index)

plt.xticks(rotation = 90)

ax.set_xlabel('Attack Types', fontdict = {'size':15})

ax.set_ylabel('Count', fontdict = {'size':15})

ax.set_title('Frequency of Attack Types', fontdict = {'size':20, 'weight':'bold'})

ax.grid(True, c= 'black')
pd.crosstab(terrorism_df['Region'],terrorism_df['AttackType'])
ax = pd.crosstab(terrorism_df['Region'],terrorism_df['AttackType']).plot.barh(stacked = True, width = 1, color = sns.color_palette('magma_r',12))

fig = plt.gcf()

fig.set_size_inches(18,6)

ax.grid(True, color = 'black')

plt.show()
pd.crosstab(terrorism_df['Region'], terrorism_df['TargetType'])
plt.figure(figsize=(10,20),

#            facecolor='k'

          )

wordcloud = WordCloud(background_color= 'black', 

                      max_font_size=500, 

                      collocations= False, 

                      relative_scaling= 0.2, 

                      colormap = 'Reds', 

                      stopwords = ['Unknown','Member','Bu']).generate(' '.join([str(x).replace(' ','') for x in terrorism_df['Target']]))

plt.title('Frequent Targets', fontdict = {'size':20, 'weight':'bold'})

plt.imshow(wordcloud, interpolation="bilinear")

plt.tight_layout(pad= 0)

plt.axis('off')

plt.show()
plt.figure(figsize = (25,10))

ax = sns.countplot(terrorism_df['Target'], palette= 'Reds_r', order= terrorism_df['Target'].value_counts().iloc[:10].index.drop(['Unknown']))

# plt.xticks(rotation = 90)

ax.set_xticklabels(terrorism_df['Target'].value_counts().iloc[:10].index.drop(['Unknown']), fontsize = 15)

ax.set_yticklabels(range(0,7000,1000), fontsize = 15)

ax.set_xlabel('Targets', fontdict = {'size':20})

ax.set_ylabel('Count', fontdict = {'size':20})

ax.set_title('Major Targets who suffer', fontdict = {'size':25, 'weight':'bold'})

ax.grid(True, c= 'black')
ax =  pd.crosstab(terrorism_df['Region'],terrorism_df['TargetType']).plot.barh(stacked = True, width = 2, color = sns.color_palette('viridis',12))

fig = plt.gcf()

fig.set_size_inches(20,8)

ax.grid(True, color = 'black')

plt.show()
plt.figure(figsize=(10,20), 

#            facecolor='k'

          )

wordcloud = WordCloud(background_color= 'black', 

                      max_font_size=500, 

                      collocations= False, 

                      relative_scaling= 0.2, 

                      colormap = 'Reds').generate(' '.join([x for x in terrorism_df['Country']]))

plt.title('Frequent Attack Countries', fontdict = {'size':20, 'weight':'bold'})

plt.imshow(wordcloud, interpolation="bilinear")

plt.tight_layout(pad = 0)

plt.axis('off')

plt.show()
Image.open('../input/iraqflag/iraqflag.png')
plt.figure(figsize = (20,5))

ax = sns.countplot(terrorism_df['Country'], order = terrorism_df['Country'].value_counts().iloc[:10].index, palette='Purples_r')

ax.set_xlabel('Country', fontdict = {'size': 15})

ax.set_ylabel('Frequency', fontdict = {'size': 15})

ax.set_title('Frequency of terrorism in Country',fontdict = {'size': 20,'weight':'bold'})

ax.grid(True, color= 'black')
terrorism_df.head(2)
terrorism_df.groupby(['TargetType'])['PropCost'].sum().reset_index()
terrorism_df.groupby(['TargetType'])['PropCost'].sum().sort_values(ascending = False).reset_index()
plt.figure(figsize=(20,10))

ax = sns.barplot(y = terrorism_df.groupby(['TargetType'])['PropCost'].sum().sort_values(ascending = False).reset_index()['PropCost'],

                 x = pd.Series(terrorism_df['TargetType'].value_counts().iloc[:10].index), palette='magma', label = True)

plt.xticks(rotation = 90)

ax.set_xlabel('Target Type', fontdict = {'size':15})

ax.set_ylabel('Property Cost', fontdict = {'size':15})

ax.set_title('Targets loss of Property', fontdict = {'size':20, 'weight':'bold'})

ax.grid(True, color = 'black')
city_to_state = pd.read_csv('../input/usa-city-to-state/cities.csv', encoding = 'iso-8859-1')
state_abr = pd.read_csv('../input/usa-latlong-for-state-abbreviations/statelatlong.csv')
state_city_abr = city_to_state.merge(state_abr, left_on = 'Alabama', right_on = 'City')
state_city_abr.head(2)
terrorism_df = terrorism_df.merge(state_city_abr, how = 'left', left_on = 'City', right_on = 'Alexander City')
terrorism_df.head(2)
terrorism_df.drop(['Alexander City', 'Latitude_y', 'Longitude_y', 'City_y'], axis= 1, inplace = True)
terrorism_df.rename(columns= {'Alabama':'USA-states_txt', 

                              'State':'USA-states', 

                              'City_x':'City', 

                              'Latitude_x':'Latitude', 

                              'Longitude_x':'Longitude'},inplace = True)
terrorism_df.head(2)
terrorism_df.groupby(['USA-states'])['Casualities'].sum().sort_values(ascending = False).reset_index().head()
plt.figure(figsize=(20,10))

ax = sns.barplot(y = terrorism_df.groupby(['USA-states_txt'])['Casualities'].sum().sort_values(ascending = False).reset_index().iloc[:30]['Casualities'],

                 x = terrorism_df.groupby(['USA-states_txt'])['Casualities'].sum().sort_values(ascending = False).reset_index().iloc[:30]['USA-states_txt'], palette='magma', label = True)

plt.xticks(rotation = 90)

ax.set_xlabel('States', fontdict = {'size':15})

ax.set_ylabel('Casualities', fontdict = {'size':15})

ax.set_title('Casualities Count as per State', fontdict = {'size':20, 'weight':'bold'})

ax.grid(True, color = 'black')
from plotly.offline import iplot
USA_city_casualities = pd.DataFrame({'State':terrorism_df.groupby(['USA-states'])['Casualities'].sum().sort_values(ascending = False).reset_index()['USA-states'], 

                                     'Casualities':terrorism_df.groupby(['USA-states'])['Casualities'].sum().sort_values(ascending = False).reset_index()['Casualities'], 

                                     'State_txt':terrorism_df.groupby(['USA-states_txt'])['Casualities'].sum().sort_values(ascending = False).reset_index()['USA-states_txt']})
data = dict(type='choropleth',

            colorscale = 'viridis',

            reversescale = True,

            locations = USA_city_casualities['State'],

            z = USA_city_casualities['Casualities'],

            locationmode = 'USA-states',

            text = '<br>'+USA_city_casualities['State_txt'],

            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),

            colorbar = {'title':"Casualities Count in Thousands"}

            )
layout= dict(title = '<b>Casualities Count</b>',

              geo = dict(scope='usa',

                         showlakes = True,

                         lakecolor = 'rgb(85,173,240)')

             )
choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
terrorism_df.head(2)