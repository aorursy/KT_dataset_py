# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv', encoding ='ISO-8859-1')

data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country', 
                     'provstate':'City','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                     'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                     'weaptype1_txt':'Weapon_type', 'motive':'Motive'},inplace=True)

data = data[['Year','Month','Day','Country','City','latitude','longitude','AttackType','Killed','Wounded',
             'Target','Summary','Group','Target_type','Weapon_type','Motive']]

data = data[data.Country== 'Turkey']

data['Killed'] = data['Killed'].fillna(0)
data['Wounded'] = data['Wounded'].fillna(0)

data['Casualties'] = data['Killed'] + data['Wounded']

data.info()

data.Year.plot(kind = 'hist', color = 'b', bins=range(1970, 2018), figsize = (16,7), alpha=0.5, grid=True)
plt.xticks(range(1970, 2018), rotation=90, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Years", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Number of Attacks By Year", fontsize=16)
plt.show()
data.City.value_counts().drop('Unknown').head(10).plot.bar(figsize=[16,9], grid=True, alpha=0.8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Cities", fontsize=15)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Most Targeted Cities", fontsize=16)
plt.show()
from wordcloud import WordCloud
df = data[data.City != 'Unknown']
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(" ".join(df.City))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig("graph.png")
plt.show()
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.graph_objs as go

newData = pd.DataFrame(data.groupby(['City', 'Year']).Killed.sum())  # Groupby City and Year
#Chose five(5) cities that have highest death results. And if there is unknown cities, drop them.
cityNames = data.groupby('City').Killed.sum().sort_values(ascending=False).drop('Unknown').head().index
newData = newData.loc[cityNames]

trace = [0] * 5
i = 0
#Plot death result for each city years by years.
for city in cityNames:
    trace[i] = go.Scatter (
                        x = newData.loc[city].index,
                        y = newData.loc[city].Killed,
                        mode = "lines",
                        name = city,
                        marker=dict(size=12),
                        #text = city
    )
    i += 1
    
data_result = trace
layout_result = dict(title = "Top 5 Cities That Have Highest Death Results According to Years", hovermode='closest',
             xaxis = dict(title = "Years", ticklen = 5, zeroline = False),
             yaxis = dict(title = "Death Results", ticklen = 5, zeroline = False))
fig = dict(data = data_result, layout = layout_result)
iplot(fig)
casualties = data.groupby('Weapon_type').Casualties.sum()
casualties = casualties[ casualties > 0]
plt.figure()
casualties.plot.bar(alpha=0.8, grid=True, figsize=(16,6))
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel("Weapon Type", fontsize=15)
plt.ylabel("Casaulties", fontsize=15)
plt.title("Casulties by Weapon Type", fontsize=16)
plt.show()
data.Group.value_counts().head(10).plot.bar(figsize=[18,9], grid=True, alpha=0.8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Number of Attacks", fontsize=15)
plt.show()
pkk = data[data.Group == "Kurdistan Workers' Party (PKK)"]
pkk.City.value_counts().drop("Unknown").head(10).plot.bar(figsize=[16,8], grid=True, alpha=0.8)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel("Number of Attacks", fontsize=15)
plt.title("Most Targeted Cities", fontsize=16)
plt.show()
import plotly.plotly as py
pkk = pkk[pkk.latitude.notnull() & pkk.Casualties > 0]
pkk['text'] = "City : " + pkk["City"].astype(str) + " <br>"+"Year : " + pkk['Year'].astype(str) +\
                 " <br>" + "Casualties : " + (pkk["Casualties"].astype(int)).astype(str) +\
                 " <br>" + "Attack Type : " + pkk["AttackType"]
attacks = dict(
               type = 'scattergeo',
               lon = pkk['longitude'],
               lat = pkk['latitude'],
               text = pkk['text'],
               hoverinfo = 'text',
               mode = 'markers',
               marker = dict(
                     size = pkk["Casualties"] ** 0.25 * 10,
                     opacity = 0.7,
                     color = 'rgb(10, 160, 200)'
               )
         )
        
layout = dict(
            title = 'PKK Attacks in Turkey',
            hovermode='closest',
            geo = dict(
                showframe=False,
                showcountries = True,
                lonaxis = dict( range= [ 25.0, 46.0 ] ),
                lataxis = dict( range= [ 35.0, 43.0 ] ),
            )
         )
figure = dict(data = [attacks], layout = layout)
iplot(figure)