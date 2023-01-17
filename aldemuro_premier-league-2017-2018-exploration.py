import pandas as pd

import matplotlib.pyplot as plt

#import matplotlib.patches as mpatches

import seaborn as sns

import numpy as np



import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)

import numpy as np
#insert the data

df = pd.read_csv('../input/epldata_final.csv')

df['fpl_ratio'] = pd.DataFrame(df['fpl_points']/df['fpl_value'])

df.head()
#transport type of fpl_selection 

df['fpl_sel'] = df['fpl_sel'].replace('%','',regex=True).astype('float')/100

#print(df.dtypes)

df.head()
#These countries bellow is listed as united kingdom in iso3166, thus their value have to be change

df=df.replace("Northern Ireland","United Kingdom")

df=df.replace("England","United Kingdom")

df=df.replace("Wales","United Kingdom")

df=df.replace("Scotland","United Kingdom")

df=df.replace("United Kingdom","United Kingdom of Great Britain and Northern Ireland")
pd.crosstab(df.club,df.position).plot.bar(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))



fig=plt.gcf()

fig.set_size_inches(12,8)

plt.title('Number Of Player in every Clubs and position')



plt.show()
#df.groupby(['club']).count()


plt.subplots(figsize=(15,6))

sns.set_color_codes()

sns.distplot(df['age'], color = "R")

plt.xticks(rotation=90)

plt.title('Distribution of Premier League Players Age')

plt.show()
#number of premier league player position

plt.subplots(figsize=(15,6))

sns.countplot('position',data=df,palette='hot',edgecolor=sns.color_palette('dark',7),order=df['position'].value_counts().index)

plt.xticks(rotation=90)

plt.title('number of premier league player position')

plt.show()
#most market value

dfmarketv = df.nlargest(10, 'market_value').sort_values('market_value',ascending=False)

plt.subplots(figsize=(15,6))

sns.barplot(x="name", y="market_value",  data=dfmarketv ,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('top 10 bigest market value of premier league player season 2017/2018')

plt.show()
#df.nlargest(10, 'market_value').sort_values('market_value',ascending=False)
#club with their market value average

df_meanmv=pd.DataFrame(df.groupby(['club'])['market_value'].mean()).reset_index().sort_values('market_value',ascending=False)

plt.subplots(figsize=(15,6))

sns.barplot(x="club", y="market_value",data=df_meanmv,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Average of Market Value in Every Clubs')

plt.show()
#most view player

dfview = df.nlargest(10, 'page_views').sort_values('page_views',ascending=False)

plt.subplots(figsize=(15,6))

sns.barplot(x="name", y="page_views",  data=dfview ,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('top 10 most viewed premier league player season 2017/2018')

plt.show()
#fpl 10 most valuable player

dfview = df.nlargest(10, 'fpl_value').sort_values('fpl_value',ascending=False)

plt.subplots(figsize=(15,6))

sns.barplot(x="name", y="fpl_value",  data=dfview ,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('top 10 most valuable premier league player season 2017/2018')

plt.show()
#club with their average of fpl value

df_meanfv=pd.DataFrame(df.groupby(['club'])['fpl_value'].mean()).reset_index().sort_values('fpl_value',ascending=False)

plt.subplots(figsize=(15,6))

sns.barplot(x="club", y="fpl_value",data=df_meanfv,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Average of FPL Value in Every Clubs')

plt.show()
#club with their fpl selection

df_meanfs=pd.DataFrame(df.groupby(['club'])['fpl_sel'].sum()/0.14956).reset_index().sort_values('fpl_sel',ascending=False)

plt.subplots(figsize=(15,6))

sns.barplot(x="club", y="fpl_sel",data=df_meanfs,palette='hot',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Total Persentage of Club Selected by the user')

plt.show()
from iso3166 import countries

import iso3166

#countries.get(dftotal['Country'])

countlist= pd.DataFrame(iso3166.countries_by_alpha3).T.reset_index()



countlist = countlist[[0,2]]

countlist.rename(columns={0:'nationality',2:'code'},inplace=True)

countries.get('gbr')

#countlist
#Preparing data for geogrphical spread of  market value

df_meanmv=pd.DataFrame(df.groupby(['nationality'])['market_value'].mean()).reset_index()

df_meanmv = pd.merge(df_meanmv, countlist, on=['nationality', 'nationality'])

#df_meanmv.head(3)
data = dict(type='choropleth',

            locations=df_meanmv['code'],

            #locationmode='USA-states',

            text=df_meanmv['nationality'],

            z=df_meanmv['market_value'],

            autocolorscale = True,

            reversescale = False,

            ) 



layout = dict(

    title = 'Geographical distribution of market value average of premier league Players 2017/2018 Season',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)





choromap = go.Figure(data=[data], layout=layout)

#choromap = dict( data=data, layout=layout )

#plot.iplot( choromap, validate=False, filename='d3' )

py.iplot( choromap, filename='d3' )
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import collections



pd.options.display.max_columns = 999



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



import warnings

warnings.filterwarnings('ignore')











trace1 = {

  "x": df['fpl_value'],

  "y": df['fpl_points'],

  "marker": {

    "color": 'red',

    "colorsrc": "Aldemuro:22:1a1899",

    "size": df['fpl_ratio']

  }, 

  "mode": "markers", 

  "name": "fpl_points", 

  "text": df['name']+", Club:"+df['club']+", Pos:"+df['position'],

  #"textsrc": "Aldemuro:22:5dc54a", 

  "type": "scatter", 

  "uid": "0d217c", 

  "xsrc": "Aldemuro:22:d61533", 

  "ysrc": "Aldemuro:22:1c3243",

  

}

data = [trace1]

#data = [trace]

layout = {

  "autosize": True, 

  "hovermode": "closest",

  "title": "Relation between FPL points and FPL value from every players",

  "xaxis": {

    "autorange": True, 

    "range": [3.48535752785, 13.0146424722], 

    "title": "fpl value", 

    "type": "linear"

  }, 

  "yaxis": {

    "autorange": True, 

    "range": [-17.5245518316, 281.524551832], 

    "title": "fpl points", 

    "type": "linear"

  }

}

# Plot and embed in ipython notebook!

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-scatter')
#df['fage_pointration'] = pd.DataFrame(df['fpl_points']/df['fpl_value'])

trace1 = {

  "x": df['age'],

  "y": df['fpl_points'],

  "marker": {

    "color": 'red',

    "colorsrc": "Aldemuro:22:1a1899",

    #"size": df['fpl_ratio']

  }, 

  "mode": "markers", 

  "name": "fpl_points", 

  "text": df['name']+", Club:"+df['club']+", Pos:"+df['position'],

  #"textsrc": "Aldemuro:22:5dc54a", 

  "type": "scatter", 

  "uid": "0d217c", 

  "xsrc": "Aldemuro:22:d61533", 

  "ysrc": "Aldemuro:22:1c3243",

  

}

data = [trace1]

#data = [trace]

layout = {

  "autosize": True, 

  "hovermode": "closest",

  "title": "Relation between FPL points and player age",

  "xaxis": {

    "autorange": True, 

    "range": [3.48535752785, 13.0146424722], 

    "title": "player age", 

    "type": "linear"

  }, 

  "yaxis": {

    "autorange": True, 

    "range": [-17.5245518316, 281.524551832], 

    "title": "fpl points", 

    "type": "linear"

  }

}

# Plot and embed in ipython notebook!

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-scatter')
df.dtypes
#dfrel = pd.DataFrame(df['age'],df['position_cat'],df['market_value']).reset_index()

#frel = df.drop(['name', 'club','position','nationality','fpl_ratio','market_value','fpl_value','fpl_sel','region'],

#               axis=1)

dfrel = df[['age','page_views','fpl_points','fpl_value','market_value']].copy()

dfrel.head()
import seaborn as sns; sns.set(style="ticks", color_codes=True)

#iris = sns.load_dataset("iris")

g = sns.pairplot(dfrel)

plt.show()
dfcountry = pd.DataFrame(df.groupby('nationality')['nationality'].count())

dfcountry.rename(columns={'nationality':'count'},inplace=True)

dfcountry = dfcountry.reset_index()
dftotal = pd.merge(dfcountry, countlist, on=['nationality', 'nationality'])
data = dict(type='choropleth',

            locations=dftotal['code'],

            #locationmode='USA-states',

            text=dftotal['nationality'],

            z=dftotal['count'],

            autocolorscale = True,

            reversescale = False,

            ) 



layout = dict(

    title = 'Geographical distribution of Premier League Players 2017/2018 Season',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)





choromap = go.Figure(data=[data], layout=layout)

#choromap = dict( data=data, layout=layout )

#plot.iplot( choromap, validate=False, filename='d3' )

py.iplot( choromap, filename='d3' )