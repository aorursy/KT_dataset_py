import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud

import os

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt





import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
path = '../input/'

df = pd.read_csv(path + (os.listdir(path)[0]))
df.head()
plt.figure(figsize=(15,4))

#df.apply(lambda x: x.notnull().sum() / len(df)).plot(kind='barh')

sns.barplot(x=df.apply(lambda x: x.notnull().sum() / len(df)).index,

           y=df.apply(lambda x: x.notnull().sum() / len(df)).values)

plt.ylabel('Filling ratio')

plt.show()
df = df.assign(Cat=df.Categories.apply(lambda cat: str(cat).split(',')))
list_categories = list()

for k,row in df.iterrows():

    for i in row['Cat']:

        list_categories.append(i)
list_no_restaurant = [i.replace("Restaurants","",1) for i in list_categories]
cloud = WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=[' ', '']).generate(" ".join(list_no_restaurant))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
plt.figure(figsize=(15,4))

#df['State'].value_counts().plot(kind='barh')

sns.countplot(y=df['State'])

plt.show()
df2 = df.groupby(['State','City']).size()[df.groupby(['State','City']).size() > 10].unstack('State').fillna(0)

df2.plot(kind='bar', figsize=(15,4), stacked=True)

plt.show()
df_count = df.groupby('State', as_index=False).size().reset_index()

df_count.columns = ['State', 'Count']

df = df.merge(df_count, on='State')
data = [ dict(

        type = 'choropleth',

        locations = df['State'],

        locationmode='USA-states',

        z = df['Count'],

        #movie_titlext = df['State'].value_counts(), #df['movie_title'],

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            #tickprefix = '',

            title = 'Number of restaurants'),

      ) ]



layout = dict(

    title = 'Number of restaurants',

    geo=dict(

        countrycolor='rgb(102, 102, 102)',

        countrywidth=0.1,

        lakecolor='rgb(255, 255, 255)',

        landcolor='rgba(237, 247, 138, 0.28)',

        lonaxis=dict(

            gridwidth=1.5999999999999999,

            range=[-180, -50],

            showgrid=False

        ),

        projection=dict(

            type='albers usa'

        ),

        scope='usa',

        showland=True,

        showrivers=False,

        showsubunits=True,

        subunitcolor='rgb(102, 102, 102)',

        subunitwidth=0.5

    ),

    hovermode='closest',

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='d3-world-map' )
cloud = WordCloud(width=1440, height=1080, relative_scaling=0.5, stopwords=['Restaurant']).generate_from_frequencies(df.Name.value_counts())

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
